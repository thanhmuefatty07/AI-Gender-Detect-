#!/usr/bin/env python3
"""
Streamlit Monitoring Dashboard for Gender Classification Data Collection

Features:
- Real-time collection progress monitoring
- Dataset quality metrics and visualizations
- Collection statistics and analytics
- Data validation reports
- Interactive data exploration
- Export capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import time
from collections import defaultdict
import psutil
import os

# Add project root to path
project_root = Path(__file__).parent.parent

# Configure page
st.set_page_config(
    page_title="🎯 Gender Classification Data Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class DataMonitor:
    """Data collection monitoring and analytics"""

    def __init__(self):
        self.project_root = project_root
        self.data_dir = project_root / "datasets"
        self.logs_dir = project_root / "logs"
        self.config_path = project_root / "config" / "collector_config.yaml"

        # Load config
        self.config = self._load_config()

    def _load_config(self):
        """Load collector configuration"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Failed to load config: {e}")
            return {}

    def get_collection_stats(self) -> Dict:
        """Get comprehensive collection statistics"""
        stats = {
            'total_videos': 0,
            'total_faces': 0,
            'total_audio': 0,
            'datasets': {},
            'quality_metrics': {},
            'recent_activity': [],
            'system_info': self._get_system_info()
        }

        # Count files in each collected directory
        collected_dir = self.data_dir / "collected"

        for platform in ['youtube', 'tiktok', 'instagram']:
            platform_dir = collected_dir / platform
            if platform_dir.exists():
                raw_videos = len(list(platform_dir.glob("raw_videos/*.mp4")))
                processed_items = len(list(platform_dir.glob("processed/*"))) if (platform_dir / "processed").exists() else 0

                stats['datasets'][platform] = {
                    'raw_videos': raw_videos,
                    'processed': processed_items
                }

                stats['total_videos'] += raw_videos

        # Count academic datasets
        academic_dir = self.data_dir / "academic"
        if academic_dir.exists():
            merged_files = list(academic_dir.glob("merged/*.csv"))
            if merged_files:
                latest_merge = max(merged_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(latest_merge)
                stats['datasets']['academic'] = {
                    'total_samples': len(df),
                    'datasets_count': df['dataset'].nunique() if 'dataset' in df.columns else 0
                }

        # Load metadata files for detailed stats
        metadata_files = list(self.data_dir.glob("**/metadata_*.json"))
        if metadata_files:
            latest_meta = max(metadata_files, key=lambda x: x.stat().st_mtime)

            try:
                with open(latest_meta, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Extract statistics
                df_meta = pd.DataFrame(metadata)
                stats['total_faces'] = len(df_meta)
                stats['quality_metrics'] = self._calculate_quality_metrics(df_meta)

            except Exception as e:
                st.warning(f"Could not load metadata: {e}")

        # Get recent logs
        stats['recent_activity'] = self._get_recent_logs()

        return stats

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate quality metrics from metadata"""
        metrics = {}

        if 'quality_score' in df.columns:
            metrics['avg_quality'] = df['quality_score'].mean()
            metrics['quality_distribution'] = df['quality_score'].value_counts(bins=5).to_dict()

        if 'gender' in df.columns:
            gender_dist = df['gender'].value_counts().to_dict()
            metrics['gender_balance'] = gender_dist
            total = sum(gender_dist.values())
            metrics['gender_ratio'] = {k: v/total for k, v in gender_dist.items()}

        if 'file_size' in df.columns:
            metrics['avg_file_size'] = df['file_size'].mean()
            metrics['total_size'] = df['file_size'].sum()

        return metrics

    def _get_recent_logs(self, hours: int = 24) -> List[Dict]:
        """Get recent collection activity from logs"""
        activities = []

        if not self.logs_dir.exists():
            return activities

        cutoff_time = datetime.now() - timedelta(hours=hours)

        for log_file in self.logs_dir.glob("*.log"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-50:]  # Last 50 lines

                for line in lines:
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            timestamp_str = parts[0].strip()
                            try:
                                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                                if timestamp > cutoff_time:
                                    activities.append({
                                        'timestamp': timestamp,
                                        'level': parts[1].strip(),
                                        'message': parts[2].strip(),
                                        'source': log_file.stem
                                    })
                            except ValueError:
                                continue

            except Exception:
                continue

        # Sort by timestamp
        activities.sort(key=lambda x: x['timestamp'], reverse=True)
        return activities[:20]  # Return 20 most recent

    def _get_system_info(self) -> Dict:
        """Get system resource information"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_free': psutil.disk_usage('/').free / (1024**3),  # GB
            'timestamp': datetime.now().isoformat()
        }

    def get_data_visualizations(self):
        """Generate data visualization plots"""
        plots = {}

        # Load latest metadata
        metadata_files = list(self.data_dir.glob("**/metadata_*.json"))
        if metadata_files:
            latest_meta = max(metadata_files, key=lambda x: x.stat().st_mtime)

            try:
                with open(latest_meta, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                df = pd.DataFrame(metadata)

                # Gender distribution
                if 'gender' in df.columns:
                    gender_counts = df['gender'].value_counts()
                    plots['gender_dist'] = px.pie(
                        values=gender_counts.values,
                        names=gender_counts.index,
                        title="Gender Distribution"
                    )

                # Quality score distribution
                if 'quality_score' in df.columns:
                    plots['quality_dist'] = px.histogram(
                        df, x='quality_score',
                        title="Quality Score Distribution",
                        nbins=20
                    )

                # Source distribution
                if 'source' in df.columns:
                    source_counts = df['source'].value_counts()
                    plots['source_dist'] = px.bar(
                        x=source_counts.index,
                        y=source_counts.values,
                        title="Data Sources Distribution"
                    )

                # File size distribution
                if 'file_size' in df.columns:
                    df_plot = df.copy()
                    df_plot['file_size_mb'] = df_plot['file_size'] / (1024 * 1024)
                    plots['size_dist'] = px.box(
                        df_plot, y='file_size_mb',
                        title="File Size Distribution (MB)"
                    )

            except Exception as e:
                st.warning(f"Could not generate plots: {e}")

        return plots


def main():
    """Main dashboard function"""

    # Initialize monitor
    monitor = DataMonitor()

    # Header
    st.markdown('<h1 class="main-header">🎯 Gender Classification Data Monitor</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")

        # Refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()

        # Auto refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            ["Last 24h", "Last 7 days", "Last 30 days"],
            index=0
        )

        st.markdown("---")

        # System info
        st.subheader("🖥️ System Status")
        sys_info = monitor._get_system_info()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU", f"{sys_info['cpu_percent']:.1f}%")
            st.metric("Memory", f"{sys_info['memory_percent']:.1f}%")
        with col2:
            st.metric("Disk Free", f"{sys_info['disk_free']:.1f} GB")

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Analytics", "🎯 Quality", "📝 Logs"])

    # Tab 1: Overview
    with tab1:
        st.header("📊 Collection Overview")

        # Get stats
        stats = monitor.get_collection_stats()

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Videos", f"{stats['total_videos']:,}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Processed Faces", f"{stats['total_faces']:,}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            quality = stats['quality_metrics'].get('avg_quality', 0)
            st.metric("Avg Quality", f"{quality:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            academic_count = stats['datasets'].get('academic', {}).get('total_samples', 0)
            st.metric("Academic Samples", f"{academic_count:,}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Dataset breakdown
        st.subheader("📁 Dataset Breakdown")

        datasets_df = pd.DataFrame([
            {
                'Source': source,
                'Videos': data.get('raw_videos', 0),
                'Processed': data.get('processed', 0),
                'Total': data.get('raw_videos', 0) + data.get('processed', 0)
            }
            for source, data in stats['datasets'].items()
            if source != 'academic'
        ])

        if not datasets_df.empty:
            st.dataframe(datasets_df, use_container_width=True)

            # Progress bars
            for _, row in datasets_df.iterrows():
                if row['Videos'] > 0:
                    progress = min(row['Processed'] / row['Videos'], 1.0)
                    st.progress(progress, text=f"{row['Source']} Processing Progress")

    # Tab 2: Analytics
    with tab2:
        st.header("📈 Data Analytics")

        plots = monitor.get_data_visualizations()

        if plots:
            # Create columns for plots
            col1, col2 = st.columns(2)

            with col1:
                if 'gender_dist' in plots:
                    st.plotly_chart(plots['gender_dist'], use_container_width=True)

                if 'source_dist' in plots:
                    st.plotly_chart(plots['source_dist'], use_container_width=True)

            with col2:
                if 'quality_dist' in plots:
                    st.plotly_chart(plots['quality_dist'], use_container_width=True)

                if 'size_dist' in plots:
                    st.plotly_chart(plots['size_dist'], use_container_width=True)
        else:
            st.info("No data available for visualization. Start collecting data first!")

    # Tab 3: Quality
    with tab3:
        st.header("🎯 Data Quality Assessment")

        stats = monitor.get_collection_stats()
        quality = stats['quality_metrics']

        if quality:
            # Quality metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_quality = quality.get('avg_quality', 0)
                quality_status = "🟢 Good" if avg_quality > 0.7 else "🟡 Fair" if avg_quality > 0.5 else "🔴 Poor"
                st.metric("Average Quality", f"{avg_quality:.2f}", quality_status)

            with col2:
                gender_balance = quality.get('gender_ratio', {})
                if gender_balance:
                    male_ratio = gender_balance.get('male', 0)
                    female_ratio = gender_balance.get('female', 0)
                    balance_score = 1 - abs(male_ratio - female_ratio)
                    st.metric("Gender Balance", f"{balance_score:.2f}")

            with col3:
                avg_size = quality.get('avg_file_size', 0) / (1024 * 1024)  # MB
                st.metric("Avg File Size", f"{avg_size:.1f} MB")

            # Quality distribution
            if 'quality_distribution' in quality:
                st.subheader("Quality Score Distribution")
                quality_bins = quality['quality_distribution']

                # Create bar chart
                bins = list(quality_bins.keys())
                counts = list(quality_bins.values())

                fig = px.bar(
                    x=[f"Bin {i+1}" for i in range(len(bins))],
                    y=counts,
                    title="Quality Score Distribution",
                    labels={'x': 'Quality Range', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No quality metrics available yet.")

    # Tab 4: Logs
    with tab4:
        st.header("📝 Recent Activity Logs")

        activities = stats['recent_activity']

        if activities:
            # Create dataframe for display
            logs_df = pd.DataFrame(activities)

            # Color coding for log levels
            def colorize_level(level):
                colors = {
                    'INFO': 'green',
                    'WARNING': 'orange',
                    'ERROR': 'red',
                    'SUCCESS': 'blue'
                }
                color = colors.get(level.upper(), 'black')
                return f'<span style="color: {color}; font-weight: bold;">{level}</span>'

            logs_df['level_colored'] = logs_df['level'].apply(colorize_level)

            # Display logs
            for _, row in logs_df.iterrows():
                timestamp = row['timestamp'].strftime("%H:%M:%S")
                st.markdown(f"**{timestamp}** | {row['level_colored']} | {row['message']}",
                          unsafe_allow_html=True)
        else:
            st.info("No recent logs found.")

    # Footer
    st.markdown("---")
    st.markdown("*Dashboard last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")

    # Auto refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()

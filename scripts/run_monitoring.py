#!/usr/bin/env python3
"""
Script to run the Streamlit monitoring dashboard
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the monitoring dashboard"""

    # Get project root
    project_root = Path(__file__).parent.parent
    dashboard_path = project_root / "app" / "monitoring_dashboard.py"

    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        sys.exit(1)

    print("🚀 Starting Gender Classification Data Monitor...")
    print(f"📊 Dashboard URL: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port", "8501"]
        subprocess.run(cmd, cwd=str(project_root))

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


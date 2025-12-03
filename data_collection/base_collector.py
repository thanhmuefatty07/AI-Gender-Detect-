"""
Base class cho mọi collectors - Template Method Pattern Implementation

Tất cả collectors đều kế thừa từ class này và implement các abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger
import yaml
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time
from tqdm import tqdm
import hashlib


@dataclass
class CollectionMetadata:
    """Standardized metadata for all collected items"""
    source: str
    item_id: str
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[int] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    collected_at: Optional[str] = None
    quality_score: Optional[float] = None
    inferred_gender: Optional[str] = None
    inferred_age: Optional[int] = None
    tags: List[str] = None
    extra_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.extra_metadata is None:
            self.extra_metadata = {}
        if self.collected_at is None:
            self.collected_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CollectionMetadata':
        """Create from dictionary"""
        return cls(**data)


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors

    Implements Template Method Pattern:
    1. search() - Find content
    2. download() - Download content
    3. validate() - Check quality
    4. process() - Extract features
    5. save_metadata() - Save results
    """

    def __init__(self, config_path: str = "config/collector_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Setup paths
        self.project_root = Path(__file__).parent.parent
        self.output_dir = Path(self.config['output']['base_path'])
        self.logs_dir = self.project_root / "logs"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Rate limiting
        self.last_request_time = 0
        self.rate_limits = self.config.get('rate_limiting', {}).get(self.__class__.__name__.lower(), {})

        logger.info(f"✅ Initialized {self.__class__.__name__}")

    def _load_config(self) -> Dict:
        """Load YAML config with environment variable substitution"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Substitute environment variables (e.g., ${YOUTUBE_API_KEY})
        config = self._substitute_env_vars(config)

        return config

    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in config"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]  # Remove ${}
            return os.getenv(env_var, config)  # Return original if not found
        else:
            return config

    def setup_logging(self):
        """Configure structured logging with rotation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"collector_{self.__class__.__name__.lower()}_{timestamp}.log"

        logger.remove()  # Remove default handler
        logger.add(
            log_file,
            rotation="500 MB",
            retention="10 days",
            level=self.config.get('monitoring', {}).get('log_level', 'INFO'),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            level="INFO",
            format="{time:HH:mm:ss} | {level} | {message}"
        )

    def _rate_limit_wait(self):
        """Implement rate limiting"""
        if not self.rate_limits:
            return

        requests_per_minute = self.rate_limits.get('requests_per_minute', 60)
        min_interval = 60 / requests_per_minute

        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(".2f")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    @abstractmethod
    def search(self, query: str, max_results: int) -> List[Dict]:
        """Search for content matching query"""
        pass

    @abstractmethod
    def download(self, item: Dict) -> Optional[Path]:
        """Download a single item"""
        pass

    @abstractmethod
    def validate(self, file_path: Path) -> bool:
        """Check if downloaded file is valid"""
        pass

    @abstractmethod
    def process(self, file_path: Path, metadata: Dict) -> Dict:
        """Process downloaded file (extract faces, audio, etc.)"""
        pass

    def collect(self, queries: List[str], max_per_query: int = 50) -> List[CollectionMetadata]:
        """
        Main collection pipeline (Template Method)

        1. Search for each query
        2. Download valid items
        3. Validate downloads
        4. Process and extract features
        5. Save metadata
        """
        all_metadata = []

        for query in queries:
            logger.info(f"🔍 Searching for: '{query}'")

            try:
                # Step 1: Search
                items = self.search(query, max_per_query)
                logger.info(f"📋 Found {len(items)} items for query: '{query}'")

                if not items:
                    continue

                # Step 2-5: Download, validate, process each item
                processed_items = []
                for item in tqdm(items, desc=f"Processing {query[:30]}..."):

                    try:
                        # Rate limiting
                        self._rate_limit_wait()

                        # Download
                        file_path = self.download(item)
                        if not file_path:
                            continue

                        # Validate
                        if not self.validate(file_path):
                            logger.warning(f"❌ Validation failed: {item.get('id', 'unknown')}")
                            file_path.unlink()  # Delete invalid file
                            continue

                        # Process (extract features)
                        processed_data = self.process(file_path, item)

                        # Create standardized metadata
                        metadata = CollectionMetadata(
                            source=self.__class__.__name__,
                            item_id=item.get('id', self._generate_id(item)),
                            url=item.get('url', ''),
                            title=item.get('title'),
                            description=item.get('description'),
                            duration=item.get('duration'),
                            file_path=str(file_path),
                            file_size=file_path.stat().st_size if file_path.exists() else 0,
                            inferred_gender=item.get('inferred_gender'),
                            quality_score=processed_data.get('quality_score'),
                            tags=item.get('tags', []),
                            extra_metadata={
                                'processed_data': processed_data,
                                'original_metadata': item
                            }
                        )

                        processed_items.append(metadata)
                        logger.success(f"✅ Processed: {file_path.name}")

                    except Exception as e:
                        logger.error(f"❌ Error processing item {item.get('id', 'unknown')}: {e}")
                        continue

                all_metadata.extend(processed_items)
                logger.info(f"📊 Query '{query}' completed: {len(processed_items)}/{len(items)} successful")

            except Exception as e:
                logger.error(f"❌ Error in query '{query}': {e}")
                continue

        # Save all metadata
        if all_metadata:
            self._save_metadata(all_metadata)

        logger.info(f"🎯 Collection completed: {len(all_metadata)} items collected")
        return all_metadata

    def _generate_id(self, item: Dict) -> str:
        """Generate unique ID for item if not provided"""
        content = f"{item.get('url', '')}{item.get('title', '')}{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _save_metadata(self, metadata: List[CollectionMetadata]):
        """Save collection metadata to JSON with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"metadata_{self.__class__.__name__.lower()}_{timestamp}.json"

        # Convert to dicts for JSON serialization
        metadata_dicts = [meta.to_dict() for meta in metadata]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_dicts, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Metadata saved: {output_file} ({len(metadata)} items)")

        # Also save summary statistics
        self._save_summary_stats(metadata, timestamp)

    def _save_summary_stats(self, metadata: List[CollectionMetadata], timestamp: str):
        """Save summary statistics"""
        stats = {
            'timestamp': timestamp,
            'source': self.__class__.__name__,
            'total_items': len(metadata),
            'avg_file_size': sum(m.file_size or 0 for m in metadata) / len(metadata) if metadata else 0,
            'gender_distribution': {},
            'quality_scores': []
        }

        for meta in metadata:
            gender = meta.inferred_gender or 'unknown'
            stats['gender_distribution'][gender] = stats['gender_distribution'].get(gender, 0) + 1

            if meta.quality_score is not None:
                stats['quality_scores'].append(meta.quality_score)

        stats['avg_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores']) if stats['quality_scores'] else 0

        stats_file = self.output_dir / f"stats_{self.__class__.__name__.lower()}_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def get_status(self) -> Dict:
        """Get current collection status"""
        return {
            'source': self.__class__.__name__,
            'config_loaded': bool(self.config),
            'output_dir': str(self.output_dir),
            'last_request_time': self.last_request_time,
            'rate_limits': self.rate_limits
        }

    def cleanup(self):
        """Cleanup resources"""
        logger.info(f"🧹 Cleaning up {self.__class__.__name__}")
        # Override in subclasses if needed


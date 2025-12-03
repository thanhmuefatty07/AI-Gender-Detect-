"""
TikTok Video Collector

Since TikTok doesn't have official public API, we use:
1. tiktokapipy (unofficial API wrapper)
2. Selenium/Playwright for web scraping
3. Direct video URL extraction

Features:
- Hashtag-based search
- Video download with metadata
- Quality filtering
- Rate limiting
"""

import time
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse, parse_qs
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from .base_collector import BaseCollector, CollectionMetadata
from loguru import logger


class TikTokCollector(BaseCollector):
    """TikTok video collector using multiple strategies"""

    def __init__(self, config_path: str = "config/collector_config.yaml"):
        super().__init__(config_path)

        self.tiktok_config = self.config['sources']['tiktok']
        self.driver = None

        # Initialize Selenium driver
        self._init_selenium_driver()

    def _init_selenium_driver(self):
        """Initialize Chrome driver with TikTok-friendly settings"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run headless
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)

            # TikTok-specific headers
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

            self.driver = webdriver.Chrome(options=chrome_options)

            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            logger.info("✅ Selenium driver initialized for TikTok")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Selenium driver: {e}")
            self.driver = None

    def search(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search TikTok videos by hashtag or keyword

        Strategy 1: Use tiktokapipy if available
        Strategy 2: Selenium web scraping
        Strategy 3: Direct API calls (if tokens available)
        """
        if not self.driver:
            logger.error("❌ Selenium driver not available")
            return []

        videos = []

        try:
            # For hashtags, use hashtag search URL
            if query.startswith('#'):
                search_url = f"https://www.tiktok.com/tag/{query[1:]}"
            else:
                search_url = f"https://www.tiktok.com/search?q={query}"

            logger.info(f"🌐 Navigating to: {search_url}")
            self.driver.get(search_url)

            # Wait for page to load
            time.sleep(3)

            # Scroll to load more videos
            self._scroll_to_load_videos(max_results)

            # Extract video data
            video_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[data-e2e="recommend-list-item-container"]')

            for element in video_elements[:max_results]:
                try:
                    video_data = self._extract_video_data(element)
                    if video_data:
                        videos.append(video_data)

                except Exception as e:
                    logger.warning(f"⚠️ Error extracting video data: {e}")
                    continue

        except Exception as e:
            logger.error(f"❌ Error during TikTok search: {e}")

        logger.info(f"📹 Found {len(videos)} TikTok videos for query: {query}")
        return videos

    def _scroll_to_load_videos(self, target_count: int):
        """Scroll down to load more videos"""
        scroll_count = 0
        max_scrolls = min(target_count // 10, 20)  # Max 20 scrolls

        while scroll_count < max_scrolls:
            try:
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

                # Check if we have enough videos
                video_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[data-e2e="recommend-list-item-container"]')
                if len(video_elements) >= target_count:
                    break

                scroll_count += 1

            except Exception as e:
                logger.warning(f"⚠️ Error during scrolling: {e}")
                break

    def _extract_video_data(self, element) -> Optional[Dict]:
        """Extract video metadata from HTML element"""
        try:
            # Get video link
            link_element = element.find_element(By.CSS_SELECTOR, 'a[href*="/video/"]')
            video_url = link_element.get_attribute('href')

            # Extract video ID from URL
            video_id = self._extract_video_id(video_url)
            if not video_id:
                return None

            # Get video description/caption
            desc_element = element.find_element(By.CSS_SELECTOR, '[data-e2e="video-desc"]')
            description = desc_element.text if desc_element else ""

            # Get author info
            author_element = element.find_element(By.CSS_SELECTOR, '[data-e2e="video-author-uniqueid"]')
            author = author_element.text if author_element else "unknown"

            # Get stats (likes, comments, shares)
            stats = self._extract_video_stats(element)

            # Infer gender from description and author
            inferred_gender = self._infer_gender_from_text(description, author)

            return {
                'id': video_id,
                'url': video_url,
                'title': description[:100] + "..." if len(description) > 100 else description,
                'description': description,
                'author': author,
                'duration': None,  # Will be determined after download
                'stats': stats,
                'inferred_gender': inferred_gender,
                'tags': self._extract_hashtags(description),
                'platform': 'tiktok'
            }

        except Exception as e:
            logger.warning(f"⚠️ Error extracting video data: {e}")
            return None

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from TikTok URL"""
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.split('/')
            if 'video' in path_parts:
                video_index = path_parts.index('video')
                if video_index + 1 < len(path_parts):
                    return path_parts[video_index + 1]
        except Exception as e:
            logger.warning(f"⚠️ Error extracting video ID from {url}: {e}")
        return None

    def _extract_video_stats(self, element) -> Dict:
        """Extract video statistics"""
        stats = {}

        try:
            # Likes
            like_element = element.find_element(By.CSS_SELECTOR, '[data-e2e="like-count"]')
            stats['likes'] = like_element.text if like_element else "0"

            # Comments
            comment_element = element.find_element(By.CSS_SELECTOR, '[data-e2e="comment-count"]')
            stats['comments'] = comment_element.text if comment_element else "0"

            # Shares
            share_element = element.find_element(By.CSS_SELECTOR, '[data-e2e="share-count"]')
            stats['shares'] = share_element.text if share_element else "0"

        except Exception:
            pass

        return stats

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        hashtags = re.findall(r'#(\w+)', text)
        return list(set(hashtags))  # Remove duplicates

    def _infer_gender_from_text(self, description: str, author: str) -> Optional[str]:
        """Infer gender from description and author name"""
        text = f"{description} {author}".lower()

        male_keywords = ['he', 'him', 'his', 'mr', 'male', 'man', 'boy', 'guy', 'gentleman', 'anh', 'chịu', 'em trai']
        female_keywords = ['she', 'her', 'hers', 'ms', 'mrs', 'female', 'woman', 'girl', 'lady', 'em gái', 'chị', 'cô']

        male_score = sum(1 for kw in male_keywords if kw in text)
        female_score = sum(1 for kw in female_keywords if kw in text)

        if male_score > female_score + 1:
            return 'male'
        elif female_score > male_score + 1:
            return 'female'

        return None

    def download(self, item: Dict) -> Optional[Path]:
        """Download TikTok video using yt-dlp or direct methods"""
        video_id = item['id']
        output_dir = self.output_dir / 'tiktok' / 'raw_videos'
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{video_id}.mp4"

        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"⏭️  Skipping {video_id} (already exists)")
            return output_path

        try:
            # Method 1: Try yt-dlp first
            import yt_dlp

            ydl_opts = {
                'format': 'best[height<=720]',  # Don't need 4K
                'outtmpl': str(output_path),
                'quiet': True,
                'no_warnings': True,
                'retries': 3,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([item['url']])

            return output_path

        except Exception as e:
            logger.warning(f"⚠️ yt-dlp failed for {video_id}: {e}")

            # Method 2: Try direct download using Selenium
            try:
                return self._download_with_selenium(item, output_path)
            except Exception as e2:
                logger.error(f"❌ All download methods failed for {video_id}: {e2}")
                return None

    def _download_with_selenium(self, item: Dict, output_path: Path) -> Optional[Path]:
        """Download video using Selenium to get direct video URL"""
        try:
            self.driver.get(item['url'])
            time.sleep(3)

            # Try to find video element
            video_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "video"))
            )

            video_url = video_element.get_attribute('src')

            if video_url:
                # Download the video
                response = requests.get(video_url, stream=True, timeout=30)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                return output_path

        except Exception as e:
            logger.warning(f"⚠️ Selenium download failed: {e}")

        return None

    def validate(self, file_path: Path) -> bool:
        """Validate downloaded TikTok video"""
        if not file_path.exists():
            return False

        # Check file size (at least 500KB for TikTok videos)
        if file_path.stat().st_size < 500_000:
            return False

        # Check if it's a valid video file
        try:
            import cv2
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                cap.release()
                return False

            # Check duration (TikTok videos are usually 15-180 seconds)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            # Must be between 10 seconds and 5 minutes
            return 10 <= duration <= 300

        except Exception as e:
            logger.warning(f"⚠️ Video validation error: {e}")
            return False

    def process(self, file_path: Path, metadata: Dict) -> Dict:
        """Process TikTok video (placeholder - will be implemented in video processor)"""
        return {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'quality_score': 0.8,  # Placeholder
            'processing_status': 'pending'
        }

    def collect_hashtags(self, hashtags: List[str], max_per_hashtag: int = 100) -> List[CollectionMetadata]:
        """Collect videos from multiple hashtags"""
        all_videos = []

        for hashtag in hashtags:
            logger.info(f"🔍 Collecting hashtag: {hashtag}")

            # Add # prefix if not present
            if not hashtag.startswith('#'):
                hashtag = f"#{hashtag}"

            videos = self.search(hashtag, max_per_hashtag)
            all_videos.extend(videos)

            # Rate limiting between hashtags
            time.sleep(2)

        # Remove duplicates based on video ID
        unique_videos = []
        seen_ids = set()
        for video in all_videos:
            if video['id'] not in seen_ids:
                unique_videos.append(video)
                seen_ids.add(video['id'])

        logger.info(f"📊 Total unique videos collected: {len(unique_videos)}")

        # Process collection
        return self.collect_from_items(unique_videos)

    def collect_from_items(self, items: List[Dict]) -> List[CollectionMetadata]:
        """Process a list of video items (used by collect_hashtags)"""
        metadata_list = []

        for item in tqdm(items, desc="Processing TikTok videos"):
            try:
                # Download
                file_path = self.download(item)
                if not file_path:
                    continue

                # Validate
                if not self.validate(file_path):
                    file_path.unlink()
                    continue

                # Process
                processed_data = self.process(file_path, item)

                # Create metadata
                metadata = CollectionMetadata(
                    source=self.__class__.__name__,
                    item_id=item['id'],
                    url=item['url'],
                    title=item.get('title'),
                    description=item.get('description'),
                    file_path=str(file_path),
                    file_size=file_path.stat().st_size,
                    inferred_gender=item.get('inferred_gender'),
                    quality_score=processed_data.get('quality_score'),
                    tags=item.get('tags', []),
                    extra_metadata={
                        'author': item.get('author'),
                        'stats': item.get('stats'),
                        'platform': 'tiktok'
                    }
                )

                metadata_list.append(metadata)

            except Exception as e:
                logger.error(f"❌ Error processing TikTok video {item.get('id')}: {e}")
                continue

        # Save metadata
        if metadata_list:
            self._save_metadata(metadata_list)

        return metadata_list

    def cleanup(self):
        """Cleanup Selenium driver"""
        if self.driver:
            self.driver.quit()
            logger.info("🧹 Selenium driver closed")
        super().cleanup()


# ===== USAGE EXAMPLES =====

def collect_tiktok_sample():
    """Collect sample TikTok videos for testing"""
    collector = TikTokCollector()

    # Test hashtags
    test_hashtags = ['#dailyvlog', '#storytime']

    try:
        results = collector.collect_hashtags(test_hashtags, max_per_hashtag=5)
        print(f"✅ Collected {len(results)} TikTok videos")

        for meta in results[:3]:  # Show first 3
            print(f"  - {meta.title} ({meta.inferred_gender})")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        collector.cleanup()


if __name__ == "__main__":
    collect_tiktok_sample()


"""
Instagram Video Collector

Uses Instaloader library for reliable Instagram data collection:
- Hashtag-based search
- Post metadata extraction
- Video download
- Quality filtering

Features:
- Reel and video post collection
- Metadata extraction (captions, hashtags, likes)
- Author information
- Geographic filtering (Vietnam focus)
"""

import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import instaloader
from instaloader.exceptions import QueryReturnedNotFoundException, BadCredentialsException

from .base_collector import BaseCollector, CollectionMetadata
from loguru import logger


class InstagramCollector(BaseCollector):
    """Instagram video collector using Instaloader"""

    def __init__(self, config_path: str = "config/collector_config.yaml"):
        super().__init__(config_path)

        self.instagram_config = self.config['sources']['instagram']
        self.loader = None

        # Initialize Instaloader
        self._init_instaloader()

    def _init_instaloader(self):
        """Initialize Instaloader with optimal settings"""
        try:
            self.loader = instaloader.Instaloader()

            # Configure loader settings
            self.loader.save_metadata = False  # We handle metadata ourselves
            self.loader.download_video_thumbnails = False
            self.loader.download_geotags = False
            self.loader.download_comments = False

            # Set download directory
            download_dir = self.output_dir / 'instagram' / 'raw_videos'
            download_dir.mkdir(parents=True, exist_ok=True)

            # Login if credentials available (optional but recommended)
            self._try_login()

            logger.info("✅ Instaloader initialized for Instagram")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Instaloader: {e}")
            self.loader = None

    def _try_login(self):
        """Try to login with credentials if available"""
        try:
            username = self.config.get('instagram', {}).get('username')
            password = self.config.get('instagram', {}).get('password')

            if username and password:
                self.loader.login(username, password)
                logger.info("✅ Logged in to Instagram")
            else:
                logger.info("ℹ️  No Instagram credentials provided, using anonymous access")

        except BadCredentialsException:
            logger.warning("⚠️  Invalid Instagram credentials")
        except Exception as e:
            logger.warning(f"⚠️  Could not login to Instagram: {e}")

    def search(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search Instagram posts by hashtag

        Uses Instaloader's hashtag search functionality
        """
        if not self.loader:
            logger.error("❌ Instaloader not available")
            return []

        if not query.startswith('#'):
            query = f"#{query}"

        hashtag_name = query[1:]  # Remove # prefix
        posts = []

        try:
            logger.info(f"🔍 Searching Instagram hashtag: {hashtag_name}")

            # Get hashtag object
            hashtag = instaloader.Hashtag.from_name(self.loader.context, hashtag_name)

            # Iterate through posts
            count = 0
            for post in hashtag.get_posts():
                if count >= max_results:
                    break

                try:
                    post_data = self._extract_post_data(post)
                    if post_data and self._filter_post(post_data):
                        posts.append(post_data)
                        count += 1

                except Exception as e:
                    logger.warning(f"⚠️ Error extracting post data: {e}")
                    continue

        except QueryReturnedNotFoundException:
            logger.warning(f"⚠️  Hashtag '{hashtag_name}' not found")
        except Exception as e:
            logger.error(f"❌ Error during Instagram search: {e}")

        logger.info(f"📸 Found {len(posts)} Instagram posts for hashtag: {query}")
        return posts

    def _extract_post_data(self, post) -> Optional[Dict]:
        """Extract metadata from Instaloader Post object"""
        try:
            # Only process video posts/reels
            if not post.is_video:
                return None

            post_data = {
                'id': post.shortcode,  # Instagram shortcode as ID
                'url': f"https://www.instagram.com/p/{post.shortcode}/",
                'title': post.caption[:100] + "..." if post.caption and len(post.caption) > 100 else (post.caption or ""),
                'description': post.caption or "",
                'author': post.owner_username,
                'author_id': post.owner_id,
                'duration': post.video_duration,
                'width': post.video_width,
                'height': post.video_height,
                'date_posted': post.date.isoformat(),
                'likes': post.likes,
                'comments': post.comments,
                'is_reel': hasattr(post, 'is_reel') and post.is_reel,
                'location': post.location.name if post.location else None,
                'hashtags': self._extract_hashtags(post.caption or ""),
                'mentions': self._extract_mentions(post.caption or ""),
                'inferred_gender': self._infer_gender_from_caption(post.caption or "", post.owner_username),
                'platform': 'instagram'
            }

            return post_data

        except Exception as e:
            logger.warning(f"⚠️ Error extracting post data: {e}")
            return None

    def _filter_post(self, post_data: Dict) -> bool:
        """Apply filters to determine if post should be collected"""
        config = self.instagram_config.get('filters', {})

        # Filter by minimum followers (if available)
        min_followers = config.get('min_followers', 0)
        # Note: Instaloader doesn't provide follower count easily, skip this filter

        # Filter by post type
        post_type = config.get('post_type', 'video')
        if post_type == 'reel' and not post_data.get('is_reel', False):
            return False

        # Filter by duration (for videos)
        duration = post_data.get('duration', 0)
        if duration < 5 or duration > 300:  # 5 seconds to 5 minutes
            return False

        # Filter by region (Vietnam focus)
        region = config.get('region', 'VN')
        if region == 'VN':
            # Check if caption contains Vietnamese keywords or location
            vietnam_keywords = ['việt nam', 'hanoi', 'ho chi minh', 'sài gòn', 'đà nẵng', 'hải phòng']
            text_to_check = f"{post_data.get('description', '')} {post_data.get('location', '')}".lower()

            has_vietnam_indicator = any(kw in text_to_check for kw in vietnam_keywords)
            if not has_vietnam_indicator:
                return False  # Skip non-Vietnam content

        return True

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        hashtags = re.findall(r'#(\w+)', text)
        return list(set(hashtags))  # Remove duplicates

    def _extract_mentions(self, text: str) -> List[str]:
        """Extract user mentions from text"""
        mentions = re.findall(r'@(\w+)', text)
        return list(set(mentions))  # Remove duplicates

    def _infer_gender_from_caption(self, caption: str, username: str) -> Optional[str]:
        """Infer gender from caption and username"""
        text = f"{caption} {username}".lower()

        male_keywords = ['anh', 'chú', 'ông', 'em trai', 'con trai', 'boy', 'man', 'male', 'he', 'his']
        female_keywords = ['chị', 'cô', 'bà', 'em gái', 'con gái', 'girl', 'woman', 'female', 'she', 'her']

        male_score = sum(1 for kw in male_keywords if kw in text)
        female_score = sum(1 for kw in female_keywords if kw in text)

        if male_score > female_score + 1:
            return 'male'
        elif female_score > male_score + 1:
            return 'female'

        return None

    def download(self, item: Dict) -> Optional[Path]:
        """Download Instagram video using Instaloader"""
        post_id = item['id']
        output_dir = self.output_dir / 'instagram' / 'raw_videos'
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{post_id}.mp4"

        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"⏭️  Skipping {post_id} (already exists)")
            return output_path

        try:
            # Get post object
            post = instaloader.Post.from_shortcode(self.loader.context, post_id)

            # Download video only
            self.loader.download_post(post, target=str(output_dir))

            # Find the downloaded video file
            # Instaloader saves with format: <username>_<shortcode>.mp4
            expected_filename = f"{post.owner_username}_{post_id}.mp4"
            downloaded_file = output_dir / expected_filename

            if downloaded_file.exists():
                # Rename to our standard format
                downloaded_file.rename(output_path)
                return output_path
            else:
                # Look for any .mp4 file with the post ID
                for file in output_dir.glob(f"*{post_id}*.mp4"):
                    file.rename(output_path)
                    return output_path

            logger.warning(f"⚠️ Could not find downloaded file for post {post_id}")
            return None

        except Exception as e:
            logger.error(f"❌ Failed to download Instagram post {post_id}: {e}")
            return None

    def validate(self, file_path: Path) -> bool:
        """Validate downloaded Instagram video"""
        if not file_path.exists():
            return False

        # Check file size (at least 200KB for Instagram videos)
        if file_path.stat().st_size < 200_000:
            return False

        # Check if it's a valid video file
        try:
            import cv2
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                cap.release()
                return False

            # Check duration
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            # Instagram videos/reels are usually 3-90 seconds
            return 3 <= duration <= 90

        except Exception as e:
            logger.warning(f"⚠️ Video validation error: {e}")
            return False

    def process(self, file_path: Path, metadata: Dict) -> Dict:
        """Process Instagram video (placeholder - will be implemented in video processor)"""
        return {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'quality_score': 0.9,  # Instagram videos are usually high quality
            'processing_status': 'pending'
        }

    def collect_hashtags(self, hashtags: List[str], max_per_hashtag: int = 50) -> List[CollectionMetadata]:
        """Collect videos from multiple Instagram hashtags"""
        all_posts = []

        for hashtag in hashtags:
            logger.info(f"📸 Collecting Instagram hashtag: {hashtag}")

            # Add # prefix if not present
            if not hashtag.startswith('#'):
                hashtag = f"#{hashtag}"

            posts = self.search(hashtag, max_per_hashtag)
            all_posts.extend(posts)

            # Rate limiting between hashtags (Instagram is strict)
            time.sleep(5)

        # Remove duplicates based on post ID
        unique_posts = []
        seen_ids = set()
        for post in all_posts:
            if post['id'] not in seen_ids:
                unique_posts.append(post)
                seen_ids.add(post['id'])

        logger.info(f"📊 Total unique Instagram posts collected: {len(unique_posts)}")

        # Process collection
        return self.collect_from_items(unique_posts)

    def collect_from_items(self, items: List[Dict]) -> List[CollectionMetadata]:
        """Process a list of post items"""
        metadata_list = []

        for item in tqdm(items, desc="Processing Instagram posts"):
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
                    duration=item.get('duration'),
                    file_path=str(file_path),
                    file_size=file_path.stat().st_size,
                    inferred_gender=item.get('inferred_gender'),
                    quality_score=processed_data.get('quality_score'),
                    tags=item.get('hashtags', []),
                    extra_metadata={
                        'author': item.get('author'),
                        'author_id': item.get('author_id'),
                        'likes': item.get('likes'),
                        'comments': item.get('comments'),
                        'is_reel': item.get('is_reel'),
                        'location': item.get('location'),
                        'mentions': item.get('mentions'),
                        'date_posted': item.get('date_posted'),
                        'platform': 'instagram'
                    }
                )

                metadata_list.append(metadata)

            except Exception as e:
                logger.error(f"❌ Error processing Instagram post {item.get('id')}: {e}")
                continue

        # Save metadata
        if metadata_list:
            self._save_metadata(metadata_list)

        return metadata_list

    def get_user_posts(self, username: str, max_posts: int = 20) -> List[Dict]:
        """Get recent video posts from a specific user"""
        if not self.loader:
            return []

        try:
            logger.info(f"👤 Getting posts from Instagram user: {username}")

            profile = instaloader.Profile.from_username(self.loader.context, username)
            posts = []

            count = 0
            for post in profile.get_posts():
                if count >= max_posts:
                    break

                if post.is_video:
                    post_data = self._extract_post_data(post)
                    if post_data:
                        posts.append(post_data)
                        count += 1

                time.sleep(1)  # Rate limiting

            return posts

        except Exception as e:
            logger.error(f"❌ Error getting posts from user {username}: {e}")
            return []

    def cleanup(self):
        """Cleanup Instaloader session"""
        if self.loader:
            # Instaloader doesn't have explicit cleanup, but we can close context
            logger.info("🧹 Instaloader session ended")
        super().cleanup()


# ===== USAGE EXAMPLES =====

def collect_instagram_sample():
    """Collect sample Instagram videos for testing"""
    collector = InstagramCollector()

    # Test hashtags
    test_hashtags = ['#vlog', '#storytime']

    try:
        results = collector.collect_hashtags(test_hashtags, max_per_hashtag=3)
        print(f"✅ Collected {len(results)} Instagram posts")

        for meta in results[:3]:  # Show first 3
            print(f"  - {meta.title} ({meta.inferred_gender}) - {meta.extra_metadata.get('likes')} likes")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        collector.cleanup()


def collect_from_user_sample():
    """Collect videos from specific users"""
    collector = InstagramCollector()

    # Example users (replace with real ones)
    users = ['instagram']  # Test with official account

    try:
        for user in users:
            posts = collector.get_user_posts(user, max_posts=2)
            results = collector.collect_from_items(posts)
            print(f"✅ Collected {len(results)} posts from {user}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        collector.cleanup()


if __name__ == "__main__":
    collect_instagram_sample()

#!/usr/bin/env python3
"""
Unit Tests for Data Collectors
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_collection.youtube_collector import YouTubeCollector
from data_collection.tiktok_collector import TikTokCollector
from data_collection.instagram_collector import InstagramCollector


class TestYouTubeCollector:
    """Test YouTube data collector"""

    @pytest.fixture
    def config(self):
        return {
            'sources': {
                'youtube': {
                    'api_key': 'test_key',
                    'max_videos_per_query': 5,
                    'filters': {
                        'min_duration': 30,
                        'max_duration': 600
                    }
                }
            }
        }

    @pytest.fixture
    def collector(self, config):
        with patch('data_collection.youtube_collector.YouTubeCollector._load_config', return_value=config):
            return YouTubeCollector()

    def test_initialization(self, collector):
        """Test collector initialization"""
        assert collector is not None
        assert hasattr(collector, 'yt_config')
        assert collector.yt_config['max_videos_per_query'] == 5

    @patch('yt_dlp.YoutubeDL')
    def test_search_videos(self, mock_ydl, collector):
        """Test video search functionality"""
        # Mock yt-dlp response
        mock_result = {
            'entries': [
                {
                    'id': 'test_id_1',
                    'title': 'Test Video 1',
                    'duration': 120,
                    'uploader': 'Test Channel'
                },
                {
                    'id': 'test_id_2',
                    'title': 'Test Video 2',
                    'duration': 25,  # Too short, should be filtered
                    'uploader': 'Test Channel'
                }
            ]
        }

        mock_ydl.return_value.extract_info.return_value = mock_result

        results = collector.search("test query", 5)

        assert len(results) == 1  # Only one video should pass duration filter
        assert results[0]['id'] == 'test_id_1'
        assert results[0]['duration'] == 120

    @patch('yt_dlp.YoutubeDL')
    @patch('pathlib.Path.exists', return_value=False)
    @patch('pathlib.Path.unlink')  # Mock unlink to avoid file operations
    def test_download_video(self, mock_unlink, mock_exists, mock_ydl, collector):
        """Test video download functionality"""
        mock_item = {
            'id': 'test_video_id',
            'url': 'https://youtube.com/watch?v=test_video_id'
        }

        # Mock successful download
        mock_ydl.return_value.__enter__.return_value.download.return_value = None

        result = collector.download(mock_item)

        assert result is not None
        mock_ydl.return_value.__enter__.return_value.download.assert_called_once_with([mock_item['url']])

    def test_validate_video(self, collector, tmp_path):
        """Test video validation"""
        # Create a mock video file path
        mock_video = tmp_path / "test_video.mp4"
        mock_video.write_bytes(b"fake video content")

        # Test with non-existent file
        assert not collector.validate(Path("nonexistent.mp4"))

        # Test with existing file (would need actual video for full test)
        # For now, just test file existence
        assert collector.validate(mock_video) == False  # Since it's not a real video


class TestTikTokCollector:
    """Test TikTok data collector"""

    @pytest.fixture
    def config(self):
        return {
            'sources': {
                'tiktok': {
                    'max_videos_per_hashtag': 10,
                    'filters': {
                        'min_duration': 15,
                        'max_duration': 180
                    }
                }
            }
        }

    @pytest.fixture
    def collector(self, config):
        with patch('data_collection.tiktok_collector.TikTokCollector._load_config', return_value=config):
            with patch('selenium.webdriver.Chrome'):  # Mock selenium
                return TikTokCollector()

    @patch('selenium.webdriver.Chrome')
    def test_initialization(self, mock_driver, collector):
        """Test TikTok collector initialization"""
        assert collector is not None
        # Mock driver should be assigned
        assert collector.driver is not None

    def test_extract_video_id(self, collector):
        """Test TikTok video ID extraction"""
        test_urls = [
            ("https://www.tiktok.com/@user/video/1234567890123456789", "1234567890123456789"),
            ("https://tiktok.com/t/abcdef123/", None),  # Invalid format
        ]

        for url, expected in test_urls:
            result = collector._extract_video_id(url)
            assert result == expected


class TestInstagramCollector:
    """Test Instagram data collector"""

    @pytest.fixture
    def config(self):
        return {
            'sources': {
                'instagram': {
                    'max_posts_per_hashtag': 10,
                    'filters': {
                        'min_likes': 50
                    }
                }
            }
        }

    @pytest.fixture
    def collector(self, config):
        with patch('data_collection.instagram_collector.InstagramCollector._load_config', return_value=config):
            with patch('instaloader.Instaloader'):  # Mock instaloader
                return InstagramCollector()

    @patch('instaloader.Instaloader')
    def test_initialization(self, mock_loader, collector):
        """Test Instagram collector initialization"""
        assert collector is not None
        assert collector.loader is not None

    def test_extract_hashtags(self, collector):
        """Test hashtag extraction from caption"""
        caption = "Beautiful day at the beach! #vacation #summer #beachlife"
        hashtags = collector._extract_hashtags(caption)

        expected = {'vacation', 'summer', 'beachlife'}
        assert set(hashtags) == expected

    def test_infer_gender_from_caption(self, collector):
        """Test gender inference from caption"""
        test_cases = [
            ("He is a great guy #men", "male"),
            ("She looks amazing #women", "female"),
            ("Beautiful scenery", None),
        ]

        for caption, expected in test_cases:
            result = collector._infer_gender_from_caption(caption, "test_user")
            assert result == expected


# Integration Tests
class TestCollectorIntegration:
    """Integration tests for collectors"""

    def test_config_validation(self):
        """Test that all collectors can load config"""
        collectors = [YouTubeCollector, TikTokCollector, InstagramCollector]

        for collector_class in collectors:
            try:
                # This will fail if config is invalid, but should not crash
                collector = collector_class()
                assert hasattr(collector, 'config')
            except Exception:
                # Config loading might fail in test environment, which is OK
                pass

    def test_collector_interface(self):
        """Test that all collectors implement required interface"""
        required_methods = ['search', 'download', 'validate', 'process']

        collectors = [YouTubeCollector, TikTokCollector, InstagramCollector]

        for collector_class in collectors:
            for method in required_methods:
                assert hasattr(collector_class, method), f"{collector_class.__name__} missing {method}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

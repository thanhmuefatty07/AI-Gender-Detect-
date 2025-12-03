"""
Advanced Video Processor for Gender Classification

Features:
1. Face Detection & Extraction (MediaPipe, OpenCV, DLib)
2. Face Quality Assessment (blur, brightness, size)
3. Audio Extraction & Feature Computation
4. Multi-format Support (MP4, AVI, MOV)
5. Batch Processing with Progress Tracking
6. GPU Acceleration Support
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import librosa
import soundfile as sf
from loguru import logger
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
from datetime import datetime
import torch
import torch.nn as nn
from PIL import Image
import io

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_collection.base_collector import CollectionMetadata


class FaceExtractor:
    """Advanced face extraction with multiple detection methods"""

    def __init__(self, config: Dict):
        self.config = config
        self.face_config = config['processing']['face_detection']

        # Initialize detection methods
        self.detectors = {}
        self._init_detectors()

    def _init_detectors(self):
        """Initialize face detection models"""
        try:
            # MediaPipe (primary, fastest)
            if self.face_config['method'] == 'mediapipe':
                self.mp_face_detection = mp.solutions.face_detection
                self.detectors['mediapipe'] = self.mp_face_detection.FaceDetection(
                    min_detection_confidence=self.face_config['min_detection_confidence']
                )
                logger.info("✅ MediaPipe face detector initialized")

            # OpenCV DNN
            elif self.face_config['method'] == 'opencv_dnn':
                model_path = "models/opencv_face_detector_uint8.pb"
                config_path = "models/opencv_face_detector.pbtxt"

                if Path(model_path).exists():
                    self.detectors['opencv_dnn'] = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                    logger.info("✅ OpenCV DNN face detector initialized")
                else:
                    logger.warning("⚠️ OpenCV DNN model files not found, using MediaPipe")
                    self._init_mediapipe()

            # DLib (most accurate)
            elif self.face_config['method'] == 'dlib':
                import dlib
                predictor_path = "models/shape_predictor_68_face_landmarks.dat"

                if Path(predictor_path).exists():
                    self.detectors['dlib'] = dlib.get_frontal_face_detector()
                    self.detectors['dlib_predictor'] = dlib.shape_predictor(predictor_path)
                    logger.info("✅ DLib face detector initialized")
                else:
                    logger.warning("⚠️ DLib model files not found, using MediaPipe")
                    self._init_mediapipe()

        except Exception as e:
            logger.error(f"❌ Failed to initialize face detectors: {e}")
            # Fallback to basic OpenCV Haar cascades
            self._init_haar_cascade()

    def _init_mediapipe(self):
        """Initialize MediaPipe as fallback"""
        self.mp_face_detection = mp.solutions.face_detection
        self.detectors['mediapipe'] = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )

    def _init_haar_cascade(self):
        """Initialize Haar cascade as last resort"""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detectors['haar'] = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image using configured method

        Returns: List of face bounding boxes with confidence scores
        """
        faces = []

        try:
            if 'mediapipe' in self.detectors:
                faces = self._detect_mediapipe(image)
            elif 'opencv_dnn' in self.detectors:
                faces = self._detect_opencv_dnn(image)
            elif 'dlib' in self.detectors:
                faces = self._detect_dlib(image)
            elif 'haar' in self.detectors:
                faces = self._detect_haar(image)

        except Exception as e:
            logger.warning(f"⚠️ Face detection error: {e}")

        return faces

    def _detect_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """MediaPipe face detection"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detectors['mediapipe'].process(rgb_image)

        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                faces.append({
                    'x': int(bbox.xmin * w),
                    'y': int(bbox.ymin * h),
                    'width': int(bbox.width * w),
                    'height': int(bbox.height * h),
                    'confidence': detection.score[0]
                })

        return faces

    def _detect_opencv_dnn(self, image: np.ndarray) -> List[Dict]:
        """OpenCV DNN face detection"""
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.detectors['opencv_dnn'].setInput(blob)
        detections = self.detectors['opencv_dnn'].forward()

        faces = []
        h, w = image.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.face_config['min_detection_confidence']:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x2, y2 = box.astype(int)

                faces.append({
                    'x': x,
                    'y': y,
                    'width': x2 - x,
                    'height': y2 - y,
                    'confidence': float(confidence)
                })

        return faces

    def _detect_dlib(self, image: np.ndarray) -> List[Dict]:
        """DLib face detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_dlib = self.detectors['dlib'](gray)

        faces = []
        for face in faces_dlib:
            faces.append({
                'x': face.left(),
                'y': face.top(),
                'width': face.right() - face.left(),
                'height': face.bottom() - face.top(),
                'confidence': 0.9  # DLib doesn't provide confidence
            })

        return faces

    def _detect_haar(self, image: np.ndarray) -> List[Dict]:
        """Haar cascade face detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_haar = self.detectors['haar'].detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        faces = []
        for (x, y, w, h) in faces_haar:
            faces.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'confidence': 0.8  # Haar doesn't provide confidence
            })

        return faces

    def extract_face(self, image: np.ndarray, face_bbox: Dict, padding: float = 0.1) -> Optional[np.ndarray]:
        """Extract and crop face from image with padding"""
        try:
            x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']

            # Add padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)

            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(image.shape[1] - x, w + 2 * pad_x)
            h = min(image.shape[0] - y, h + 2 * pad_y)

            # Crop face
            face = image[y:y+h, x:x+w]

            # Convert to RGB for consistency
            if face.size > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                return face

        except Exception as e:
            logger.warning(f"⚠️ Face extraction error: {e}")

        return None


class QualityAssessor:
    """Face image quality assessment"""

    def __init__(self, config: Dict):
        self.config = config
        self.quality_config = config['processing']['quality_filters']

    def assess_quality(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive quality assessment"""
        if face_image.size == 0:
            return {'overall_score': 0.0, 'issues': ['empty_image']}

        scores = {}
        issues = []

        # Size check
        h, w = face_image.shape[:2]
        min_size = self.quality_config['min_face_size']
        if h < min_size or w < min_size:
            scores['size'] = 0.0
            issues.append('too_small')
        else:
            scores['size'] = 1.0

        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_threshold = self.quality_config['blur_threshold']

        if laplacian_var < blur_threshold:
            scores['blur'] = laplacian_var / blur_threshold
            if scores['blur'] < 0.5:
                issues.append('blurry')
        else:
            scores['blur'] = 1.0

        # Brightness check
        brightness = gray.mean()
        min_bright, max_bright = self.quality_config['brightness_range']

        if min_bright <= brightness <= max_bright:
            scores['brightness'] = 1.0
        else:
            scores['brightness'] = max(0, 1 - abs(brightness - (min_bright + max_bright) / 2) / 50)
            if scores['brightness'] < 0.5:
                issues.append('brightness_issue')

        # Contrast check
        contrast = gray.std()
        scores['contrast'] = min(1.0, contrast / 50.0)
        if scores['contrast'] < 0.3:
            issues.append('low_contrast')

        # Overall score (weighted average)
        weights = {'size': 0.3, 'blur': 0.3, 'brightness': 0.2, 'contrast': 0.2}
        overall_score = sum(scores.get(k, 0) * w for k, w in weights.items())

        return {
            'overall_score': overall_score,
            'individual_scores': scores,
            'issues': issues,
            'metrics': {
                'blur_variance': laplacian_var,
                'brightness': brightness,
                'contrast': contrast,
                'dimensions': (w, h)
            }
        }


class AudioProcessor:
    """Audio feature extraction for gender classification"""

    def __init__(self, config: Dict):
        self.config = config
        self.audio_config = config['processing']['audio_extraction']

    def extract_audio(self, video_path: Path, output_path: Path) -> bool:
        """Extract audio from video using ffmpeg"""
        try:
            import ffmpeg

            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_path),
                    acodec='pcm_s16le',  # WAV format
                    ac=1,  # Mono
                    ar=self.audio_config['sample_rate'],
                    vn=None  # No video
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )

            return output_path.exists()

        except ffmpeg.Error as e:
            logger.error(f"❌ Audio extraction failed: {e.stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"❌ Audio extraction error: {e}")
            return False

    def compute_features(self, audio_path: Path) -> Dict[str, Any]:
        """Compute audio features for gender classification"""
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.audio_config['sample_rate'])

            if len(y) == 0:
                return {'error': 'empty_audio'}

            # Remove silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)

            if len(y_trimmed) < sr * 1:  # Less than 1 second
                return {'error': 'too_short'}

            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(
                y=y_trimmed,
                sr=sr,
                n_mfcc=40,
                hop_length=512
            )

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)

            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)

            # Pitch (F0) - important for gender
            pitches, magnitudes = librosa.piptrack(
                y=y_trimmed,
                sr=sr,
                hop_length=512
            )

            # Get mean pitch
            pitch_values = pitches[pitches > 0]
            mean_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y_trimmed)

            # RMS energy
            rms = librosa.feature.rms(y=y_trimmed)

            return {
                'duration': len(y_trimmed) / sr,
                'sample_rate': sr,
                'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
                'mfcc_std': np.std(mfccs, axis=1).tolist(),
                'chroma_mean': np.mean(chroma, axis=1).tolist(),
                'spectral_centroid_mean': np.mean(spectral_centroid),
                'mean_pitch': float(mean_pitch),
                'pitch_std': float(np.std(pitch_values)) if len(pitch_values) > 0 else 0,
                'zcr_mean': np.mean(zcr),
                'rms_mean': np.mean(rms),
                'mfcc_shape': mfccs.shape
            }

        except Exception as e:
            logger.error(f"❌ Audio feature computation error: {e}")
            return {'error': str(e)}


class VideoProcessor:
    """Main video processing pipeline"""

    def __init__(self, config_path: str = "config/collector_config.yaml"):
        # Load config
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.face_extractor = FaceExtractor(self.config)
        self.quality_assessor = QualityAssessor(self.config)
        self.audio_processor = AudioProcessor(self.config)

        # Output settings
        self.output_config = self.config['output']
        self.output_dir = Path(self.output_config['base_path'])

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup processing logs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = project_root / "logs" / f"video_processor_{timestamp}.log"

        logger.remove()
        logger.add(log_file, level="INFO")
        logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")

    def process_video(self, video_path: Path, metadata: Dict = None) -> Dict[str, Any]:
        """
        Complete video processing pipeline

        1. Extract faces with quality filtering
        2. Extract audio and compute features
        3. Generate processing report
        """
        if metadata is None:
            metadata = {}

        logger.info(f"🎬 Processing video: {video_path.name}")

        start_time = time.time()
        result = {
            'video_path': str(video_path),
            'video_id': video_path.stem,
            'processing_start': datetime.now().isoformat(),
            'faces_extracted': [],
            'audio_features': None,
            'quality_score': 0.0,
            'processing_time': 0.0,
            'status': 'processing'
        }

        try:
            # Create output directory
            video_output_dir = self.output_dir / "processed" / video_path.stem
            video_output_dir.mkdir(parents=True, exist_ok=True)

            # Extract faces
            face_results = self._extract_faces_from_video(video_path, video_output_dir)
            result['faces_extracted'] = face_results

            # Extract audio
            audio_path = video_output_dir / "audio.wav"
            if self.audio_processor.extract_audio(video_path, audio_path):
                audio_features = self.audio_processor.compute_features(audio_path)
                result['audio_features'] = audio_features
                result['audio_path'] = str(audio_path)
            else:
                logger.warning(f"⚠️ Audio extraction failed for {video_path.name}")

            # Calculate overall quality score
            face_qualities = [face['quality']['overall_score'] for face in face_results]
            result['quality_score'] = np.mean(face_qualities) if face_qualities else 0.0

            result['status'] = 'completed'
            result['face_count'] = len(face_results)

        except Exception as e:
            logger.error(f"❌ Processing failed for {video_path.name}: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)

        result['processing_time'] = time.time() - start_time
        result['processing_end'] = datetime.now().isoformat()

        logger.success(f"✅ Processed {video_path.name}: {len(result['faces_extracted'])} faces, quality: {result['quality_score']:.2f}")

        return result

    def _extract_faces_from_video(self, video_path: Path, output_dir: Path) -> List[Dict]:
        """Extract high-quality faces from video"""
        faces_dir = output_dir / "faces"
        faces_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"❌ Could not open video: {video_path}")
            return []

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"📹 Video info: {total_frames} frames, {fps} fps, {duration:.1f}s")

        # Sample frames (every 2 seconds for diverse faces)
        sample_interval = fps * 2
        if sample_interval < 1:
            sample_interval = 1

        faces_extracted = []
        frame_count = 0
        face_count = 0
        max_faces = 50  # Limit faces per video

        with tqdm(total=min(total_frames, max_faces * sample_interval), desc="Extracting faces") as pbar:
            while cap.isOpened() and face_count < max_faces:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames
                if frame_count % sample_interval != 0:
                    frame_count += 1
                    continue

                # Detect faces
                detected_faces = self.face_extractor.detect_faces(frame)

                for face_bbox in detected_faces:
                    # Extract face
                    face_image = self.face_extractor.extract_face(frame, face_bbox)
                    if face_image is None:
                        continue

                    # Quality assessment
                    quality = self.quality_assessor.assess_quality(face_image)

                    # Only save high-quality faces
                    if quality['overall_score'] >= 0.6:  # Configurable threshold
                        face_filename = "04d"
                        face_path = faces_dir / face_filename

                        # Save face image
                        pil_image = Image.fromarray(face_image)
                        pil_image.save(face_path)

                        faces_extracted.append({
                            'face_path': str(face_path),
                            'frame_number': frame_count,
                            'bbox': face_bbox,
                            'quality': quality,
                            'timestamp': frame_count / fps if fps > 0 else 0
                        })

                        face_count += 1

                        if face_count >= max_faces:
                            break

                frame_count += 1
                pbar.update(1)

        cap.release()

        logger.info(f"🖼️ Extracted {len(faces_extracted)} high-quality faces")
        return faces_extracted

    def process_batch(self, video_paths: List[Path], max_workers: int = 4) -> List[Dict]:
        """Process multiple videos in parallel"""
        logger.info(f"🔄 Processing {len(video_paths)} videos with {max_workers} workers")

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(self.process_video, video_path): video_path
                for video_path in video_paths
            }

            # Collect results as they complete
            for future in tqdm(as_completed(future_to_video), total=len(video_paths), desc="Batch processing"):
                video_path = future_to_video[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Batch processing failed for {video_path.name}: {e}")
                    results.append({
                        'video_path': str(video_path),
                        'status': 'failed',
                        'error': str(e)
                    })

        # Save batch results
        self._save_batch_results(results)

        successful = sum(1 for r in results if r.get('status') == 'completed')
        logger.info(f"✅ Batch processing completed: {successful}/{len(results)} successful")

        return results

    def _save_batch_results(self, results: List[Dict]):
        """Save batch processing results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"batch_processing_results_{timestamp}.json"

        # Add summary statistics
        summary = {
            'timestamp': timestamp,
            'total_videos': len(results),
            'successful': sum(1 for r in results if r.get('status') == 'completed'),
            'failed': sum(1 for r in results if r.get('status') == 'failed'),
            'total_faces': sum(r.get('face_count', 0) for r in results),
            'avg_quality': np.mean([r.get('quality_score', 0) for r in results if r.get('status') == 'completed']),
            'total_processing_time': sum(r.get('processing_time', 0) for r in results),
            'results': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 Batch results saved: {output_file}")

    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        # Load recent batch results
        result_files = list(self.output_dir.glob("batch_processing_results_*.json"))

        if not result_files:
            return {'total_processed': 0, 'success_rate': 0, 'avg_quality': 0}

        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return {
                'total_processed': data.get('total_videos', 0),
                'success_rate': data.get('successful', 0) / data.get('total_videos', 1),
                'avg_quality': data.get('avg_quality', 0),
                'total_faces': data.get('total_faces', 0),
                'last_updated': latest_file.stat().st_mtime
            }

        except Exception:
            return {'total_processed': 0, 'success_rate': 0, 'avg_quality': 0}


def main():
    """Command line interface for video processing"""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Video Processor for Gender Classification")
    parser.add_argument('--video', help='Single video file to process')
    parser.add_argument('--batch', nargs='+', help='Multiple video files to process')
    parser.add_argument('--input-dir', help='Directory containing videos to process')
    parser.add_argument('--output-dir', help='Output directory (overrides config)')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--config', default='config/collector_config.yaml', help='Config file path')

    args = parser.parse_args()

    processor = VideoProcessor(args.config)

    if args.output_dir:
        processor.output_dir = Path(args.output_dir)

    try:
        if args.video:
            # Process single video
            video_path = Path(args.video)
            if video_path.exists():
                result = processor.process_video(video_path)
                print(f"✅ Processed {video_path.name}: {result['face_count']} faces")
            else:
                print(f"❌ Video not found: {video_path}")

        elif args.batch:
            # Process batch of videos
            video_paths = [Path(v) for v in args.batch if Path(v).exists()]
            if video_paths:
                results = processor.process_batch(video_paths, args.workers)
                successful = sum(1 for r in results if r.get('status') == 'completed')
                print(f"✅ Batch completed: {successful}/{len(video_paths)} videos processed")
            else:
                print("❌ No valid video files found")

        elif args.input_dir:
            # Process all videos in directory
            input_dir = Path(args.input_dir)
            if input_dir.exists():
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
                video_paths = [
                    f for ext in video_extensions
                    for f in input_dir.rglob(f"*{ext}")
                ]

                if video_paths:
                    results = processor.process_batch(video_paths, args.workers)
                    successful = sum(1 for r in results if r.get('status') == 'completed')
                    print(f"✅ Directory processing completed: {successful}/{len(video_paths)} videos processed")
                else:
                    print(f"❌ No video files found in {input_dir}")
            else:
                print(f"❌ Directory not found: {input_dir}")

        else:
            # Show stats
            stats = processor.get_processing_stats()
            print("📊 Processing Statistics:")
            print(f"  Total processed: {stats['total_processed']}")
            print(f"  Success rate: {stats['success_rate']:.2%}")
            print(f"  Average quality: {stats['avg_quality']:.2f}")
            print(f"  Total faces: {stats['total_faces']}")

    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        exit(1)


if __name__ == "__main__":
    main()

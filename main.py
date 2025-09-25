#!/usr/bin/env python3
"""
MLX Talks Categorizer - Audio File Management System
Organizes, transcribes, and categorizes MP3/WAV files using MLX for Apple Silicon optimization.
"""

import os
import json
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse

# Audio processing libraries
import librosa
import soundfile as sf
import numpy as np

# Title Generation
import requests
import json

# Video processing for MP4 files
import subprocess
import re

# MLX libraries for Apple Silicon optimization
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. Falling back to CPU processing.")

# MLX Whisper for transcription (Apple Silicon optimized)
try:
    import mlx_whisper
    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False
    # Fallback to regular Whisper
    try:
        import whisper
        WHISPER_AVAILABLE = True
        print("Warning: MLX Whisper not available. Using regular Whisper. Install MLX Whisper with: pip install mlx-whisper")
    except ImportError:
        WHISPER_AVAILABLE = False
        print("Warning: Neither MLX Whisper nor regular Whisper available. Install with: pip install mlx-whisper")

# Speaker identification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import euclidean
from scipy.stats import mode
from fastdtw import fastdtw


class Config:
    """Configuration management for the audio processing system."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.default_config = {
            "min_duration_minutes": 10,
            "supported_formats": [".mp3", ".wav", ".mp4"],
            "audio_inputs": [
                "~/Google Drive/Audio",
                "~/Audio Hijack"
            ],
            "output_base_path": "./organized_talks",
            "speakers_path": "./organized_talks/speakers",
            "talks_path": "./organized_talks/talks",
            "transcripts_path": "./organized_talks/transcripts",
            "raw_talks_path": "./organized_talks/raw talks",
            "last_run_file": "last_run.json",
            "whisper_model": "medium",  # Uses MLX-optimized models when available
            "speaker_similarity_threshold": 0.75,  # Lowered for better recall
            "speaker_confidence_margin": 0.05,  # Minimum difference between top matches
            "max_audio_duration_for_features": 120,  # Max seconds to process for features
            "voice_activity_energy_percentile": 30,  # Energy threshold percentile
            "voice_activity_zcr_percentile": 70,  # ZCR threshold percentile
            "cleanup_days": 30,
            "title_generation": {
                "method": "ollama",
                "ollama_base_url": "http://localhost:11434",
                "ollama_model": "llama3.2:3b",
                "max_title_words": 3,
                "fallback_to_simple": True
            },
            "feature_extraction": {
                "n_mfcc": 26,  # Increased from 13
                "n_fft": 2048,
                "hop_length": 512,
                "include_deltas": True,
                "include_spectral_features": True,
                "gmm_components": 3  # For speakers with multiple samples
            }
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults for any missing keys
            for key, value in self.default_config.items():
                if key not in config:
                    config[key] = value
            return config
        else:
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def save_config(self, config: Dict = None):
        """Save configuration to file."""
        config_to_save = config or self.config
        with open(self.config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)


class AudioProcessor:
    """Handles audio file processing and filtering."""
    
    def __init__(self, config: Config):
        self.config = config
        self.min_duration = config.get("min_duration_minutes") * 60  # Convert to seconds
        self.supported_formats = config.get("supported_formats")
    
    def get_audio_duration(self, file_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            # For MP4 files, try ffprobe first as it handles video containers better
            if file_path.lower().endswith('.mp4'):
                try:
                    result = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                        '-show_format', file_path
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        import json
                        probe_data = json.loads(result.stdout)
                        duration = float(probe_data['format']['duration'])
                        return duration
                except Exception:
                    # Fallback to librosa
                    pass
            
            # Use librosa for audio files or as fallback
            duration = librosa.get_duration(path=file_path)
            return duration
        except Exception as e:
            logging.error(f"Error getting duration for {file_path}: {e}")
            return 0
    
    def is_valid_audio_file(self, file_path: str) -> bool:
        """Check if file is a valid audio file with minimum duration."""
        if not os.path.exists(file_path):
            return False
        
        # Check file extension
        ext = Path(file_path).suffix.lower()
        if ext not in self.supported_formats:
            return False
        
        # Check duration
        duration = self.get_audio_duration(file_path)
        return duration >= self.min_duration
    
    def detect_voice_activity(self, audio: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """Detect voice activity using energy and spectral features."""
        try:
            # Calculate short-time energy
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

            # Calculate spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]

            # Calculate zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]

            # Use configurable thresholds
            energy_percentile = self.config.get("voice_activity_energy_percentile", 30)
            zcr_percentile = self.config.get("voice_activity_zcr_percentile", 70)

            energy_threshold = np.percentile(energy, energy_percentile)
            zcr_threshold = np.percentile(zcr, zcr_percentile)

            # Voice activity: high energy, moderate ZCR
            voice_activity = (energy > energy_threshold) & (zcr < zcr_threshold)

            # Convert frame-based detection to sample-based
            voice_samples = np.zeros_like(audio, dtype=bool)
            for i, is_voice in enumerate(voice_activity):
                start_sample = i * hop_length
                end_sample = min(start_sample + hop_length, len(audio))
                voice_samples[start_sample:end_sample] = is_voice

            return voice_samples
        except Exception as e:
            logging.error(f"Error in voice activity detection: {e}")
            return np.ones_like(audio, dtype=bool)  # Return all True as fallback

    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess audio with normalization and filtering."""
        try:
            # Volume normalization
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            # Apply voice activity detection
            voice_mask = self.detect_voice_activity(audio, sr)

            # Extract only voice segments
            if np.any(voice_mask):
                audio = audio[voice_mask]

            # Pre-emphasis filter to balance frequency spectrum
            emphasized_audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

            return emphasized_audio
        except Exception as e:
            logging.error(f"Error in audio preprocessing: {e}")
            return audio

    def extract_audio_features(self, file_path: str, max_duration: float = None) -> Optional[np.ndarray]:
        """Extract comprehensive audio features for speaker identification."""
        try:
            # Load audio
            if file_path.lower().endswith('.mp4'):
                y, sr = librosa.load(file_path, sr=22050)
            else:
                y, sr = librosa.load(file_path, sr=22050)

            # Limit duration for processing efficiency
            if max_duration is None:
                max_duration = self.config.get("max_audio_duration_for_features", 120)
            max_samples = int(max_duration * sr)
            if len(y) > max_samples:
                # Take middle portion to avoid intro/outro effects
                start_idx = (len(y) - max_samples) // 2
                y = y[start_idx:start_idx + max_samples]

            # Preprocess audio
            y_processed = self.preprocess_audio(y, sr)

            if len(y_processed) < sr:  # Less than 1 second of voice
                logging.warning(f"Insufficient voice activity in {file_path}")
                return None

            # Get feature extraction config
            feat_config = self.config.get("feature_extraction", {})
            n_mfcc = feat_config.get("n_mfcc", 26)
            n_fft = feat_config.get("n_fft", 2048)
            hop_length = feat_config.get("hop_length", 512)

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y_processed, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

            features_list = [mfccs]

            # Extract delta and delta-delta features if enabled
            if feat_config.get("include_deltas", True):
                delta_mfccs = librosa.feature.delta(mfccs)
                delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                features_list.extend([delta_mfccs, delta2_mfccs])

            # Extract spectral features if enabled
            if feat_config.get("include_spectral_features", True):
                spectral_centroid = librosa.feature.spectral_centroid(y=y_processed, sr=sr, hop_length=hop_length)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y_processed, sr=sr, hop_length=hop_length)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_processed, sr=sr, hop_length=hop_length)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y_processed, hop_length=hop_length)
                features_list.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossing_rate])

            # Combine all features
            all_features = np.vstack(features_list)

            # Statistical features: mean, std, min, max
            feature_stats = np.vstack([
                np.mean(all_features, axis=1),
                np.std(all_features, axis=1),
                np.min(all_features, axis=1),
                np.max(all_features, axis=1)
            ])

            # Flatten to create feature vector
            features = feature_stats.flatten()

            logging.info(f"Extracted {len(features)} features from {file_path}")
            return features

        except Exception as e:
            logging.error(f"Error extracting features from {file_path}: {e}")
            return None


class Transcriber:
    """Handles audio transcription using Whisper and title generation using Ollama."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.title_config = config.get("title_generation", {})
        self.use_mlx = False

        # Try MLX Whisper first for Apple Silicon optimization
        if MLX_WHISPER_AVAILABLE:
            try:
                model_name = config.get("whisper_model", "medium")
                # MLX Whisper uses different model naming convention
                mlx_model_name = self._convert_to_mlx_model_name(model_name)
                # For MLX Whisper, we store the model name rather than loading it
                self.model = mlx_model_name
                self.use_mlx = True
                logging.info(f"Configured MLX Whisper model: {mlx_model_name} (Apple Silicon optimized)")
            except Exception as e:
                logging.error(f"Error configuring MLX Whisper model: {e}")
                self.model = None

        # Fallback to regular Whisper if MLX Whisper failed or unavailable
        if self.model is None and WHISPER_AVAILABLE:
            try:
                model_name = config.get("whisper_model", "medium")
                self.model = whisper.load_model(model_name)
                logging.info(f"Loaded regular Whisper model: {model_name}")
            except Exception as e:
                logging.error(f"Error loading regular Whisper model: {e}")

    def _convert_to_mlx_model_name(self, model_name: str) -> str:
        """Convert regular Whisper model name to MLX Whisper format."""
        # MLX Whisper uses community models from Hugging Face
        mlx_model_mapping = {
            "tiny": "mlx-community/whisper-tiny",
            "base": "mlx-community/whisper-base",
            "small": "mlx-community/whisper-small",
            "medium": "mlx-community/whisper-medium",
            "large": "mlx-community/whisper-large-v2",
            "large-v2": "mlx-community/whisper-large-v2",
            "large-v3": "mlx-community/whisper-large-v3"
        }
        return mlx_model_mapping.get(model_name, f"mlx-community/whisper-{model_name}")
    
    def transcribe_audio(self, file_path: str) -> Optional[str]:
        """Transcribe audio file to text."""
        if not self.model:
            logging.error("Whisper model not available")
            return None

        try:
            if self.use_mlx:
                # MLX Whisper transcription
                result = mlx_whisper.transcribe(file_path, path_or_hf_repo=self.model)
                return result["text"].strip()
            else:
                # Regular Whisper transcription
                result = self.model.transcribe(file_path)
                return result["text"].strip()
        except Exception as e:
            logging.error(f"Error transcribing {file_path}: {e}")
            return None
    
    def generate_ollama_title(self, transcript: str) -> Optional[str]:
        """Generate title using Ollama LLM."""
        try:
            base_url = self.title_config.get("ollama_base_url", "http://localhost:11434")
            model = self.title_config.get("ollama_model", "llama3.2:3b")
            max_words = self.title_config.get("max_title_words", 6)

            # Create a focused prompt for title generation
            prompt = f"""Based on this transcript, create a concise {max_words}-word title that captures the main topic or theme.

            Transcript: {transcript[:1000]}...

            Respond with only the {max_words}-word title, no explanations or additional text."""

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 20
                }
            }

            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                title = result.get("response", "").strip()

                # Clean up the title - remove quotes, periods, etc.
                title = title.strip('"\'.,!?')

                # Ensure it's roughly the right length
                words = title.split()
                if len(words) > max_words:
                    title = " ".join(words[:max_words])

                if title:
                    logging.info(f"Generated Ollama title: {title}")
                    return title.title()
                else:
                    logging.warning("Ollama returned empty title")
                    return None
            else:
                logging.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            logging.warning("Could not connect to Ollama server")
            return None
        except requests.exceptions.Timeout:
            logging.warning("Ollama request timed out")
            return None
        except Exception as e:
            logging.error(f"Error generating Ollama title: {e}")
            return None

    def generate_simple_title(self, transcript: str) -> str:
        """Generate simple title by extracting key words."""
        max_words = self.title_config.get("max_title_words", 3)
        words = transcript.lower().split()

        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }

        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

        # Take first meaningful words up to max_words
        description_words = filtered_words[:max_words] if len(filtered_words) >= max_words else words[:max_words]
        return " ".join(description_words).title()

    def generate_description(self, transcript: str) -> str:
        """Generate description from transcript using configured method."""
        method = self.title_config.get("method", "simple")
        fallback = self.title_config.get("fallback_to_simple", True)

        if method == "ollama":
            title = self.generate_ollama_title(transcript)
            if title:
                return title
            elif fallback:
                logging.info("Falling back to simple title generation")
                return self.generate_simple_title(transcript)
            else:
                return "Untitled Recording"
        else:
            return self.generate_simple_title(transcript)


class SpeakerIdentifier:
    """Handles speaker identification using voice samples."""

    def __init__(self, config: Config, audio_processor: AudioProcessor):
        self.config = config
        self.audio_processor = audio_processor
        self.speakers_path = Path(config.get("speakers_path"))
        self.speaker_features = {}
        self.speaker_models = {}
        self.scaler = StandardScaler()
        self.load_speaker_samples()
    
    def load_speaker_samples(self):
        """Load and process speaker samples with multiple files per speaker."""
        if not self.speakers_path.exists():
            self.speakers_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created speakers directory: {self.speakers_path}")
            return

        speaker_features_dict = {}  # Dictionary to hold multiple samples per speaker
        all_features = []

        # Load speaker samples from all supported formats
        supported_formats = self.config.get("supported_formats", [".mp3", ".wav", ".mp4"])

        for format_ext in supported_formats:
            pattern = f"*{format_ext}"
            for speaker_file in self.speakers_path.glob(pattern):
                # Extract speaker name (handle multiple files per speaker)
                speaker_name = speaker_file.stem
                # Remove numeric suffixes for multiple samples (e.g., "John_1" -> "John")
                speaker_name = re.sub(r'_\d+$', '', speaker_name)

                features = self.audio_processor.extract_audio_features(str(speaker_file))
                if features is not None:
                    if speaker_name not in speaker_features_dict:
                        speaker_features_dict[speaker_name] = []
                    speaker_features_dict[speaker_name].append(features)
                    all_features.append(features)
                    logging.info(f"Loaded speaker sample: {speaker_name} from {speaker_file.name}")

        if all_features:
            # Fit scaler on all features
            features_array = np.array(all_features)
            self.scaler.fit(features_array)

            # Process each speaker's samples
            for speaker_name, feature_list in speaker_features_dict.items():
                # Normalize all samples for this speaker
                normalized_features = []
                for features in feature_list:
                    norm_features = self.scaler.transform(features.reshape(1, -1))[0]
                    normalized_features.append(norm_features)

                # Store normalized features
                self.speaker_features[speaker_name] = normalized_features

                # Train GMM model for each speaker if multiple samples available
                if len(normalized_features) > 1:
                    try:
                        # Use configurable components for GMM
                        feat_config = self.config.get("feature_extraction", {})
                        max_components = feat_config.get("gmm_components", 3)
                        n_components = min(max_components, len(normalized_features))
                        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)
                        features_matrix = np.array(normalized_features)
                        gmm.fit(features_matrix)
                        self.speaker_models[speaker_name] = gmm
                        logging.info(f"Trained GMM model for {speaker_name} with {len(normalized_features)} samples")
                    except Exception as e:
                        logging.warning(f"Could not train GMM for {speaker_name}: {e}")
                        # Fallback to mean features
                        self.speaker_features[speaker_name] = [np.mean(normalized_features, axis=0)]
                else:
                    # Single sample case
                    self.speaker_features[speaker_name] = normalized_features

                logging.info(f"Loaded {len(feature_list)} samples for speaker: {speaker_name}")
    
    def calculate_speaker_similarity(self, test_features: np.ndarray, speaker_name: str) -> float:
        """Calculate similarity between test features and a speaker's model."""
        try:
            if speaker_name in self.speaker_models:
                # Use GMM log-likelihood for speakers with trained models
                gmm = self.speaker_models[speaker_name]
                log_likelihood = gmm.score(test_features.reshape(1, -1))
                # Convert log-likelihood to similarity score (0-1 range)
                similarity = 1.0 / (1.0 + np.exp(-log_likelihood))
                return similarity
            else:
                # Use multiple similarity metrics for single-sample speakers
                speaker_samples = self.speaker_features[speaker_name]
                similarities = []

                for speaker_features in speaker_samples:
                    # Cosine similarity
                    cos_sim = cosine_similarity(
                        test_features.reshape(1, -1),
                        speaker_features.reshape(1, -1)
                    )[0][0]

                    # Euclidean distance (converted to similarity)
                    eucl_dist = euclidean(test_features, speaker_features)
                    eucl_sim = 1.0 / (1.0 + eucl_dist)

                    # Combined similarity
                    combined_sim = 0.7 * cos_sim + 0.3 * eucl_sim
                    similarities.append(combined_sim)

                # Return best similarity among samples
                return max(similarities)

        except Exception as e:
            logging.error(f"Error calculating similarity for {speaker_name}: {e}")
            return 0.0

    def identify_speaker(self, audio_file: str) -> str:
        """Identify speaker from audio file using improved matching algorithm."""
        if not self.speaker_features:
            return "Miscellaneous Speakers"

        features = self.audio_processor.extract_audio_features(audio_file)
        if features is None:
            return "Miscellaneous Speakers"

        # Normalize features
        try:
            normalized_features = self.scaler.transform(features.reshape(1, -1))[0]
        except Exception as e:
            logging.error(f"Error normalizing features: {e}")
            return "Miscellaneous Speakers"

        # Calculate similarities for all speakers
        speaker_scores = {}
        threshold = self.config.get("speaker_similarity_threshold", 0.75)  # Lowered threshold

        for speaker_name in self.speaker_features.keys():
            similarity = self.calculate_speaker_similarity(normalized_features, speaker_name)
            speaker_scores[speaker_name] = similarity

        # Find best match
        if speaker_scores:
            best_speaker = max(speaker_scores, key=speaker_scores.get)
            best_similarity = speaker_scores[best_speaker]

            # Apply threshold with confidence margin
            if best_similarity > threshold:
                # Check if the best match is significantly better than second best
                sorted_scores = sorted(speaker_scores.values(), reverse=True)
                if len(sorted_scores) > 1:
                    confidence_margin = sorted_scores[0] - sorted_scores[1]
                    min_margin = self.config.get("speaker_confidence_margin", 0.05)
                    if confidence_margin < min_margin:  # Very close scores
                        logging.info(f"Speaker identification uncertain: {best_speaker} (similarity: {best_similarity:.3f}, margin: {confidence_margin:.3f})")
                        return "Miscellaneous Speakers"

                logging.info(f"Speaker identification: {best_speaker} (similarity: {best_similarity:.3f})")
                return best_speaker
            else:
                logging.info(f"Best match {best_speaker} below threshold (similarity: {best_similarity:.3f}, threshold: {threshold})")

        return "Miscellaneous Speakers"


class FileOrganizer:
    """Handles file organization and directory management."""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_base = Path(config.get("output_base_path"))
        self.speakers_path = Path(config.get("speakers_path"))
        self.talks_path = Path(config.get("talks_path"))
        self.transcripts_path = Path(config.get("transcripts_path"))
        self.raw_talks_path = Path(config.get("raw_talks_path"))
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directory structure."""
        for path in [self.output_base, self.speakers_path, self.talks_path, self.transcripts_path, self.raw_talks_path]:
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Directory ready: {path}")
    
    def organize_file(self, source_file: str, speaker_name: str, description: str, transcript: str):
        """Organize a processed audio file."""
        source_path = Path(source_file)
        year = str(datetime.now().year)
        
        # Create speaker directory for this year (for audio files)
        speaker_year_dir = self.talks_path / year / speaker_name
        speaker_year_dir.mkdir(parents=True, exist_ok=True)
        
        # Create speaker directory for transcripts
        transcript_speaker_year_dir = self.transcripts_path / year / speaker_name
        transcript_speaker_year_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        base_filename = f"{speaker_name} - {description}"
        audio_filename = f"{base_filename}{source_path.suffix}"
        transcript_filename = f"{base_filename} (Transcript).txt"
        
        # Copy audio file to talks directory
        audio_dest = speaker_year_dir / audio_filename
        shutil.copy2(source_file, audio_dest)
        logging.info(f"Organized audio: {audio_dest}")
        
        # Save transcript to separate transcripts directory
        transcript_dest = transcript_speaker_year_dir / transcript_filename
        with open(transcript_dest, 'w', encoding='utf-8') as f:
            f.write(transcript)
        logging.info(f"Saved transcript: {transcript_dest}")
        
        # Copy to raw talks for backup
        raw_dest = self.raw_talks_path / source_path.name
        if not raw_dest.exists():
            shutil.copy2(source_file, raw_dest)
            logging.info(f"Backed up to raw talks: {raw_dest}")


class AudioFileManager:
    """Main class that orchestrates the audio file management process."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = Config(config_path)
        self.setup_logging()
        
        self.audio_processor = AudioProcessor(self.config)
        self.transcriber = Transcriber(self.config)
        self.speaker_identifier = SpeakerIdentifier(self.config, self.audio_processor)
        self.file_organizer = FileOrganizer(self.config)
        
        self.last_run_file = self.config.get("last_run_file")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('audio_manager.log'),
                logging.StreamHandler()
            ]
        )
    
    def get_last_run_date(self) -> datetime:
        """Get the date of the last successful run."""
        if os.path.exists(self.last_run_file):
            try:
                with open(self.last_run_file, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data['last_run'])
            except Exception as e:
                logging.error(f"Error reading last run file: {e}")
        
        # Default to 30 days ago if no last run file
        return datetime.now() - timedelta(days=30)
    
    def update_last_run_date(self):
        """Update the last successful run date."""
        with open(self.last_run_file, 'w') as f:
            json.dump({'last_run': datetime.now().isoformat()}, f)
    
    def find_audio_files(self, directory: str, since_date: datetime) -> List[str]:
        """Find audio files modified since the given date."""
        audio_files = []
        directory_path = Path(os.path.expanduser(directory))
        
        if not directory_path.exists():
            logging.warning(f"Directory does not exist: {directory}")
            return audio_files
        
        for ext in self.config.get("supported_formats"):
            for file_path in directory_path.rglob(f"*{ext}"):
                if file_path.is_file():
                    # Check modification time
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mod_time >= since_date:
                        if self.audio_processor.is_valid_audio_file(str(file_path)):
                            audio_files.append(str(file_path))
                            logging.info(f"Found valid audio file: {file_path}")
        
        return audio_files
    
    def process_audio_file(self, file_path: str) -> bool:
        """Process a single audio file."""
        try:
            logging.info(f"Processing: {file_path}")
            
            # Transcribe audio
            transcript = self.transcriber.transcribe_audio(file_path)
            if not transcript:
                logging.error(f"Failed to transcribe: {file_path}")
                return False
            
            # Generate description
            description = self.transcriber.generate_description(transcript)
            
            # Identify speaker
            speaker_name = self.speaker_identifier.identify_speaker(file_path)
            
            # Organize file
            self.file_organizer.organize_file(file_path, speaker_name, description, transcript)
            
            logging.info(f"Successfully processed: {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return False
    
    def cleanup_old_files(self):
        """Clean up old files from raw talks directory."""
        cleanup_days = self.config.get("cleanup_days", 30)
        cutoff_date = datetime.now() - timedelta(days=cleanup_days)
        
        for file_path in self.file_organizer.raw_talks_path.rglob("*"):
            if file_path.is_file():
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mod_time < cutoff_date:
                    try:
                        file_path.unlink()
                        logging.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logging.error(f"Error deleting {file_path}: {e}")
    
    def run(self, force_full_scan: bool = False):
        """Run the complete audio file management process."""
        logging.info("Starting MLX Talks Categorizer")
        
        # Determine date range for processing
        if force_full_scan:
            since_date = datetime.min
            logging.info("Performing full scan of all files")
        else:
            since_date = self.get_last_run_date()
            logging.info(f"Processing files modified since: {since_date}")
        
        # Find audio files from all input sources
        all_files = []

        # Process each audio input directory
        audio_inputs = self.config.get("audio_inputs", [])
        for input_path in audio_inputs:
            input_files = self.find_audio_files(input_path, since_date)
            all_files.extend(input_files)
            logging.info(f"Found {len(input_files)} files in {input_path}")
        
        if not all_files:
            logging.info("No new audio files found to process")
            return
        
        logging.info(f"Processing {len(all_files)} audio files")
        
        # Process each file
        successful_count = 0
        for file_path in all_files:
            if self.process_audio_file(file_path):
                successful_count += 1
        
        logging.info(f"Successfully processed {successful_count}/{len(all_files)} files")
        
        # Cleanup old files
        self.cleanup_old_files()
        
        # Update last run date
        self.update_last_run_date()
        
        logging.info("MLX Talks Categorizer completed successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MLX Talks Categorizer - Audio File Management System")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--full-scan", action="store_true", help="Process all files, not just new ones")
    parser.add_argument("--setup", action="store_true", help="Setup initial configuration and directories")
    
    args = parser.parse_args()
    
    if args.setup:
        config = Config(args.config)
        print(f"Configuration created at: {args.config}")
        print("Please:")
        print("1. Add speaker sample files to the speakers/ directory")
        print("2. Adjust paths in config.json as needed")
        print("3. Run again without --setup to process audio files")
        return
    
    try:
        manager = AudioFileManager(args.config)
        manager.run(force_full_scan=args.full_scan)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
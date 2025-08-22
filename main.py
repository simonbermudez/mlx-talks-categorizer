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

# Whisper for transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Install with: pip install openai-whisper")

# Speaker identification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class Config:
    """Configuration management for the audio processing system."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.default_config = {
            "min_duration_minutes": 10,
            "supported_formats": [".mp3", ".wav", ".mp4"],
            "google_drive_path": "~/Google Drive/Audio",
            "local_audio_path": "~/Audio Hijack",
            "output_base_path": "./organized_talks",
            "speakers_path": "./organized_talks/speakers",
            "talks_path": "./organized_talks/talks",
            "transcripts_path": "./organized_talks/transcripts",
            "raw_talks_path": "./organized_talks/raw talks",
            "last_run_file": "last_run.json",
            "whisper_model": "medium",
            "speaker_similarity_threshold": 0.85,
            "cleanup_days": 30
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
    
    def extract_audio_features(self, file_path: str) -> Optional[np.ndarray]:
        """Extract MFCC features for speaker identification."""
        try:
            # For MP4 files, extract audio first if needed
            if file_path.lower().endswith('.mp4'):
                # librosa can handle MP4 files directly for audio extraction
                y, sr = librosa.load(file_path, sr=22050)
            else:
                y, sr = librosa.load(file_path, sr=22050)
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # Take mean across time dimension
            features = np.mean(mfccs, axis=1)
            return features
        except Exception as e:
            logging.error(f"Error extracting features from {file_path}: {e}")
            return None


class Transcriber:
    """Handles audio transcription using Whisper."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        if WHISPER_AVAILABLE:
            try:
                model_name = config.get("whisper_model", "medium")
                self.model = whisper.load_model(model_name)
                logging.info(f"Loaded Whisper model: {model_name}")
            except Exception as e:
                logging.error(f"Error loading Whisper model: {e}")
    
    def transcribe_audio(self, file_path: str) -> Optional[str]:
        """Transcribe audio file to text."""
        if not self.model:
            logging.error("Whisper model not available")
            return None
        
        try:
            result = self.model.transcribe(file_path)
            return result["text"].strip()
        except Exception as e:
            logging.error(f"Error transcribing {file_path}: {e}")
            return None
    
    def generate_description(self, transcript: str) -> str:
        """Generate 3-word description from transcript."""
        # Simple approach: extract key words and use first 3
        words = transcript.lower().split()
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Take first 3 meaningful words
        description_words = filtered_words[:3] if len(filtered_words) >= 3 else words[:3]
        return " ".join(description_words).title()


class SpeakerIdentifier:
    """Handles speaker identification using voice samples."""
    
    def __init__(self, config: Config, audio_processor: AudioProcessor):
        self.config = config
        self.audio_processor = audio_processor
        self.speakers_path = Path(config.get("speakers_path"))
        self.speaker_features = {}
        self.scaler = StandardScaler()
        self.load_speaker_samples()
    
    def load_speaker_samples(self):
        """Load and process speaker samples."""
        if not self.speakers_path.exists():
            self.speakers_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created speakers directory: {self.speakers_path}")
            return
        
        features_list = []
        speaker_names = []
        
        # Load speaker samples from all supported formats
        supported_formats = self.config.get("supported_formats", [".mp3", ".wav", ".mp4"])
        
        for format_ext in supported_formats:
            pattern = f"*{format_ext}"
            for speaker_file in self.speakers_path.glob(pattern):
                speaker_name = speaker_file.stem
                features = self.audio_processor.extract_audio_features(str(speaker_file))
                if features is not None:
                    self.speaker_features[speaker_name] = features
                    features_list.append(features)
                    speaker_names.append(speaker_name)
                    logging.info(f"Loaded speaker sample: {speaker_name} (from {speaker_file.suffix})")
        
        if features_list:
            # Fit scaler on all speaker features
            features_array = np.array(features_list)
            self.scaler.fit(features_array)
            # Normalize speaker features
            for speaker_name in speaker_names:
                self.speaker_features[speaker_name] = self.scaler.transform(
                    self.speaker_features[speaker_name].reshape(1, -1)
                )[0]
    
    def identify_speaker(self, audio_file: str) -> str:
        """Identify speaker from audio file."""
        if not self.speaker_features:
            return "Unknown"
        
        features = self.audio_processor.extract_audio_features(audio_file)
        if features is None:
            return "Unknown"
        
        # Normalize features
        normalized_features = self.scaler.transform(features.reshape(1, -1))[0]
        
        # Calculate similarities
        best_speaker = "Unknown"
        best_similarity = 0
        threshold = self.config.get("speaker_similarity_threshold", 0.85)
        
        for speaker_name, speaker_features in self.speaker_features.items():
            similarity = cosine_similarity(
                normalized_features.reshape(1, -1),
                speaker_features.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_speaker = speaker_name
        
        logging.info(f"Speaker identification: {best_speaker} (similarity: {best_similarity:.3f})")
        return best_speaker


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
        
        # Find audio files from all sources
        all_files = []
        
        # Google Drive files
        google_drive_path = self.config.get("google_drive_path")
        if google_drive_path:
            google_files = self.find_audio_files(google_drive_path, since_date)
            all_files.extend(google_files)
            logging.info(f"Found {len(google_files)} files in Google Drive")
        
        # Local Audio Hijack files
        local_audio_path = self.config.get("local_audio_path")
        if local_audio_path:
            local_files = self.find_audio_files(local_audio_path, since_date)
            all_files.extend(local_files)
            logging.info(f"Found {len(local_files)} files in local storage")
        
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
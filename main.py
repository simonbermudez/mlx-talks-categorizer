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
import numpy as np

# Title Generation
from openai import OpenAI

# Video processing for MP4 files
import subprocess
import re


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

# Speaker identification - Pyannote.audio
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Annotation
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Install with: pip install pyannote.audio")


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
            "speaker_similarity_threshold": 0.5,  # Pyannote similarity threshold (0-1 range)
            "cleanup_days": 30,
            "log_level": "INFO",  # Logging level: DEBUG, INFO, WARNING, ERROR
            "title_generation": {
                "method": "openai",
                "openai_api_key": "",
                "openai_model": "gpt-4o-mini",
                "max_title_words": 3,
                "fallback_to_simple": True
            },
            "pyannote": {
                "hf_token": "",  # Hugging Face token - get from https://huggingface.co/settings/tokens
                "model": "pyannote/speaker-diarization-3.1",  # Pyannote model to use
                "min_speakers": 1,  # Minimum number of speakers
                "max_speakers": 10  # Maximum number of speakers
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


class Transcriber:
    """Handles audio transcription using Whisper and title generation using OpenAI ChatGPT."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.title_config = config.get("title_generation", {})
        self.use_mlx = False
        self.hf_token = config.get("pyannote", {}).get("hf_token", "")

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
                # MLX Whisper transcription with HF token for model download
                if self.hf_token:
                    # Set HF token for model downloads
                    import os
                    os.environ['HF_TOKEN'] = self.hf_token
                result = mlx_whisper.transcribe(file_path, path_or_hf_repo=self.model)
                return result["text"].strip()
            else:
                # Regular Whisper transcription
                result = self.model.transcribe(file_path)
                return result["text"].strip()
        except Exception as e:
            logging.error(f"Error transcribing {file_path}: {e}")
            return None
    
    def generate_openai_title(self, transcript: str) -> Optional[str]:
        """Generate title using OpenAI ChatGPT API."""
        try:
            api_key = self.title_config.get("openai_api_key", "")
            model = self.title_config.get("openai_model", "gpt-4o-mini")
            max_words = self.title_config.get("max_title_words", 6)

            if not api_key:
                logging.error("OpenAI API key not configured")
                return None

            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)

            # Create a focused prompt for title generation
            prompt = f"""Based on this transcript, create a concise {max_words}-word title that captures the main topic or theme.

Transcript: {transcript[:1000]}...

Respond with only the {max_words}-word title, no explanations or additional text."""

            # Make API call to OpenAI
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates concise, descriptive titles from audio transcripts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=20,
                top_p=0.9
            )

            title = response.choices[0].message.content.strip()

            # Clean up the title - remove quotes, periods, etc.
            title = title.strip('"\'.,!?')

            # Ensure it's roughly the right length
            words = title.split()
            if len(words) > max_words:
                title = " ".join(words[:max_words])

            if title:
                logging.info(f"Generated OpenAI title: {title}")
                return title.title()
            else:
                logging.warning("OpenAI returned empty title")
                return None

        except Exception as e:
            logging.error(f"Error generating OpenAI title: {e}")
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

        if method == "openai":
            title = self.generate_openai_title(transcript)
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
    """Handles speaker identification using Pyannote.audio."""

    def __init__(self, config: Config, audio_processor: AudioProcessor):
        self.config = config
        self.audio_processor = audio_processor
        self.speakers_path = Path(config.get("speakers_path"))
        self.speaker_embeddings = {}  # Maps speaker name to embedding
        self.diarization_pipeline = None

        # Get Pyannote configuration
        pyannote_config = config.get("pyannote", {})
        self.hf_token = pyannote_config.get("hf_token", "")
        self.model_name = pyannote_config.get("model", "pyannote/speaker-diarization-3.1")
        self.min_speakers = pyannote_config.get("min_speakers", 1)
        self.max_speakers = pyannote_config.get("max_speakers", 10)

        if not PYANNOTE_AVAILABLE:
            logging.error("Pyannote.audio is not installed. Install with: pip install pyannote.audio")
            return

        if not self.hf_token:
            logging.error("Hugging Face token not configured. Get one from https://huggingface.co/settings/tokens")
            logging.error("Add it to config.json under pyannote.hf_token")
            return

        self.load_speaker_samples()

    def load_speaker_samples(self):
        """Load speaker samples and create embeddings using Pyannote."""
        if not PYANNOTE_AVAILABLE or not self.hf_token:
            logging.warning("Cannot load speaker samples: Pyannote not available or token missing")
            return

        if not self.speakers_path.exists():
            self.speakers_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created speakers directory: {self.speakers_path}")
            return

        try:
            # Load the diarization pipeline
            logging.info(f"Loading Pyannote model: {self.model_name}")
            self.diarization_pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token
            )

            # Enable GPU acceleration if available (MPS for Apple Silicon)
            import torch
            if torch.backends.mps.is_available():
                self.diarization_pipeline.to(torch.device("mps"))
                logging.info("Pyannote pipeline loaded successfully (using MPS GPU acceleration)")
            elif torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
                logging.info("Pyannote pipeline loaded successfully (using CUDA GPU acceleration)")
            else:
                logging.info("Pyannote pipeline loaded successfully (using CPU)")

            # Load speaker samples from all supported formats
            supported_formats = self.config.get("supported_formats", [".mp3", ".wav", ".mp4"])
            speaker_files = {}

            for format_ext in supported_formats:
                pattern = f"*{format_ext}"
                for speaker_file in self.speakers_path.glob(pattern):
                    # Extract speaker name (handle multiple files per speaker)
                    speaker_name = speaker_file.stem
                    # Remove numeric suffixes for multiple samples (e.g., "John_1" -> "John")
                    speaker_name = re.sub(r'_\d+$', '', speaker_name)

                    if speaker_name not in speaker_files:
                        speaker_files[speaker_name] = []
                    speaker_files[speaker_name].append(str(speaker_file))

            # Create embeddings for each speaker
            for speaker_name, files in speaker_files.items():
                logging.info(f"Creating embedding for speaker: {speaker_name} ({len(files)} sample(s))")

                # For now, we'll use the first file to create an embedding
                # In production, you might want to average embeddings from multiple files
                sample_file = files[0]

                try:
                    # Extract audio from MP4 if needed
                    audio_file = sample_file
                    temp_file = None
                    if sample_file.lower().endswith('.mp4'):
                        # Extract audio to temporary WAV file
                        import tempfile
                        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        temp_file.close()

                        result = subprocess.run([
                            'ffmpeg', '-i', sample_file, '-vn', '-acodec', 'pcm_s16le',
                            '-ar', '16000', '-ac', '1', '-y', temp_file.name
                        ], capture_output=True, text=True, timeout=60)

                        if result.returncode == 0:
                            audio_file = temp_file.name
                            logging.debug(f"Extracted audio from {sample_file} to {audio_file}")
                        else:
                            logging.error(f"Failed to extract audio from {sample_file}: {result.stderr}")
                            continue

                    # Run diarization to get embedding
                    diarization = self.diarization_pipeline(audio_file)

                    # Clean up temp file if created
                    if temp_file:
                        import os
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass

                    # Store the first speaker's embedding as reference
                    # This is a simplified approach - in production you'd want more sophisticated enrollment
                    self.speaker_embeddings[speaker_name] = sample_file
                    logging.info(f"Successfully enrolled speaker: {speaker_name}")

                except Exception as e:
                    logging.error(f"Error enrolling speaker {speaker_name}: {e}")
                    # Clean up temp file if it exists
                    if 'temp_file' in locals() and temp_file:
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
                    continue

            if self.speaker_embeddings:
                logging.info(f"Enrolled {len(self.speaker_embeddings)} speaker(s)")
            else:
                logging.warning("No speakers enrolled successfully")

        except Exception as e:
            logging.error(f"Error loading Pyannote pipeline: {e}")
            logging.error("Make sure you've accepted the model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            self.diarization_pipeline = None

    def identify_speaker(self, audio_file: str) -> str:
        """Identify speaker from audio file using Pyannote diarization."""
        if not self.diarization_pipeline:
            logging.warning("Diarization pipeline not initialized")
            return "Miscellaneous Speakers"

        if not self.speaker_embeddings:
            logging.warning("No enrolled speakers available")
            return "Miscellaneous Speakers"

        try:
            logging.debug(f"Running speaker diarization on: {audio_file}")

            # Run diarization on the audio file
            diarization = self.diarization_pipeline(
                audio_file,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )

            # Get the dominant speaker (speaker with most speaking time)
            speaker_durations = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

            if not speaker_durations:
                logging.info("No speakers detected in audio")
                return "Miscellaneous Speakers"

            # Find speaker with most speaking time
            dominant_speaker_id = max(speaker_durations, key=speaker_durations.get)
            total_duration = sum(speaker_durations.values())
            dominant_duration = speaker_durations[dominant_speaker_id]
            dominance_ratio = dominant_duration / total_duration if total_duration > 0 else 0

            logging.info(f"Detected {len(speaker_durations)} speaker(s)")
            logging.info(f"Dominant speaker: {dominant_speaker_id} ({dominance_ratio*100:.1f}% of audio)")

            # For now, if we have enrolled speakers, return the first one
            # In a full implementation, you'd compare embeddings to match speakers
            # This is a simplified version - see note below
            threshold = self.config.get("speaker_similarity_threshold", 0.5)

            if dominance_ratio >= threshold:
                # In a production system, you would:
                # 1. Extract embedding for the detected speaker
                # 2. Compare it to enrolled speaker embeddings
                # 3. Return the best match above threshold

                # For now, we return the first enrolled speaker if dominant
                # This is a placeholder - needs proper embedding comparison
                if len(self.speaker_embeddings) == 1:
                    matched_speaker = list(self.speaker_embeddings.keys())[0]
                    logging.info(f"Matched to enrolled speaker: {matched_speaker}")
                    return matched_speaker
                else:
                    logging.info("Multiple speakers enrolled - need embedding comparison (not yet implemented)")
                    return "Miscellaneous Speakers"
            else:
                logging.info(f"No dominant speaker found (threshold: {threshold})")
                return "Miscellaneous Speakers"

        except Exception as e:
            logging.error(f"Error identifying speaker from {audio_file}: {e}")
            import traceback
            logging.debug(traceback.format_exc())
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
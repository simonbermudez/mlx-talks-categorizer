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
import gc
import psutil

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


def log_memory_usage(context: str = ""):
    """Log current memory usage for monitoring (memory optimization helper)."""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        logging.info(f"Memory usage {context}: {mem_mb:.1f} MB")
    except Exception as e:
        logging.debug(f"Could not log memory usage: {e}")


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
        """Get duration of audio file in seconds (MEMORY OPTIMIZED - no file loading)."""
        try:
            # MEMORY OPTIMIZATION: Always use ffprobe to avoid loading entire file
            # This reads metadata only, not the actual audio data
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', file_path
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                duration = float(probe_data['format']['duration'])
                return duration
            else:
                # Fallback to librosa only if ffprobe fails
                logging.debug(f"ffprobe failed, using librosa for {file_path}")
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

    def extract_audio_chunk(self, file_path: str, start_sec: float, duration_sec: float, output_path: str) -> bool:
        """
        Extract a chunk of audio without loading the entire file (MEMORY OPTIMIZED).

        Args:
            file_path: Source audio file
            start_sec: Start time in seconds
            duration_sec: Duration to extract in seconds
            output_path: Output file path for the chunk

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use ffmpeg to extract chunk without loading full file into memory
            result = subprocess.run([
                'ffmpeg', '-ss', str(start_sec), '-t', str(duration_sec),
                '-i', file_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', output_path
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logging.debug(f"Extracted {duration_sec}s chunk from {file_path} starting at {start_sec}s")
                return True
            else:
                logging.error(f"Failed to extract chunk: {result.stderr}")
                return False
        except Exception as e:
            logging.error(f"Error extracting audio chunk: {e}")
            return False


class Transcriber:
    """Handles audio transcription using Whisper and title generation using OpenAI ChatGPT."""

    def __init__(self, config: Config, audio_processor: AudioProcessor = None):
        self.config = config
        self.audio_processor = audio_processor
        self.model = None
        self.model_name = config.get("whisper_model", "tiny")
        self.title_config = config.get("title_generation", {})
        self.use_mlx = False
        self.hf_token = config.get("pyannote", {}).get("hf_token", "")
        self.memory_opts = config.get("memory_optimizations", {})

        # Try MLX Whisper first for Apple Silicon optimization
        if MLX_WHISPER_AVAILABLE:
            try:
                # MLX Whisper uses different model naming convention
                mlx_model_name = self._convert_to_mlx_model_name(self.model_name)
                # For MLX Whisper, we store the model name rather than loading it
                self.model = mlx_model_name
                self.use_mlx = True
                logging.info(f"Configured MLX Whisper model: {mlx_model_name} (Apple Silicon optimized)")
            except Exception as e:
                logging.error(f"Error configuring MLX Whisper model: {e}")
                self.model = None

        # MEMORY OPTIMIZATION: Don't load regular Whisper at init - use lazy loading
        # If MLX not available, model will be None and loaded on first use
        if self.model is None and WHISPER_AVAILABLE:
            logging.info(f"Will lazy-load Whisper model '{self.model_name}' when needed")

    def _convert_to_mlx_model_name(self, model_name: str) -> str:
        """Convert regular Whisper model name to MLX Whisper format."""
        # MLX Whisper uses community models from Hugging Face with -mlx suffix
        mlx_model_mapping = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-v3-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo"
        }
        return mlx_model_mapping.get(model_name, f"mlx-community/whisper-{model_name}-mlx")
    
    def transcribe_audio(self, file_path: str) -> Optional[str]:
        """Transcribe audio file to text (with optional chunking for memory optimization)."""
        # MEMORY OPTIMIZATION: Check if chunked transcription is enabled
        if self.memory_opts.get("enable_chunked_transcription", False) and self.audio_processor:
            return self._transcribe_audio_chunked(file_path)
        else:
            return self._transcribe_audio_full(file_path)

    def _transcribe_audio_full(self, file_path: str) -> Optional[str]:
        """Transcribe entire audio file (original method)."""
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
                # MEMORY OPTIMIZATION: Lazy load regular Whisper model
                if self.model is None:
                    if not WHISPER_AVAILABLE:
                        logging.error("Whisper not available")
                        return None
                    logging.info(f"Loading Whisper model: {self.model_name}")
                    self.model = whisper.load_model(self.model_name)
                    logging.info(f"Whisper model loaded successfully")

                # Regular Whisper transcription
                result = self.model.transcribe(file_path)
                return result["text"].strip()
        except Exception as e:
            logging.error(f"Error transcribing {file_path}: {e}")
            return None

    def _transcribe_audio_chunked(self, file_path: str) -> Optional[str]:
        """
        Transcribe audio in chunks to minimize memory usage (MEMORY OPTIMIZED).
        Perfect for 30-60 minute files - processes 5-minute chunks at a time.
        """
        import tempfile

        try:
            # Get total duration
            duration = self.audio_processor.get_audio_duration(file_path)
            if duration <= 0:
                logging.error(f"Could not get duration for {file_path}")
                return None

            chunk_size = self.memory_opts.get("transcription_chunk_size_seconds", 300)  # 5 minutes default
            logging.info(f"Transcribing {duration:.1f}s audio in {chunk_size}s chunks (memory optimized)")

            all_transcripts = []
            num_chunks = int(np.ceil(duration / chunk_size))

            for i in range(num_chunks):
                start_time = i * chunk_size
                chunk_duration = min(chunk_size, duration - start_time)

                logging.debug(f"Processing chunk {i+1}/{num_chunks}: {start_time:.1f}s - {start_time + chunk_duration:.1f}s")

                # Create temporary file for chunk
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_chunk:
                    temp_chunk_path = temp_chunk.name

                try:
                    # Extract chunk using ffmpeg (no memory overhead)
                    if not self.audio_processor.extract_audio_chunk(file_path, start_time, chunk_duration, temp_chunk_path):
                        logging.error(f"Failed to extract chunk {i+1}")
                        continue

                    # Transcribe the small chunk
                    chunk_transcript = self._transcribe_audio_full(temp_chunk_path)
                    if chunk_transcript:
                        all_transcripts.append(chunk_transcript)

                    # Clean up chunk file immediately
                    os.unlink(temp_chunk_path)

                    # Force garbage collection after each chunk
                    gc.collect()

                except Exception as e:
                    logging.error(f"Error processing chunk {i+1}: {e}")
                    # Clean up on error
                    if os.path.exists(temp_chunk_path):
                        os.unlink(temp_chunk_path)
                    continue

            # Combine all transcripts
            full_transcript = " ".join(all_transcripts)
            logging.info(f"Chunked transcription complete: {len(full_transcript)} characters")
            return full_transcript.strip()

        except Exception as e:
            logging.error(f"Error in chunked transcription: {e}")
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
        self.embedding_model = None  # Separate embedding model for speaker identification
        self._models_loaded = False  # Track if models are loaded

        # Cache directory for speaker embeddings
        self.cache_dir = Path(config.get("cache_dir", "./cache"))
        self.embeddings_cache_file = self.cache_dir / "speaker_embeddings.pkl"

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

        # MEMORY OPTIMIZATION: Don't load models at init, use lazy loading
        # self.load_speaker_samples()

    def _ensure_models_loaded(self):
        """Lazy load models only when needed (memory optimization)."""
        if not self._models_loaded:
            logging.info("Lazy loading Pyannote models...")
            self.load_speaker_samples()
            self._models_loaded = True

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file for cache validation."""
        import hashlib
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logging.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""

    def _load_embeddings_cache(self) -> dict:
        """Load cached embeddings from disk."""
        if not self.embeddings_cache_file.exists():
            return {}

        try:
            import pickle
            with open(self.embeddings_cache_file, 'rb') as f:
                cache = pickle.load(f)
            logging.info(f"Loaded embeddings cache with {len(cache)} speaker(s)")
            return cache
        except Exception as e:
            logging.warning(f"Failed to load embeddings cache: {e}")
            return {}

    def _save_embeddings_cache(self, cache: dict):
        """Save embeddings cache to disk."""
        try:
            import pickle
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(cache, f)
            logging.info(f"Saved embeddings cache with {len(cache)} speaker(s)")
        except Exception as e:
            logging.error(f"Failed to save embeddings cache: {e}")

    def unload_models(self):
        """Unload models to free memory (aggressive memory optimization)."""
        import gc
        import torch

        # Move models to CPU before deleting to release GPU memory
        try:
            if self.diarization_pipeline is not None:
                if hasattr(self.diarization_pipeline, 'to'):
                    self.diarization_pipeline.to(torch.device("cpu"))
                del self.diarization_pipeline
                self.diarization_pipeline = None
                logging.debug("Unloaded diarization pipeline")

            if self.embedding_model is not None:
                if hasattr(self.embedding_model, 'to'):
                    self.embedding_model.to(torch.device("cpu"))
                del self.embedding_model
                self.embedding_model = None
                logging.debug("Unloaded embedding model")
        except Exception as e:
            logging.debug(f"Error moving models to CPU: {e}")

        self._models_loaded = False

        # Multiple rounds of garbage collection for better cleanup
        for _ in range(3):
            gc.collect()

        # Clear GPU/MPS cache
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logging.debug("Cleared MPS cache")
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("Cleared CUDA cache")
        except Exception as e:
            logging.debug(f"Error clearing GPU cache: {e}")

    def load_speaker_samples(self):
        """Load speaker samples and create embeddings using Pyannote (with persistent caching)."""
        if not PYANNOTE_AVAILABLE or not self.hf_token:
            logging.warning("Cannot load speaker samples: Pyannote not available or token missing")
            return

        if not self.speakers_path.exists():
            self.speakers_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created speakers directory: {self.speakers_path}")
            return

        try:
            # Load existing cache
            cache = self._load_embeddings_cache()

            # Collect all current speaker files and compute their hashes
            supported_formats = self.config.get("supported_formats", [".mp3", ".wav", ".mp4"])
            speaker_files = {}
            current_file_hashes = {}  # Maps speaker_name -> {file_path: hash}

            for format_ext in supported_formats:
                pattern = f"*{format_ext}"
                for speaker_file in self.speakers_path.glob(pattern):
                    # Extract speaker name (handle multiple files per speaker)
                    speaker_name = speaker_file.stem
                    # Remove numeric suffixes for multiple samples (e.g., "John_1" -> "John")
                    speaker_name = re.sub(r'_\d+$', '', speaker_name)

                    if speaker_name not in speaker_files:
                        speaker_files[speaker_name] = []
                        current_file_hashes[speaker_name] = {}

                    file_path = str(speaker_file)
                    speaker_files[speaker_name].append(file_path)
                    current_file_hashes[speaker_name][file_path] = self._compute_file_hash(file_path)

            # Determine which speakers need new embeddings
            speakers_to_process = {}
            speakers_from_cache = {}

            for speaker_name, files in speaker_files.items():
                # Check if speaker exists in cache with matching file hashes
                if speaker_name in cache:
                    cached_data = cache[speaker_name]
                    cached_hashes = cached_data.get('file_hashes', {})

                    # Check if all files match the cache
                    if cached_hashes == current_file_hashes[speaker_name]:
                        # Cache is valid, use it
                        speakers_from_cache[speaker_name] = cached_data['embedding']
                        logging.info(f"Using cached embedding for speaker: {speaker_name}")
                        continue

                # Cache miss or invalidated - need to regenerate
                speakers_to_process[speaker_name] = files
                logging.info(f"Will generate new embedding for speaker: {speaker_name}")

            # Load speakers from cache into memory
            self.speaker_embeddings.update(speakers_from_cache)

            # If there are no new speakers to process, still need to load embedding model for identification
            if not speakers_to_process:
                logging.info(f"All {len(self.speaker_embeddings)} speaker(s) loaded from cache")
                # Load ONLY the embedding model (not diarization) for speaker identification
                from pyannote.audio import Inference
                import torch

                embedding_models = [
                    "pyannote/embedding",  # Preferred but gated
                    "pyannote/wespeaker-voxceleb-resnet34-LM",  # Open alternative
                ]

                for model_id in embedding_models:
                    try:
                        logging.info(f"Loading embedding model for speaker identification: {model_id}")
                        self.embedding_model = Inference(model_id, use_auth_token=self.hf_token)

                        # Move embedding model to GPU if available
                        if torch.backends.mps.is_available():
                            self.embedding_model.to(torch.device("mps"))
                        elif torch.cuda.is_available():
                            self.embedding_model.to(torch.device("cuda"))

                        logging.info(f"Speaker embedding model loaded successfully: {model_id}")
                        break  # Success, exit loop
                    except Exception as e:
                        logging.warning(f"Failed to load {model_id}: {e}")
                        if model_id == embedding_models[-1]:
                            # Last model failed, raise error
                            raise Exception(f"Could not load any embedding model. Please accept terms at https://hf.co/pyannote/embedding or https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM")
                        continue

                return  # Done - embeddings loaded from cache and model ready for identification

            # Only load BOTH models if we have new speakers to process
            logging.info(f"Loading Pyannote models for {len(speakers_to_process)} new/updated speaker(s)")

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

            # Load a separate embedding model for robust speaker identification
            from pyannote.audio import Inference

            # Try multiple embedding models in order of preference
            embedding_models = [
                "pyannote/embedding",  # Preferred but gated
                "pyannote/wespeaker-voxceleb-resnet34-LM",  # Open alternative
            ]

            for model_id in embedding_models:
                try:
                    logging.info(f"Attempting to load embedding model: {model_id}")
                    self.embedding_model = Inference(model_id, use_auth_token=self.hf_token)

                    # Move embedding model to same device as diarization
                    if torch.backends.mps.is_available():
                        self.embedding_model.to(torch.device("mps"))
                    elif torch.cuda.is_available():
                        self.embedding_model.to(torch.device("cuda"))

                    logging.info(f"Speaker embedding model loaded successfully: {model_id}")
                    break  # Success, exit loop
                except Exception as e:
                    logging.warning(f"Failed to load {model_id}: {e}")
                    if model_id == embedding_models[-1]:
                        # Last model failed, raise error
                        raise Exception(f"Could not load any embedding model. Please accept terms at https://hf.co/pyannote/embedding or https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM")
                    continue

            # Create embeddings for new/updated speakers only
            for speaker_name, files in speakers_to_process.items():
                logging.info(f"Creating embedding for speaker: {speaker_name} ({len(files)} sample(s))")

                # Process all files for this speaker and average embeddings
                all_embeddings = []

                for sample_file in files[:3]:  # Use up to 3 files per speaker
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

                        # Extract embedding directly from the entire audio file
                        # The Inference model handles variable-length audio automatically
                        import numpy as np
                        embedding = self.embedding_model({"audio": audio_file})

                        # Convert to numpy array if needed
                        if hasattr(embedding, 'data'):
                            embedding = embedding.data
                        if torch.is_tensor(embedding):
                            embedding = embedding.cpu().numpy()

                        # Handle different embedding shapes
                        if len(embedding.shape) > 1:
                            # If multiple embeddings (e.g., sliding window), average them
                            embedding = np.mean(embedding, axis=0)

                        all_embeddings.append(embedding.flatten())
                        logging.debug(f"Extracted embedding from {sample_file} (shape: {embedding.shape})")

                        # Clean up temp file if created
                        if temp_file:
                            import os
                            try:
                                os.unlink(temp_file.name)
                            except:
                                pass

                    except Exception as e:
                        logging.warning(f"Failed to extract embedding from {sample_file}: {e}")
                        # Clean up temp file if it exists
                        if 'temp_file' in locals() and temp_file:
                            try:
                                import os
                                os.unlink(temp_file.name)
                            except:
                                pass
                        continue

                if all_embeddings:
                    # Average all embeddings for this speaker
                    avg_embedding = np.mean(all_embeddings, axis=0)
                    # MEMORY OPTIMIZATION: Store embeddings as float16 to reduce memory usage
                    embedding_float16 = avg_embedding.astype(np.float16)
                    self.speaker_embeddings[speaker_name] = embedding_float16

                    # Update cache with new embedding and file hashes
                    cache[speaker_name] = {
                        'embedding': embedding_float16,
                        'file_hashes': current_file_hashes[speaker_name],
                        'timestamp': datetime.now().isoformat()
                    }

                    logging.info(f"Successfully enrolled speaker: {speaker_name} (embedding shape: {avg_embedding.shape})")
                else:
                    logging.warning(f"No valid embeddings extracted for {speaker_name}")

            # Save updated cache to disk
            self._save_embeddings_cache(cache)

            total_speakers = len(self.speaker_embeddings)
            cached_count = len(speakers_from_cache)
            new_count = len([s for s in speakers_to_process if s in self.speaker_embeddings])

            if total_speakers:
                logging.info(f"Enrolled {total_speakers} speaker(s) total ({cached_count} from cache, {new_count} newly generated)")
            else:
                logging.warning("No speakers enrolled successfully")

        except Exception as e:
            logging.error(f"Error loading Pyannote pipeline: {e}")
            logging.error("Make sure you've accepted the model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            logging.error("and https://huggingface.co/pyannote/embedding")
            self.diarization_pipeline = None
            self.embedding_model = None

    def identify_speaker(self, audio_file: str) -> str:
        """Identify speaker from audio file using Pyannote embedding model (MEMORY OPTIMIZED)."""
        # MEMORY OPTIMIZATION: Lazy load models only when needed
        self._ensure_models_loaded()

        if not hasattr(self, 'embedding_model') or not self.embedding_model:
            logging.warning("Embedding model not initialized")
            return "Miscellaneous Speakers"

        if not self.speaker_embeddings:
            logging.warning("No enrolled speakers available")
            return "Miscellaneous Speakers"

        try:
            logging.debug(f"Identifying speaker in: {audio_file}")

            # MEMORY OPTIMIZATION: Extract only a small sample for speaker ID
            # Most audio files are single speaker, so we only need ~30 seconds
            import tempfile
            memory_opts = self.config.get("memory_optimizations", {})
            sample_duration = memory_opts.get("speaker_identification_sample_seconds", 30)
            sample_offset = memory_opts.get("speaker_identification_sample_offset", 60)

            # Check file duration and adjust offset for short files
            file_duration = self.audio_processor.get_audio_duration(audio_file)
            required_duration = sample_offset + sample_duration

            if file_duration < required_duration:
                # For short files, extract from the beginning or middle
                if file_duration < sample_duration:
                    # Very short file - use entire file
                    sample_offset = 0
                    sample_duration = file_duration
                    logging.info(f"Short file ({file_duration:.1f}s) - using entire file for speaker ID")
                else:
                    # Adjust offset to extract from middle
                    sample_offset = max(0, (file_duration - sample_duration) / 2)
                    logging.info(f"Short file ({file_duration:.1f}s) - extracting {sample_duration}s from {sample_offset:.1f}s offset")
            else:
                logging.info(f"Extracting {sample_duration}s sample from {sample_offset}s offset for speaker ID (memory optimized)")

            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            processing_file = temp_file.name

            # Extract a small sample from the audio
            result = subprocess.run([
                'ffmpeg', '-ss', str(sample_offset), '-t', str(sample_duration),
                '-i', audio_file, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', processing_file
            ], capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                logging.error(f"Failed to extract audio sample: {result.stderr}")
                os.unlink(processing_file)
                return "Miscellaneous Speakers"

            # Extract embedding from the small sample
            # The Inference model handles variable-length audio automatically
            import numpy as np
            import torch

            embedding = self.embedding_model({"audio": processing_file})

            # Convert to numpy array if needed
            if hasattr(embedding, 'data'):
                embedding = embedding.data
            if torch.is_tensor(embedding):
                embedding = embedding.cpu().numpy()

            # Handle different embedding shapes
            if len(embedding.shape) > 1:
                # If multiple embeddings (e.g., sliding window), average them
                unknown_embedding = np.mean(embedding, axis=0).flatten()
            else:
                unknown_embedding = embedding.flatten()

            logging.debug(f"Extracted embedding for unknown speaker (shape: {unknown_embedding.shape})")

            # Compare to enrolled speakers using cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity

            best_match = None
            best_similarity = -1

            for speaker_name, enrolled_embedding in self.speaker_embeddings.items():
                # MEMORY OPTIMIZATION: Convert float16 back to float32 for comparison
                enrolled_emb = enrolled_embedding.astype(np.float32) if enrolled_embedding.dtype == np.float16 else enrolled_embedding
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    unknown_embedding.reshape(1, -1),
                    enrolled_emb.reshape(1, -1)
                )[0][0]

                logging.debug(f"Similarity with {speaker_name}: {similarity:.3f}")

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker_name

            # Check if best match is above threshold
            threshold = self.config.get("speaker_similarity_threshold", 0.85)

            # Clean up temp file
            try:
                os.unlink(processing_file)
            except:
                pass

            if best_match and best_similarity >= threshold:
                logging.info(f"Matched to speaker: {best_match} (similarity: {best_similarity:.3f})")
                return best_match
            else:
                if best_match:
                    logging.info(f"Best match: {best_match} (similarity: {best_similarity:.3f}) below threshold ({threshold})")
                else:
                    logging.info("No speaker match found")
                return "Miscellaneous Speakers"

        except Exception as e:
            logging.error(f"Error identifying speaker from {audio_file}: {e}")
            import traceback
            logging.debug(traceback.format_exc())
            # Clean up temp file if it exists
            if 'processing_file' in locals() and processing_file:
                try:
                    os.unlink(processing_file)
                except:
                    pass
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
        self.transcriber = Transcriber(self.config, self.audio_processor)  # Pass audio_processor for chunking
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
        import gc

        try:
            logging.info(f"Processing: {file_path}")
            log_memory_usage("before processing")

            # Transcribe audio
            transcript = self.transcriber.transcribe_audio(file_path)
            if not transcript:
                logging.error(f"Failed to transcribe: {file_path}")
                return False

            # MEMORY OPTIMIZATION: Clear transcription model from memory if using regular Whisper
            if hasattr(self.transcriber, 'model') and self.transcriber.model is not None:
                if not self.transcriber.use_mlx:
                    # Unload regular Whisper model after use
                    del self.transcriber.model
                    self.transcriber.model = None
                    gc.collect()
                    logging.debug("Unloaded Whisper model after transcription")

            log_memory_usage("after transcription")

            # Generate description
            description = self.transcriber.generate_description(transcript)

            # Identify speaker
            speaker_name = self.speaker_identifier.identify_speaker(file_path)

            log_memory_usage("after speaker identification")

            # MEMORY OPTIMIZATION: Unload Pyannote models after use
            self.speaker_identifier.unload_models()

            log_memory_usage("after model unloading")

            # Organize file
            self.file_organizer.organize_file(file_path, speaker_name, description, transcript)

            # MEMORY OPTIMIZATION: Clear large variables and force garbage collection
            del transcript
            del description
            del speaker_name

            # Multiple rounds of garbage collection
            for _ in range(3):
                gc.collect()

            # Clear GPU/MPS cache aggressively
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    torch.mps.synchronize()  # Wait for all operations to complete
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                logging.debug(f"Error clearing GPU cache: {e}")

            log_memory_usage("after cleanup")

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
        log_memory_usage("at startup")

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
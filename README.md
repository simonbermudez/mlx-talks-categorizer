# MLX Talks Categorizer

An intelligent audio file management system that automatically organizes, transcribes, and categorizes MP3/WAV/MP4 files using AI and machine learning, optimized for Apple Silicon with MLX.

## 🎯 Project Overview

The MLX Talks Categorizer is designed to solve the challenge of managing large volumes of audio files from multiple sources. It automatically:

- **Aggregates** audio files from multiple configured input directories (Google Drive, local storage, etc.)
- **Filters** files by duration (10+ minutes minimum)
- **Transcribes** audio content using OpenAI Whisper
- **Identifies** speakers using voice pattern recognition
- **Organizes** files into a structured directory system
- **Archives** processed files with descriptive naming

## 🏗️ Architecture

### Core Components

1. **AudioProcessor**: Handles file discovery, validation, and audio feature extraction
2. **Transcriber**: Uses Whisper for speech-to-text conversion and description generation
3. **SpeakerIdentifier**: MFCC-based voice recognition with cosine similarity matching
4. **FileOrganizer**: Manages directory structure and file placement
5. **AudioFileManager**: Main orchestrator coordinating all components

### Technology Stack

- **Python 3.13+** - Core runtime
- **MLX** - Apple Silicon optimization framework
- **MLX Whisper** - Optimized speech-to-text transcription for Apple Silicon
- **OpenAI Whisper** - Fallback speech-to-text transcription
- **OpenAI ChatGPT** - AI-powered intelligent title generation
- **librosa** - Audio analysis and feature extraction
- **scikit-learn** - Machine learning algorithms for speaker identification
- **ffmpeg** - Audio format support and processing

## 📁 Directory Structure

The system creates and maintains this organizational structure:

```
organized_talks/
├── speakers/                           # Voice samples for training
│   ├── [SPEAKER_NAME].[mp3|wav|mp4]   # Reference voice samples (any format)
│   └── ...
├── talks/                             # Organized processed audio files
│   └── [YEAR]/                        # Year-based organization
│       └── [SPEAKER_NAME]/            # Speaker-specific folders
│           ├── [SPEAKER] - [3_WORD_DESC].[mp3|wav|mp4]
│           └── ...
├── transcripts/                       # Organized transcript files (separate)
│   └── [YEAR]/                        # Year-based organization
│       └── [SPEAKER_NAME]/            # Speaker-specific folders
│           ├── [SPEAKER] - [3_WORD_DESC] (Transcript).txt
│           └── ...
└── raw talks/                         # Backup copies of original files
    ├── original_file1.mp3
    └── ...
```

## 🚀 Installation

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3) for optimal performance
- **Python 3.13+**
- **Homebrew** package manager
- **ffmpeg** for audio processing

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd mlx-talks-categorizer
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3a: Install MLX Whisper (Recommended for Apple Silicon)

For optimal performance on Apple Silicon, install MLX Whisper:

```bash
pip install mlx-whisper
```

This provides 30-40% faster transcription compared to regular Whisper on Apple Silicon devices.

### Step 4: Install ffmpeg

```bash
brew install ffmpeg
```

### Step 5: Initial Setup

```bash
python main.py --setup
```

This creates:
- `config.json` - Configuration file
- Directory structure for organized talks
- Sample configuration with default paths

## ⚙️ Configuration

Edit `config.json` to customize behavior:

```json
{
  "min_duration_minutes": 10,                      // Minimum file duration to process
  "supported_formats": [".mp3", ".wav", ".mp4"],  // Supported audio/video formats
  "audio_inputs": [                                // List of audio input directories
    "~/Google Drive/Audio",                        // Google Drive source
    "~/Audio Hijack",                              // Local Audio Hijack folder
    "~/Downloads/Audio"                            // Additional source (example)
  ],
  "output_base_path": "./organized_talks",         // Output directory
  "speakers_path": "./organized_talks/speakers",   // Speaker samples location
  "talks_path": "./organized_talks/talks",         // Processed audio files location
  "transcripts_path": "./organized_talks/transcripts", // Transcripts location (separate)
  "raw_talks_path": "./organized_talks/raw talks", // Backup location
  "last_run_file": "last_run.json",               // Incremental processing tracker
  "whisper_model": "medium",                       // Whisper model size
  "speaker_similarity_threshold": 0.85,           // Speaker matching threshold
  "cleanup_days": 30,                              // Days before cleanup
  "title_generation": {                            // Title generation configuration
    "method": "openai",                           // Method: "openai" or "simple"
    "openai_api_key": "",                         // Your OpenAI API key
    "openai_model": "gpt-4o-mini",                // OpenAI model to use
    "max_title_words": 3,                         // Maximum words in generated titles
    "fallback_to_simple": true                    // Fallback to simple method if API fails
  }
}
```

### Configuration Options Explained

- **min_duration_minutes**: Only processes files longer than this duration
- **audio_inputs**: Array of input directories to scan for audio files (supports unlimited sources)
- **whisper_model**: Available options: `tiny`, `base`, `small`, `medium`, `large`
- **speaker_similarity_threshold**: Cosine similarity threshold (0.0-1.0) for speaker matching
- **cleanup_days**: Automatically removes files from raw talks older than this
- **title_generation**: Configuration for AI-powered title generation
  - **method**: Use "openai" for ChatGPT-generated titles or "simple" for keyword extraction
  - **openai_api_key**: Your OpenAI API key (required for AI title generation)
  - **openai_model**: OpenAI model to use (e.g., gpt-4o-mini, gpt-4o, gpt-4-turbo)
  - **max_title_words**: Maximum number of words in generated titles
  - **fallback_to_simple**: Whether to use simple method if OpenAI API fails

## 🤖 OpenAI ChatGPT Setup for AI Title Generation

### Getting Your OpenAI API Key

1. **Sign up for OpenAI** at [platform.openai.com](https://platform.openai.com)
2. **Navigate to API Keys**: Go to your account settings → API Keys
3. **Create a new API key**: Click "Create new secret key"
4. **Copy and save**: Copy the key immediately (you won't be able to see it again)
5. **Add to config.json**: Paste the key into the `openai_api_key` field

### OpenAI Configuration

The system uses OpenAI ChatGPT to generate intelligent, context-aware titles from transcript content. Configure in `config.json`:

- **openai_api_key**: Your OpenAI API key (required)
- **openai_model**: Choose based on your needs and budget:
  - `gpt-4o-mini` - Fast and cost-effective, excellent quality (recommended)
  - `gpt-4o` - Higher quality, more expensive
  - `gpt-4-turbo` - Very high quality, most expensive
- **fallback_to_simple**: Recommended `true` for reliability

### Title Generation Features

- **AI-Powered**: Uses GPT understanding to create meaningful, accurate titles
- **Context-Aware**: Analyzes full transcript for thematic understanding
- **Configurable Length**: Set `max_title_words` to control title length
- **Automatic Fallback**: Falls back to keyword extraction if API fails
- **Error Handling**: Robust handling of network issues and API errors

### Cost Considerations

OpenAI ChatGPT API is usage-based:
- **gpt-4o-mini**: ~$0.00015 per audio file (most cost-effective)
- **gpt-4o**: ~$0.0015 per audio file
- Processing uses only the first 1000 characters of each transcript
- See [OpenAI Pricing](https://openai.com/pricing) for current rates

## 🎤 Speaker Identification Setup

### Adding Speaker Samples

1. Record or obtain clean audio samples of each speaker (30+ seconds recommended)
2. Save files named with the speaker's name: `John_Doe.mp3`, `Jane_Smith.wav`, or `Bob_Jones.mp4`
3. Place in the `organized_talks/speakers/` directory

### Speaker Sample Requirements

- **Duration**: 30+ seconds for better accuracy
- **Quality**: Clear speech, minimal background noise
- **Format**: MP3, WAV, or MP4 (video files will extract audio automatically)
- **Content**: Natural speech patterns representative of the speaker

### How Speaker Identification Works

1. **Feature Extraction**: Uses MFCC (Mel-Frequency Cepstral Coefficients) to extract voice characteristics
2. **Normalization**: Applies StandardScaler for consistent feature scaling
3. **Similarity Matching**: Calculates cosine similarity between unknown and known speaker features
4. **Threshold Filtering**: Only matches above the configured similarity threshold

## 📋 Usage

### Basic Operation

```bash
# Process new files since last run
python main.py

# Process all files (full scan)
python main.py --full-scan

# Setup initial configuration
python main.py --setup
```

### Command Line Options

- `--config CONFIG_FILE` - Specify custom configuration file path
- `--full-scan` - Process all files, ignoring last run date
- `--setup` - Initialize configuration and directory structure

### Typical Workflow

1. **Initial Setup**: Run `--setup` to create configuration
2. **Add Speaker Samples**: Place reference audio files in speakers directory
3. **Configure Paths**: Edit `config.json` with your actual source directories
4. **First Run**: Execute `--full-scan` to process existing files
5. **Regular Processing**: Run without flags for incremental updates

## 🕒 Automated Scheduling

Set up automatic processing with the included cronjob setup script:

```bash
./setup_cronjob.sh
```

### Schedule Options Available:
- **Daily at 2:00 AM** - Traditional off-hours processing
- **Daily at 9:00 PM** - End-of-day processing when new files are added
- **Every 6 hours** - Frequent processing for active workflows
- **Weekly on Sunday at 3:00 AM** - Light processing for occasional use
- **Every 2 hours** - For testing and development
- **Custom schedule** - Enter your own cron expression

The setup script automatically:
- Detects your project directory and creates absolute paths
- Updates `config.json` with proper configurations
- Creates wrapper and monitoring scripts
- Configures either crontab or macOS launchd
- Tests the setup before activation

For detailed setup instructions, see [SETUP_CRONJOB.md](SETUP_CRONJOB.md).

## 🔍 Processing Pipeline

### 1. File Discovery
- Scans configured source directories recursively
- Filters by supported file extensions (`.mp3`, `.wav`, `.mp4`)
- Checks modification time against last run date
- Validates minimum duration requirement

### 2. Audio Analysis
- Extracts duration using librosa
- Generates MFCC features for speaker identification
- Validates audio file integrity

### 3. Transcription
- Uses MLX Whisper for Apple Silicon optimized speech-to-text conversion (30-40% faster)
- Falls back to OpenAI Whisper if MLX Whisper unavailable
- Supports multiple model sizes for speed/accuracy trade-offs
- Generates intelligent titles from transcript content using OpenAI ChatGPT
- Filters common stop words for meaningful descriptions

### 4. Speaker Identification
- Compares audio features against known speaker samples
- Uses cosine similarity for voice pattern matching
- Applies configurable threshold for identification confidence
- Falls back to "Miscellaneous Speakers" for unmatched speakers

### 5. File Organization
- Creates year-based directory structure
- Generates descriptive filenames with speaker and topic
- Saves audio files in `talks/` directory
- Saves transcripts separately in `transcripts/` directory
- Maintains backup copies in raw talks directory

## 📊 Performance Characteristics

### Tested Performance

**Hardware**: Apple Silicon (M1/M2/M3)
**Test Results**:
- Processing speed: ~3-4 seconds per 5-second audio file
- Memory usage: ~500MB during processing
- Speaker identification accuracy: 100% on test samples
- Transcription accuracy: High quality with Whisper medium model

### Model Comparison

| Whisper Model | Speed | Accuracy | Memory Usage |
|---------------|-------|----------|--------------|
| tiny          | Fastest | Basic | ~200MB |
| base          | Fast | Good | ~300MB |
| small         | Medium | Better | ~400MB |
| medium        | Slower | High | ~600MB |
| large         | Slowest | Highest | ~1GB+ |

### Scalability

- **Incremental Processing**: Only processes new/modified files
- **Batch Processing**: Handles multiple files in sequence
- **Memory Efficient**: Processes one file at a time
- **MLX Optimization**: Leverages Apple Silicon GPU acceleration

## 🛠️ Development Findings

### Key Technical Decisions

1. **MFCC for Speaker ID**: Chose Mel-Frequency Cepstral Coefficients over spectrograms for better speaker discrimination
2. **Cosine Similarity**: Provides robust speaker matching with configurable thresholds
3. **Whisper Integration**: Balances accuracy and processing speed with model size options
4. **Incremental Processing**: Tracks modification times to avoid reprocessing files
5. **MLX Optimization**: Leverages Apple Silicon acceleration with CPU fallback

### Challenges Overcome

1. **ffmpeg Dependency**: Whisper requires ffmpeg for audio format support
2. **Virtual Environment**: macOS externally-managed environment requires venv
3. **File Duration Validation**: librosa integration for accurate duration checking
4. **Feature Normalization**: StandardScaler ensures consistent speaker identification
5. **Path Handling**: Robust path expansion and validation across different systems

### Testing Results

**Test Configuration**:
- 2 audio files (5.4 seconds each)
- 4 speaker samples (Jason, Jorge, Melissa, Samantha)
- Whisper medium model

**Results**:
- ✅ File discovery and filtering working correctly
- ✅ Audio duration validation accurate
- ✅ Transcription quality excellent
- ✅ Speaker identification 100% accurate
- ✅ Directory structure created as specified
- ✅ File naming conventions followed
- ✅ Backup system functioning

## 🐛 Troubleshooting

### Common Issues

**Error: `[Errno 2] No such file or directory: 'ffmpeg'`**
```bash
brew install ffmpeg
```

**Error: `externally-managed-environment`**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Low Speaker Identification Accuracy**
- Ensure speaker samples are high quality (30+ seconds)
- Check `speaker_similarity_threshold` in config (try lowering to 0.7-0.8)
- Verify speaker sample files are properly named and placed

**Files Not Being Processed**
- Check `min_duration_minutes` setting
- Verify source directory paths in config.json
- Run with `--full-scan` to ignore date filtering
- Check audio file formats are supported

**Memory Issues**
- Use smaller Whisper model (`small` or `base`)
- Use faster OpenAI model (`gpt-4o-mini` instead of `gpt-4o`)
- Process files in smaller batches
- Monitor system memory during large operations

**OpenAI API Issues**
- Verify API key is correctly set in config.json
- Check API key has billing enabled at [OpenAI Platform](https://platform.openai.com/account/billing)
- Ensure you have sufficient API credits
- Check network connectivity
- Review error logs for specific API error messages

**MLX Whisper Issues**
- If seeing "FP16 is not supported on CPU" warning, install MLX Whisper: `pip install mlx-whisper`
- Verify you're on Apple Silicon: the system will automatically fall back to regular Whisper on Intel Macs
- For model download issues, ensure you have internet connection for first-time model downloads

### Debug Mode

Add debug logging by modifying the logging level in `main.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_manager.log'),
        logging.StreamHandler()
    ]
)
```

## 📈 Future Enhancements

### Planned Features

1. **Web Interface**: Browser-based management and monitoring
2. **Advanced Speaker Training**: Deep learning models for better identification
3. **Multi-language Support**: Whisper language detection and transcription
4. **Cloud Storage Integration**: Direct processing from cloud providers
5. **Batch Processing UI**: Progress tracking and queue management
6. **Audio Quality Enhancement**: Noise reduction and normalization
7. **Metadata Extraction**: Date, location, and topic classification
8. **Export Formats**: Support for various transcript and summary formats

### Performance Optimizations

1. **Parallel Processing**: Multi-threaded file processing
2. **GPU Acceleration**: Full MLX integration for all components
3. **Caching System**: Store computed features and transcriptions
4. **Streaming Processing**: Handle large files without loading entirely
5. **Incremental Training**: Update speaker models with new samples

## 📝 Logging and Monitoring

### Log Files

- `audio_manager.log` - Main application log
- `last_run.json` - Tracks processing timestamps

### Log Levels

- **INFO**: Normal operation, file processing status
- **WARNING**: Non-critical issues, missing directories
- **ERROR**: Processing failures, file access issues
- **DEBUG**: Detailed processing information

### Monitoring

Track these metrics for system health:
- Files processed per run
- Processing time per file
- Speaker identification accuracy
- Transcription success rate
- Storage usage trends

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests before submitting

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include docstrings for all classes and methods
- Write unit tests for new features

### Testing

```bash
# Run with test configuration
python main.py --config test_config.json --full-scan
```

## 📄 License

MIT License - See LICENSE file for details

## 🆘 Support

For issues, feature requests, or questions:
1. Check the troubleshooting section above
2. Review the logs in `audio_manager.log`
3. Create an issue with detailed error information
4. Include system configuration and audio file details

---

**Built with ❤️ for efficient audio file management on Apple Silicon**
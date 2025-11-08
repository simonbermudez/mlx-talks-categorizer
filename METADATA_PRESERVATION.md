# Audio Metadata Preservation Feature

## Overview

This feature ensures that audio file metadata (especially recording dates and original titles) is preserved throughout the audio processing pipeline. When files are processed, transcribed, and organized, the metadata from the original source files is extracted and copied to the final processed files.

## Key Features

### 1. **Metadata Extraction**
- Extracts comprehensive metadata from original audio files
- Supports MP3, MP4/M4A, and WAV file formats
- Captures:
  - Original title
  - Recording date (TDRC tag for MP3, year for MP4)
  - Artist, album, genre
  - Comments
  - All other available tags

### 2. **Metadata Preservation**
- Copies metadata from original files to processed files
- Preserves recording date/timestamp
- Adds "Original Title" field to the processed file's metadata
- Maintains filesystem timestamps (creation, modification times)

### 3. **Original Title Field**
- Stores the original filename or title tag in a new metadata field
- Accessible in the audio file's comment field:
  - MP3: ID3 COMM tag (Comment field)
  - MP4: `©cmt` tag (Comment field)
  - WAV: ID3 COMM tag (Comment field)
- Format: `"Original Title: [original title or filename]"`

## Implementation Details

### New Class: `AudioMetadataHandler`

Located in [main.py](main.py#L73-L284), this class provides three static methods:

#### 1. `extract_metadata(file_path: str) -> Dict`
Extracts all available metadata from an audio file.

**Returns:**
```python
{
    'original_title': str,      # Original title tag or filename
    'recording_date': str,       # Recording date (TDRC for MP3, ©day for MP4)
    'creation_date': datetime,   # File creation timestamp
    'artist': str,
    'album': str,
    'genre': str,
    'comment': str,
    'year': str,
    'all_tags': dict            # All tags from the file
}
```

#### 2. `copy_metadata(source_file: str, destination_file: str, metadata: Dict = None, add_original_title: bool = True)`
Copies metadata from source to destination file.

**Parameters:**
- `source_file`: Original audio file path
- `destination_file`: Processed audio file path
- `metadata`: Pre-extracted metadata (optional, will extract if not provided)
- `add_original_title`: Whether to add original title to comment field (default: True)

**What it preserves:**
- Recording date/year
- Artist, album, genre
- Original title (stored in comment field)
- All format-specific metadata

#### 3. `preserve_file_timestamps(source_file: str, destination_file: str)`
Preserves filesystem timestamps (access time, modification time).

### Modified Functions

#### `FileOrganizer.organize_file()`
Updated signature: [main.py:1327](main.py#L1327-L1395)

```python
def organize_file(self, source_file: str, speaker_name: str, description: str,
                  transcript: str, original_file: str = None, metadata: Dict = None):
```

**New Parameters:**
- `original_file`: Path to the original unprocessed file
- `metadata`: Pre-extracted metadata dictionary

**Behavior:**
1. Copies the processed audio file to the organized talks directory
2. If `original_file` is provided:
   - Extracts metadata from original file (if not already provided)
   - Copies all metadata to the organized file
   - Preserves filesystem timestamps
   - Logs metadata preservation success

#### `AudioFileManager.process_audio_file()`
Updated at: [main.py:1465-1467](main.py#L1465-L1467)

**New Steps:**
1. **Before processing:** Extracts metadata from the original file
2. **During organization:** Passes original file path and metadata to `organize_file()`
3. **Backup preservation:** Also preserves timestamps on raw talks backup

## Usage

### Automatic Processing
The metadata preservation happens automatically during normal audio file processing. No additional configuration is required.

When you run:
```bash
python main.py
```

The system will:
1. Extract metadata from each source audio file
2. Process the audio (transcription, speaker identification, etc.)
3. Preserve all metadata to the final organized file
4. Add the original title as a comment field

### Testing Metadata Extraction

Use the test script to verify metadata extraction:

```bash
python test_metadata.py /path/to/audio/file.mp3
```

**Example output:**
```
INFO: Successfully imported AudioMetadataHandler
INFO: ============================================================
INFO: Testing metadata extraction on: /path/to/audio/file.mp3
INFO: ============================================================
INFO:
INFO: Extracted Metadata:
INFO:   Original Title: My Original Talk Title
INFO:   Recording Date: 2024-03-15
INFO:   Year: 2024
INFO:   Creation Date: 2024-03-15 10:30:00
INFO:   Artist: Speaker Name
INFO:   Album: Conference 2024
INFO:   Genre: Speech
INFO:   Comment: Original recording notes
INFO:
INFO:   All Tags (12 found):
INFO:     - TIT2
INFO:     - TPE1
INFO:     - TALB
INFO:     - TDRC
INFO:     - TCON
INFO: ============================================================
```

### Viewing Preserved Metadata

After processing, you can view the metadata in the organized files:

**Using mutagen (Python):**
```python
from mutagen import File as MutagenFile

audio = MutagenFile("organized_talks/talks/2025/Speaker Name/Speaker Name - Description.mp3")
print(audio.tags.get('COMM::eng'))  # Output: Original Title: My Original Talk Title
print(audio.tags.get('TDRC'))        # Output: 2024-03-15
```

**Using ffprobe (command line):**
```bash
ffprobe -v quiet -print_format json -show_format \
  "organized_talks/talks/2025/Speaker Name/Speaker Name - Description.mp3"
```

**Using media players:**
Most audio players (iTunes, VLC, etc.) will display the metadata in the file info panel.

## File Processing Flow

```
Original Audio File
    ↓
[1] Extract Metadata
    ├─ Original title
    ├─ Recording date
    └─ All other tags
    ↓
[2] Audio Processing
    ├─ Silence removal (optional)
    ├─ Format conversion (optional)
    └─ Creates processed file
    ↓
[3] Transcription & Speaker ID
    ├─ Whisper transcription
    └─ Speaker identification
    ↓
[4] File Organization
    ├─ Copy processed file to organized_talks/talks/
    ├─ Copy metadata from original → organized file
    ├─ Add "Original Title" comment
    └─ Preserve filesystem timestamps
    ↓
[5] Backup (if preprocessing enabled)
    ├─ Copy original file to raw talks/
    └─ Preserve timestamps on backup
    ↓
Final Result:
- Organized file with preserved metadata
- Original title stored in comment field
- Recording date preserved
- Original file backed up (if preprocessing)
```

## Dependencies

**New Requirement:**
- `mutagen==1.47.0` - Python library for reading and writing audio metadata

**Installation:**
```bash
source venv/bin/activate
pip install mutagen==1.47.0
```

Or install all requirements:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Supported Audio Formats

| Format | Metadata Support | Original Title Storage | Date Preservation |
|--------|------------------|------------------------|-------------------|
| MP3    | ✅ ID3 tags      | COMM::eng (Comment)    | TDRC (Recording date) |
| MP4/M4A| ✅ MP4 atoms     | ©cmt (Comment)         | ©day (Date) |
| WAV    | ✅ ID3 tags      | COMM::eng (Comment)    | TDRC (Recording date) |

## Fallback Behavior

If metadata cannot be read from the original file:
1. **Original Title**: Falls back to the filename (without extension)
2. **Recording Date**: Falls back to file creation timestamp
3. **Other Fields**: Set to `None` (not written to destination)

The system logs warnings but continues processing.

## Logging

Metadata-related log messages:

```
INFO: Extracted metadata from original file: [original title]
INFO: Organized audio: [destination path]
INFO: Copied metadata to [destination path]
INFO: Preserved metadata and timestamps for [destination path]
INFO: Backed up original file to raw talks: [backup path]
```

Debug level:
```
DEBUG: Extracted metadata from [file path]: {metadata dict}
DEBUG: Preserved timestamps from [source] to [destination]
```

Warnings:
```
WARNING: Could not read metadata from [file path]
WARNING: Error extracting metadata from [file path]: [error]
WARNING: Error copying metadata from [source] to [destination]: [error]
```

## Configuration

No additional configuration is required. The feature works automatically with existing settings.

Metadata preservation respects:
- `audio_preprocessing.enable_silence_removal`: Determines whether to create raw talks backup
- `output_base_path`: Base directory for organized files
- `talks_path`: Directory for organized audio files
- `raw_talks_path`: Directory for original file backups

## Example Workflow

1. **Input:** `~/Google Drive/Audio/conference_talk_2024.mp3`
   - Original title: "Introduction to MLX"
   - Recording date: 2024-08-15
   - Artist: "John Doe"

2. **Processing:**
   - Extract metadata ✅
   - Transcribe audio ✅
   - Identify speaker: "John Doe" ✅
   - Generate description: "MLX Framework Introduction" ✅

3. **Output:** `organized_talks/talks/2025/John Doe/John Doe - MLX Framework Introduction.mp3`
   - Title: "John Doe - MLX Framework Introduction" (new filename)
   - Comment: "Original Title: Introduction to MLX" ✅
   - Recording date: 2024-08-15 ✅
   - Artist: "John Doe" ✅
   - Modification time: Same as original file ✅

4. **Backup:** `organized_talks/raw talks/conference_talk_2024.mp3`
   - Exact copy of original with timestamps preserved ✅

## Benefits

1. **Traceability**: Always know the original filename and source
2. **Historical Accuracy**: Preserve when talks were actually recorded
3. **Metadata Integrity**: Don't lose valuable information during processing
4. **Compatibility**: Works with all standard audio players and tools
5. **Flexibility**: Metadata available programmatically and in UI tools

## Technical Notes

- Uses `shutil.copy2()` to preserve filesystem metadata
- Mutagen library handles format-specific metadata writing
- Graceful degradation: Processing continues even if metadata operations fail
- Memory-efficient: Metadata extraction doesn't load audio data
- Thread-safe: All operations use standard file I/O primitives

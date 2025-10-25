# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Audio File Management Specialist project designed to organize, rename, and archive MP3/WAV/MP4 files from multiple sources using AI and MLX (Apple Silicon optimization). The system processes audio files to:

- Aggregate MP3/WAV/MP4 files from multiple configured input directories (supports unlimited sources)
- Filter files that are 10+ minutes in duration
- Transcribe audio files for context understanding
- Generate intelligent titles using OpenAI ChatGPT or simple keyword extraction
- Identify speakers using voice samples
- Organize files into a structured directory system
- Automate regular updates and archiving

## Directory Structure

The project implements a specific folder organization:

```
speakers/
  └── [NAME OF THE SPEAKER].[mp3|wav|mp4]  # Sample voices for speaker training
talks/
  └── [YEAR]/
    └── [SPEAKER NAME]/
      ├── [SPEAKER NAME] - [3 WORD DESCRIPTION].mp3
      └── [SPEAKER NAME] - [3 WORD DESCRIPTION] (Transcript).txt
raw talks/                            # Unprocessed audio files
```

## Key Requirements

- **Audio Processing**: Only process files ≥10 minutes duration
- **File Formats**: Support .mp3, .wav, and .mp4 files
- **MLX Optimization**: Workflow must be optimized for Apple Silicon
- **Speaker Identification**: Use sample voices to train speaker recognition
- **Transcription**: Generate and save transcripts alongside audio files
- **Title Generation**: Use OpenAI ChatGPT for intelligent, context-aware file naming
- **Incremental Processing**: Track last successful run date to process only new files
- **Cleanup**: Monthly deletion of junk files

## Development Context

This is a new project repository with only README.md and LICENSE files present. The codebase will need to be built from scratch to implement the audio processing pipeline described in the requirements.

When developing:
- Prioritize MLX framework usage for Apple Silicon optimization
- Implement audio file filtering by duration before processing
- Design modular components for transcription, speaker identification, and file organization
- Consider batch processing capabilities for efficiency
- Implement robust error handling for file I/O operations
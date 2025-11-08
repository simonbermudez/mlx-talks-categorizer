#!/usr/bin/env python3
"""
Test script to verify metadata extraction and preservation functionality.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import the AudioMetadataHandler class
try:
    from main import AudioMetadataHandler
    logging.info("Successfully imported AudioMetadataHandler")
except ImportError as e:
    logging.error(f"Failed to import AudioMetadataHandler: {e}")
    sys.exit(1)

def test_metadata_extraction(audio_file: str):
    """Test metadata extraction from an audio file."""
    if not Path(audio_file).exists():
        logging.error(f"Test file does not exist: {audio_file}")
        return False

    logging.info(f"\n{'='*60}")
    logging.info(f"Testing metadata extraction on: {audio_file}")
    logging.info(f"{'='*60}")

    try:
        # Extract metadata
        metadata = AudioMetadataHandler.extract_metadata(audio_file)

        # Display results
        logging.info("\nExtracted Metadata:")
        logging.info(f"  Original Title: {metadata.get('original_title', 'N/A')}")
        logging.info(f"  Recording Date: {metadata.get('recording_date', 'N/A')}")
        logging.info(f"  Year: {metadata.get('year', 'N/A')}")
        logging.info(f"  Creation Date: {metadata.get('creation_date', 'N/A')}")
        logging.info(f"  Artist: {metadata.get('artist', 'N/A')}")
        logging.info(f"  Album: {metadata.get('album', 'N/A')}")
        logging.info(f"  Genre: {metadata.get('genre', 'N/A')}")
        logging.info(f"  Comment: {metadata.get('comment', 'N/A')}")

        if metadata.get('all_tags'):
            logging.info(f"\n  All Tags ({len(metadata['all_tags'])} found):")
            for tag_key in list(metadata['all_tags'].keys())[:5]:  # Show first 5 tags
                logging.info(f"    - {tag_key}")

        logging.info(f"\n{'='*60}\n")
        return True

    except Exception as e:
        logging.error(f"Error during metadata extraction test: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file_path>")
        print("\nExample:")
        print(f"  {sys.argv[0]} /path/to/audio/file.mp3")
        print(f"  {sys.argv[0]} organized_talks/raw\\ talks/example.mp3")
        sys.exit(1)

    audio_file = sys.argv[1]
    success = test_metadata_extraction(audio_file)

    sys.exit(0 if success else 1)

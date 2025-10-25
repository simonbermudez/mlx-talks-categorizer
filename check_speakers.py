#!/usr/bin/env python3
"""
Speaker Enrollment Checker
Checks the enrollment status of all speaker samples and provides recommendations.
"""

import os
import json
from pathlib import Path
import librosa

def get_audio_duration(file_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0

def check_speakers():
    """Check all speaker samples and report on enrollment readiness."""
    # Load config
    config_path = "config.json"
    if not os.path.exists(config_path):
        print("Error: config.json not found")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    speakers_path = Path(config.get("speakers_path", "./organized_talks/speakers"))
    supported_formats = config.get("supported_formats", [".mp3", ".wav", ".mp4"])

    if not speakers_path.exists():
        print(f"Error: Speakers directory not found at {speakers_path}")
        return

    print("\n" + "="*80)
    print("SPEAKER ENROLLMENT CHECK")
    print("="*80 + "\n")

    # Group files by speaker name
    speaker_files = {}
    for format_ext in supported_formats:
        pattern = f"*{format_ext}"
        for speaker_file in speakers_path.glob(pattern):
            import re
            speaker_name = speaker_file.stem
            # Remove numeric suffixes for multiple samples
            speaker_name = re.sub(r'_\d+$', '', speaker_name)

            if speaker_name not in speaker_files:
                speaker_files[speaker_name] = []
            speaker_files[speaker_name].append(speaker_file)

    if not speaker_files:
        print("❌ No speaker samples found!")
        print(f"   Add speaker samples to: {speakers_path}")
        return

    # Recommended minimum duration for Eagle enrollment
    MIN_DURATION = 30  # seconds
    RECOMMENDED_DURATION = 60  # seconds for reliable enrollment

    total_speakers = len(speaker_files)
    ready_count = 0

    for speaker_name in sorted(speaker_files.keys()):
        files = speaker_files[speaker_name]
        total_duration = 0

        print(f"\n📊 Speaker: {speaker_name}")
        print(f"   Samples: {len(files)}")

        for file in files:
            duration = get_audio_duration(str(file))
            total_duration += duration
            print(f"   • {file.name}: {duration:.1f}s")

        print(f"   Total Duration: {total_duration:.1f}s")

        if total_duration >= RECOMMENDED_DURATION:
            print(f"   ✅ READY - Excellent ({total_duration:.1f}s)")
            ready_count += 1
        elif total_duration >= MIN_DURATION:
            print(f"   ⚠️  MINIMUM - May work but add more for better accuracy")
            print(f"   Recommend adding {RECOMMENDED_DURATION - total_duration:.1f}s more audio")
            ready_count += 1
        else:
            print(f"   ❌ INSUFFICIENT - Need at least {MIN_DURATION}s total")
            print(f"   Missing: {MIN_DURATION - total_duration:.1f}s")
            print(f"   Recommendation: Add {RECOMMENDED_DURATION - total_duration:.1f}s for best results")

    print("\n" + "="*80)
    print(f"SUMMARY: {ready_count}/{total_speakers} speakers ready")
    print("="*80)

    if ready_count < total_speakers:
        print("\n⚠️  Action Required:")
        print("   Add longer or additional audio samples for speakers marked as INSUFFICIENT")
        print("   Recommendation: 60+ seconds per speaker for best Eagle enrollment")
    else:
        print("\n✅ All speakers have sufficient audio for enrollment!")

    print("\n💡 Tips:")
    print("   • Provide multiple samples per speaker for better accuracy")
    print("   • Name multiple samples: SpeakerName_1.mp3, SpeakerName_2.mp3, etc.")
    print("   • Use clear speech with minimal background noise")
    print("   • Eagle provides feedback during enrollment (check logs)")
    print()

if __name__ == "__main__":
    check_speakers()

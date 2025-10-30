#!/usr/bin/env python3
"""
Script to inspect the speaker embeddings cache file.
"""

import pickle
import numpy as np
from pathlib import Path

def inspect_cache(cache_file="./cache/speaker_embeddings.pkl"):
    """Inspect the contents of the speaker embeddings cache."""
    cache_path = Path(cache_file)

    if not cache_path.exists():
        print(f"❌ Cache file not found: {cache_file}")
        return

    # Get file size
    file_size = cache_path.stat().st_size
    print(f"📁 Cache file size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    print()

    try:
        # Load the cache
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

        print(f"✅ Cache loaded successfully!")
        print(f"📊 Number of speakers in cache: {len(cache)}")
        print()

        # Inspect each speaker's data
        total_embedding_bytes = 0
        for speaker_name, speaker_data in cache.items():
            print(f"👤 Speaker: {speaker_name}")

            if isinstance(speaker_data, dict):
                # New cache format with metadata
                print(f"   ├─ Format: Dictionary (with metadata)")

                if 'embedding' in speaker_data:
                    embedding = speaker_data['embedding']
                    print(f"   ├─ Embedding shape: {embedding.shape}")
                    print(f"   ├─ Embedding dtype: {embedding.dtype}")
                    embedding_bytes = embedding.nbytes
                    print(f"   ├─ Embedding size: {embedding_bytes:,} bytes")
                    total_embedding_bytes += embedding_bytes

                if 'file_hashes' in speaker_data:
                    print(f"   ├─ Number of files: {len(speaker_data['file_hashes'])}")
                    for file_path, file_hash in speaker_data['file_hashes'].items():
                        file_name = Path(file_path).name
                        print(f"   │  └─ {file_name}: {file_hash[:16]}...")

                if 'timestamp' in speaker_data:
                    print(f"   └─ Timestamp: {speaker_data['timestamp']}")
            else:
                # Old cache format - just the embedding
                print(f"   ├─ Format: Raw embedding (no metadata)")
                print(f"   ├─ Embedding shape: {speaker_data.shape}")
                print(f"   ├─ Embedding dtype: {speaker_data.dtype}")
                embedding_bytes = speaker_data.nbytes
                print(f"   └─ Embedding size: {embedding_bytes:,} bytes")
                total_embedding_bytes += embedding_bytes

            print()

        print(f"📦 Total embedding data: {total_embedding_bytes:,} bytes ({total_embedding_bytes/1024:.2f} KB)")
        print(f"📦 Overhead (metadata, pickle): {file_size - total_embedding_bytes:,} bytes ({(file_size - total_embedding_bytes)/1024:.2f} KB)")

        # Calculate expected size
        print()
        print("🧮 Expected sizes:")
        num_speakers = len(cache)
        embedding_dim = 256  # Standard for pyannote embeddings

        # float16 is 2 bytes per element
        expected_per_speaker_float16 = embedding_dim * 2
        expected_total_float16 = num_speakers * expected_per_speaker_float16

        print(f"   ├─ Per speaker (256-dim float16): {expected_per_speaker_float16:,} bytes")
        print(f"   ├─ Total for {num_speakers} speakers: {expected_total_float16:,} bytes ({expected_total_float16/1024:.2f} KB)")
        print(f"   └─ With metadata/overhead: ~{expected_total_float16 * 1.5 / 1024:.2f} KB (estimated)")

        # Check if size is reasonable
        print()
        if file_size < expected_total_float16 * 0.5:
            print("⚠️  WARNING: Cache file seems unusually small!")
            print("   The file might be corrupted or incomplete.")
        elif file_size > expected_total_float16 * 10:
            print("⚠️  WARNING: Cache file seems unusually large!")
            print("   There might be unnecessary data being stored.")
        else:
            print("✅ Cache file size looks reasonable!")

    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_cache()

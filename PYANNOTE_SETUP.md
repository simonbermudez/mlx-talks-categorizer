# Pyannote.audio Setup Guide

## Overview

This guide will help you set up Pyannote.audio for speaker identification in the MLX Talks Categorizer.

**Benefits:**
- ✅ **FREE** - No monthly costs
- ✅ **State-of-the-art accuracy** (~90% DER)
- ✅ **Privacy-first** - All processing on-device
- ✅ **Unlimited usage** - No API quotas

---

## Prerequisites

- Python 3.8+
- Hugging Face account (free)
- Internet connection (for initial model download)

---

## Step 1: Install Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate

# Install pyannote.audio and dependencies
pip install -r requirements.txt
```

This will install:
- `pyannote.audio==3.4.0`
- `pyannote.core==5.0.0`
- And all dependencies (PyTorch, NumPy 2.0+, etc.)

**Note:** We use version 3.4.0 instead of 4.0.1+ because 4.0+ uses torchcodec which has
compatibility issues on Apple Silicon. Version 3.4.0 supports NumPy 2.0 and works reliably
on Apple Silicon using torchaudio.

---

## Step 2: Create Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co/)
2. Click "Sign Up" (top right)
3. Create your free account
4. Verify your email address

---

## Step 3: Get Your Hugging Face Token

1. Log in to Hugging Face
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Click "New token"
4. Name it: `pyannote-mlx-talks`
5. Select role: **Read**
6. Click "Generate token"
7. **Copy the token** (you won't see it again!)

---

## Step 4: Accept Model Terms

Pyannote models require accepting terms of use:

1. Visit: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Click "Agree and access repository"
3. Accept the terms

Also accept terms for the segmentation model:
1. Visit: [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
2. Click "Agree and access repository"
3. Accept the terms

**Important:** You must accept BOTH model terms, or you'll get authentication errors!

---

## Step 5: Configure Your Token

Add your Hugging Face token to `config.json`:

```json
{
  "pyannote": {
    "hf_token": "hf_YOUR_TOKEN_HERE",
    "model": "pyannote/speaker-diarization-3.1",
    "min_speakers": 1,
    "max_speakers": 10
  }
}
```

**Replace `hf_YOUR_TOKEN_HERE` with your actual token from Step 3.**

---

## Step 6: Test the Installation

Run a simple test:

```bash
python -c "from pyannote.audio import Pipeline; print('Pyannote.audio installed successfully!')"
```

You should see: `Pyannote.audio installed successfully!`

---

## Step 7: First Model Download

The first time you run the system, Pyannote will download models (~2GB):

```bash
python main.py --setup
```

This will:
- Download speaker diarization models
- Download segmentation models
- Cache them locally (~/.cache/torch/pyannote/)

**This only happens once!** Subsequent runs use the cached models.

---

## Step 8: Add Speaker Samples

Add speaker audio samples to `organized_talks/speakers/`:

```bash
# Example structure:
organized_talks/speakers/
├── John_Doe.mp3        # 30+ seconds of John speaking
├── Jane_Smith.wav       # 30+ seconds of Jane speaking
└── Bob_Jones.mp4        # 30+ seconds of Bob speaking (video OK)
```

**Requirements:**
- **Duration:** 30+ seconds recommended
- **Quality:** Clear speech, minimal background noise
- **Format:** .mp3, .wav, or .mp4
- **Multiple samples:** You can add John_Doe_1.mp3, John_Doe_2.mp3, etc.

---

## Step 9: Run Your First Processing

```bash
# Process audio files
python main.py --full-scan
```

Check the logs to see:
```
Loading Pyannote model: pyannote/speaker-diarization-3.1
Pyannote pipeline loaded successfully
Creating embedding for speaker: John_Doe (1 sample(s))
Successfully enrolled speaker: John_Doe
Enrolled 3 speaker(s)
```

---

## Troubleshooting

### Error: "401 Client Error: Unauthorized"

**Cause:** Haven't accepted model terms

**Solution:**
1. Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Click "Agree and access repository"
3. Also do the same for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Error: "Token is invalid"

**Cause:** Wrong token or expired

**Solution:**
1. Go to [Hugging Face Tokens](https://huggingface.co/settings/tokens)
2. Generate a new token
3. Update `config.json`

### Error: "Model download failed"

**Cause:** Internet connection issues

**Solution:**
1. Check your internet connection
2. Try again - downloads resume automatically
3. If behind a proxy, set: `export HF_ENDPOINT=https://huggingface.co`

### Warning: "No speakers detected in audio"

**Cause:** Audio quality issues or very short clips

**Solution:**
- Ensure audio has clear speech
- Check file isn't corrupted
- Verify duration is adequate

### Slow First Run

**Normal!** First run downloads ~2GB of models.

Subsequent runs are fast (models are cached).

---

## Advanced Configuration

### Adjust Speaker Count

If you know the number of speakers:

```json
{
  "pyannote": {
    "min_speakers": 1,  // Minimum speakers in audio
    "max_speakers": 3   // Maximum speakers in audio
  }
}
```

Setting tighter bounds improves accuracy!

### Use Different Model

Other available models:

```json
{
  "pyannote": {
    "model": "pyannote/speaker-diarization-2.1"  // Older, faster
  }
}
```

### Enable Debug Logging

See detailed processing logs:

```json
{
  "log_level": "DEBUG"
}
```

---

## Performance Notes

### First Run (One-time)
- Model download: ~2-5 minutes
- Cache size: ~2GB

### Subsequent Runs
- Speaker enrollment: ~5-10 seconds per speaker
- Speaker identification: ~10-20 seconds per audio file

### Hardware
- **CPU:** Works fine, slower processing
- **GPU:** Significantly faster (if PyTorch detects CUDA/MPS)
- **Apple Silicon:** Uses MPS acceleration automatically

---

## Privacy & Security

✅ **All processing is local:**
- Models run on your computer
- Audio never sent to cloud
- HF token only used to download models (one-time)

✅ **Token security:**
- Store token in `config.json` (already in .gitignore)
- Never commit tokens to git
- Use read-only tokens

---

## Cost Comparison

| Service | Monthly Cost (5 hours) |
|---------|------------------------|
| **Pyannote** | **$0** ✅ |
| Deepgram | $0.60 |
| AssemblyAI | $1.35 |
| AWS Transcribe | $7.20 |
| RingCentral | $39+ |
| Picovoice Eagle | Unknown |

**Pyannote is FREE forever!**

---

## Need Help?

1. Check logs: `audio_manager.log`
2. Enable debug mode: `"log_level": "DEBUG"`
3. Review this guide
4. Check [Pyannote documentation](https://github.com/pyannote/pyannote-audio)

---

## Next Steps

✅ **You're all set!**

Your system now uses:
- FREE speaker identification
- State-of-the-art accuracy
- Complete privacy
- Unlimited processing

Start processing your audio files:

```bash
python main.py --full-scan
```

Enjoy! 🎉

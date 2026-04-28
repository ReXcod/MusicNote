# 🎵 Music Analyzer — Deployment Guide

## Why Streamlit Cloud (not Netlify/Vercel)?

| Platform | Your app |
|---|---|
| Netlify / Vercel | Built for JavaScript/static sites. Getting librosa + ffmpeg to run there requires Docker, Lambda hacks, and hours of pain. |
| **Streamlit Cloud** | Built for Python data apps. Runs your code as-is, free, deploys in 3 minutes. |

---

## Files you need (all provided)

```
music_app/
├── app.py            ← the entire web app
├── requirements.txt  ← Python libraries
└── packages.txt      ← tells Streamlit to install ffmpeg (system package)
```

---

## Step-by-Step: Deploy to Streamlit Community Cloud

### Step 1 — Create a GitHub account (if you don't have one)
Go to https://github.com and sign up. Free.

### Step 2 — Create a new GitHub repository
1. Click the **+** icon → **New repository**
2. Name it anything, e.g. `music-analyzer`
3. Set it to **Public**
4. Click **Create repository**

### Step 3 — Upload the 3 files
On your new repo page:
1. Click **Add file** → **Upload files**
2. Drag and drop all 3 files: `app.py`, `requirements.txt`, `packages.txt`
3. Click **Commit changes**

Your repo now looks like:
```
music-analyzer/
├── app.py
├── requirements.txt
└── packages.txt
```

### Step 4 — Sign up for Streamlit Community Cloud
1. Go to https://share.streamlit.io
2. Click **Sign up** — use your GitHub account (one click)

### Step 5 — Deploy your app
1. Click **New app**
2. Select your GitHub repo: `your-username/music-analyzer`
3. Branch: `main`
4. Main file path: `app.py`
5. Click **Deploy!**

Streamlit will:
- Install `ffmpeg` (from packages.txt)
- Install all Python libraries (from requirements.txt)
- Start your app

This takes about **3–5 minutes** the first time.

### Step 6 — Done!
You get a public URL like:
`https://your-username-music-analyzer-app-xxxx.streamlit.app`

Share it with anyone. It works on mobile too.

---

## Running locally (optional, for testing first)

If you have Python installed:

```bash
# 1. Install dependencies
pip install streamlit librosa midiutil numpy matplotlib

# 2. Make sure ffmpeg is installed
#    Mac:     brew install ffmpeg
#    Windows: download from https://ffmpeg.org/download.html
#    Linux:   sudo apt install ffmpeg

# 3. Run the app
streamlit run app.py
```

It opens automatically at http://localhost:8501

---

## What the app does

| Feature | Details |
|---|---|
| **Audio input** | MP3, WAV, MP4, OGG, FLAC, M4A |
| **Pitch detection** | pyin algorithm (same as your notebook) |
| **Key detection** | Chromagram + Krumhansl-Schmuckler profiles |
| **Tempo detection** | librosa beat tracker |
| **Piano Roll** | Color-coded visual score of your melody |
| **Waveform** | With beat markers overlaid |
| **Mel Spectrogram** | Full frequency-time view |
| **Note stats** | Highest, lowest, most common note, duration spread |
| **Full stats** | RMS, spectral centroid, zero-crossing rate, etc. |
| **MIDI download** | One-click .mid file download |
| **Settings** | BPM, velocity, min-duration, pitch range via sidebar |

---

## Troubleshooting

**"ffmpeg not found" error on local machine**
→ Install ffmpeg separately (see above). It is NOT a Python package.

**"No notes detected"**
→ Lower the "Min note duration" slider in the sidebar to 0.03 or less.
→ Check that FMIN/FMAX range covers your instrument's pitch range.

**App is slow on first run**
→ Normal. librosa loads the audio model the first time. Subsequent runs are faster.

**Streamlit Cloud free tier limits**
→ 1 GB RAM, sleeps after inactivity (wakes in ~30 sec).
→ For heavy use, upgrade to Streamlit Teams ($) or self-host on Railway/Render.

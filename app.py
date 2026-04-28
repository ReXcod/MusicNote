import streamlit as st
import librosa
import librosa.display
import numpy as np
from midiutil import MIDIFile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tempfile
import os
import subprocess
import io

# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Analyzer",
    page_icon="🎵",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0e0e1a; }
    h1, h2, h3 { color: #e8d5ff; }
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
    .metric-box {
        background: #1a1a2e; border: 1px solid #2d2d4e;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎵 Audio Analyzer & MIDI Converter")
st.caption("Upload any audio → get MIDI, full music analysis, and a piano roll score")

# ─────────────────────────────────────────────────────────────
#  Tuneable parameters (sidebar)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    BPM          = st.slider("Output MIDI BPM", 60, 200, 120, step=5)
    MIDI_VOLUME  = st.slider("MIDI Velocity (volume)", 50, 127, 100)
    MIN_DURATION = st.slider("Min note duration (sec)", 0.02, 0.20, 0.05, step=0.01)
    FMIN_NOTE    = st.selectbox("Lowest pitch to detect", ["C1","C2","C3"], index=1)
    FMAX_NOTE    = st.selectbox("Highest pitch to detect", ["C6","C7","C8"], index=1)
    st.markdown("---")
    st.markdown("**Tips**\n- Higher min-duration = fewer blip notes\n- Adjust BPM to match your song")

# ─────────────────────────────────────────────────────────────
#  File upload
# ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your audio file",
    type=["mp3", "wav", "mp4", "ogg", "flac", "m4a"],
    help="Supported: MP3, WAV, MP4, OGG, FLAC, M4A"
)

# ─────────────────────────────────────────────────────────────
#  Helper: Hz → MIDI note number
# ─────────────────────────────────────────────────────────────
def hz_to_midi(freq):
    return int(round(librosa.hz_to_midi(freq)))

# ─────────────────────────────────────────────────────────────
#  Helper: MIDI note number → name (e.g. 60 → "C4")
# ─────────────────────────────────────────────────────────────
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def midi_to_name(n):
    return f"{NOTE_NAMES[n % 12]}{(n // 12) - 1}"

# ─────────────────────────────────────────────────────────────
#  Helper: key detection via chromagram + key profiles
# ─────────────────────────────────────────────────────────────
MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    best_score, best_key = -np.inf, "Unknown"
    for i, note in enumerate(NOTE_NAMES):
        maj = np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_mean)[0, 1]
        min_ = np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_mean)[0, 1]
        if maj > best_score:
            best_score, best_key = maj, f"{note} Major"
        if min_ > best_score:
            best_score, best_key = min_, f"{note} Minor"
    return best_key

# ─────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────
if uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save upload
        raw_path = os.path.join(tmpdir, uploaded.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded.read())

        # 2. Convert to mono WAV with ffmpeg
        wav_path = os.path.join(tmpdir, "input.wav")
        with st.spinner("🔄 Converting audio…"):
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path, "-ac", "1", "-ar", "22050", wav_path],
                capture_output=True, text=True
            )
        if result.returncode != 0:
            st.error("ffmpeg conversion failed. Make sure the file is a valid audio/video file.")
            st.code(result.stderr[-600:])
            st.stop()

        # 3. Load audio
        with st.spinner("📊 Loading & analysing audio…"):
            y, sr = librosa.load(wav_path, sr=22050, mono=True)

        # ── Analysis ──────────────────────────────────────────
        duration     = librosa.get_duration(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val    = float(np.atleast_1d(tempo)[0])
        key          = detect_key(y, sr)
        rms          = float(np.sqrt(np.mean(y**2)))
        zcr          = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

        # ── Pitch detection (pyin) ────────────────────────────
        with st.spinner("🎼 Detecting pitches…"):
            fmin = librosa.note_to_hz(FMIN_NOTE)
            fmax = librosa.note_to_hz(FMAX_NOTE)
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048
            )

        hop_length = 512
        frame_sec  = hop_length / sr

        # ── Group frames into notes ───────────────────────────
        notes = []   # (start_sec, duration_sec, midi_note)
        i = 0
        while i < len(f0):
            if voiced_flag[i] and not np.isnan(f0[i]):
                note    = hz_to_midi(f0[i])
                start_i = i
                while (i < len(f0) and voiced_flag[i]
                       and not np.isnan(f0[i])
                       and abs(hz_to_midi(f0[i]) - note) <= 1):
                    i += 1
                dur = (i - start_i) * frame_sec
                if dur >= MIN_DURATION:
                    notes.append((start_i * frame_sec, dur, note))
            else:
                i += 1

        # ── Build MIDI in memory ──────────────────────────────
        midi_obj = MIDIFile(1)
        midi_obj.addTempo(0, 0, BPM)
        beats_per_sec = BPM / 60.0
        for start_sec, dur_sec, pitch in notes:
            sb = start_sec * beats_per_sec
            db = dur_sec   * beats_per_sec
            p  = max(0, min(127, pitch))
            midi_obj.addNote(0, 0, p, sb, db, MIDI_VOLUME)

        midi_buf = io.BytesIO()
        midi_obj.writeFile(midi_buf)
        midi_bytes = midi_buf.getvalue()

        # ════════════════════════════════════════════════════════
        #  DISPLAY
        # ════════════════════════════════════════════════════════

        st.success(f"✅ Detected **{len(notes)} notes** in {duration:.1f}s of audio")

        # ── Row 1: key metrics ────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🎵 Key",          key)
        c2.metric("🥁 Tempo",        f"{tempo_val:.1f} BPM")
        c3.metric("⏱ Duration",      f"{duration:.1f}s")
        c4.metric("🎹 Notes found",  len(notes))
        c5.metric("🔊 RMS Energy",   f"{rms:.4f}")

        st.markdown("---")

        # ── Tabs for charts ───────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🎹 Piano Roll (Score)", "📈 Waveform & Spectrum",
             "🎼 Note Analysis", "ℹ️ Full Stats"]
        )

        # ── TAB 1: Piano Roll ─────────────────────────────────
        with tab1:
            st.subheader("Piano Roll — your melody as a visual score")
            if notes:
                pitches = [n[2] for n in notes]
                pmin    = max(0,   min(pitches) - 3)
                pmax    = min(127, max(pitches) + 3)

                fig, ax = plt.subplots(figsize=(14, 5))
                fig.patch.set_facecolor("#0e0e1a")
                ax.set_facecolor("#13132a")

                # Draw piano-key background stripes
                black_keys = {1, 3, 6, 8, 10}  # semitone offsets
                for p in range(pmin, pmax + 1):
                    color = "#1a1a30" if (p % 12) in black_keys else "#16162a"
                    ax.axhspan(p - 0.5, p + 0.5, color=color, zorder=0)

                # Draw notes
                cmap = plt.cm.plasma
                total_dur = notes[-1][0] + notes[-1][1]
                for start_s, dur_s, pitch in notes:
                    norm = (pitch - pmin) / max(1, pmax - pmin)
                    color = cmap(norm)
                    rect = mpatches.FancyBboxPatch(
                        (start_s, pitch - 0.45), dur_s, 0.9,
                        boxstyle="round,pad=0.02",
                        facecolor=color, edgecolor="none", alpha=0.9
                    )
                    ax.add_patch(rect)

                # Y axis: note names
                ytick_range = range(pmin, pmax + 1, 2)
                ax.set_yticks(list(ytick_range))
                ax.set_yticklabels([midi_to_name(p) for p in ytick_range],
                                   color="#ccccdd", fontsize=8)
                ax.set_xlim(0, total_dur)
                ax.set_ylim(pmin - 0.5, pmax + 0.5)
                ax.set_xlabel("Time (seconds)", color="#aaaacc")
                ax.set_title("Piano Roll", color="#e8d5ff", fontsize=13)
                ax.tick_params(colors="#aaaacc")
                for spine in ax.spines.values():
                    spine.set_color("#2d2d4e")

                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No notes detected. Try lowering the Min note duration in the sidebar.")

        # ── TAB 2: Waveform + Spectrogram ────────────────────
        with tab2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
            fig.patch.set_facecolor("#0e0e1a")
            for ax in (ax1, ax2):
                ax.set_facecolor("#13132a")
                for spine in ax.spines.values():
                    spine.set_color("#2d2d4e")
                ax.tick_params(colors="#aaaacc")

            # Waveform
            times = np.linspace(0, duration, len(y))
            ax1.plot(times, y, color="#7c3aed", linewidth=0.4, alpha=0.85)
            ax1.set_ylabel("Amplitude", color="#aaaacc")
            ax1.set_title("Waveform", color="#e8d5ff")

            # Mark beat positions
            beat_times = librosa.frames_to_time(beats, sr=sr)
            for bt in beat_times:
                ax1.axvline(bt, color="#f59e0b", alpha=0.3, linewidth=0.8)

            # Mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_db, sr=sr, x_axis="time",
                                           y_axis="mel", ax=ax2,
                                           cmap="magma")
            ax2.set_title("Mel Spectrogram", color="#e8d5ff")
            ax2.set_ylabel("Frequency (Hz)", color="#aaaacc")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── TAB 3: Note Analysis ──────────────────────────────
        with tab3:
            if notes:
                pitches   = [n[2] for n in notes]
                durations = [n[1] for n in notes]

                col_a, col_b = st.columns(2)

                # Pitch histogram
                with col_a:
                    st.markdown("**Pitch distribution**")
                    unique_pitches, counts = np.unique(pitches, return_counts=True)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    fig.patch.set_facecolor("#0e0e1a")
                    ax.set_facecolor("#13132a")
                    bars = ax.bar(
                        [midi_to_name(p) for p in unique_pitches], counts,
                        color="#7c3aed", edgecolor="#4f46e5", linewidth=0.5
                    )
                    plt.xticks(rotation=60, ha="right", color="#aaaacc", fontsize=8)
                    ax.set_ylabel("Count", color="#aaaacc")
                    ax.set_title("Note Frequency", color="#e8d5ff")
                    ax.tick_params(colors="#aaaacc")
                    for spine in ax.spines.values():
                        spine.set_color("#2d2d4e")
                    st.pyplot(fig)
                    plt.close(fig)

                # Duration distribution
                with col_b:
                    st.markdown("**Note duration distribution**")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    fig.patch.set_facecolor("#0e0e1a")
                    ax.set_facecolor("#13132a")
                    ax.hist(durations, bins=20, color="#4f46e5",
                            edgecolor="#7c3aed", linewidth=0.5)
                    ax.set_xlabel("Duration (sec)", color="#aaaacc")
                    ax.set_ylabel("Count", color="#aaaacc")
                    ax.set_title("Note Duration Spread", color="#e8d5ff")
                    ax.tick_params(colors="#aaaacc")
                    for spine in ax.spines.values():
                        spine.set_color("#2d2d4e")
                    st.pyplot(fig)
                    plt.close(fig)

                # Summary table
                most_common_midi = unique_pitches[np.argmax(counts)]
                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Highest Note",     midi_to_name(max(pitches)))
                m2.metric("Lowest Note",      midi_to_name(min(pitches)))
                m3.metric("Most Common Note", midi_to_name(most_common_midi))
                m4.metric("Avg Duration",     f"{np.mean(durations):.2f}s")
            else:
                st.info("No notes detected.")

        # ── TAB 4: Full Stats ─────────────────────────────────
        with tab4:
            st.markdown("### Audio Statistics")
            stats = {
                "Estimated Key":            key,
                "Estimated Tempo (BPM)":    f"{tempo_val:.2f}",
                "Duration (sec)":           f"{duration:.2f}",
                "Sample Rate (Hz)":         sr,
                "Total Samples":            len(y),
                "Notes Detected":           len(notes),
                "RMS Energy":               f"{rms:.6f}",
                "Zero-Crossing Rate":       f"{zcr:.6f}",
                "Spectral Centroid (Hz)":   f"{spectral_centroid:.1f}",
                "Spectral Rolloff (Hz)":    f"{spectral_rolloff:.1f}",
                "Output MIDI BPM":          BPM,
                "Min Note Duration (s)":    MIN_DURATION,
            }
            for k, v in stats.items():
                st.markdown(f"**{k}:** `{v}`")

        st.markdown("---")

        # ── Download MIDI ─────────────────────────────────────
        st.subheader("⬇️ Download")
        st.download_button(
            label="⬇️ Download MIDI File",
            data=midi_bytes,
            file_name="output.mid",
            mime="audio/midi",
            use_container_width=True,
        )
        st.caption("Open the .mid file in GarageBand, MuseScore, FL Studio, Ableton, etc.")

else:
    st.info("👆 Upload an audio file above to get started.")
    st.markdown("""
    **What this app does:**
    - 🎵 Detects every pitch in your recording using the `pyin` algorithm
    - 🎹 Generates a downloadable MIDI file
    - 📈 Shows waveform, mel spectrogram, and beat markers
    - 🎼 Renders a piano roll (visual score) of your melody
    - 📊 Estimates musical key, tempo, and note statistics
    """)

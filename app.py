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
    .sheet-music-container svg { background: white; border-radius: 8px; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("🎵 Audio Analyzer & MIDI Converter")
st.caption("Upload any audio → classical sheet music notation, MIDI, full analysis")

# ─────────────────────────────────────────────────────────────
#  Sidebar settings
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    BPM             = st.slider("Output MIDI BPM", 60, 200, 120, step=5)
    MIDI_VOLUME     = st.slider("MIDI Velocity (volume)", 50, 127, 100)
    MIN_DURATION    = st.slider("Min note duration (sec)", 0.02, 0.20, 0.05, step=0.01)
    FMIN_NOTE       = st.selectbox("Lowest pitch to detect", ["C1","C2","C3"], index=1)
    FMAX_NOTE       = st.selectbox("Highest pitch to detect", ["C6","C7","C8"], index=1)
    MAX_SHEET_NOTES = st.slider("Max notes in sheet music", 50, 300, 150, step=25,
                                help="Fewer = faster rendering")
    st.markdown("---")
    st.markdown("**Tips**\n- Higher min-duration = fewer blip notes\n- Adjust BPM to match your song")

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def hz_to_midi(freq):
    return int(round(librosa.hz_to_midi(freq)))

def midi_to_name(n):
    return f"{NOTE_NAMES[n % 12]}{(n // 12) - 1}"

MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    best_score, best_key = -np.inf, "Unknown"
    for i, name in enumerate(NOTE_NAMES):
        maj  = np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_mean)[0, 1]
        min_ = np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_mean)[0, 1]
        if maj  > best_score: best_score, best_key = maj,  f"{name} Major"
        if min_ > best_score: best_score, best_key = min_, f"{name} Minor"
    return best_key

# ─────────────────────────────────────────────────────────────
#  Duration quantization
# ─────────────────────────────────────────────────────────────
STANDARD_DURATIONS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125, 0.0625]

def quantize(beats):
    return min(STANDARD_DURATIONS, key=lambda x: abs(x - beats))

# ─────────────────────────────────────────────────────────────
#  Build MusicXML from detected notes
# ─────────────────────────────────────────────────────────────
def build_musicxml(notes_list, bpm, key_str, max_notes):
    # music21 v9+: Rest lives in note module, not a separate rest module
    from music21 import stream, note, meter, tempo, key, clef, metadata
    m21note  = note
    m21clef  = clef
    m21tempo = tempo
    m21key   = key

    notes_list = notes_list[:max_notes]
    s = stream.Score()
    s.metadata = metadata.Metadata()
    s.metadata.title = f"MIDI Transcription — {key_str}"

    part = stream.Part()
    avg_pitch = float(np.mean([n[2] for n in notes_list]))
    part.append(m21clef.BassClef() if avg_pitch < 60 else m21clef.TrebleClef())

    try:
        tonic, mode = key_str.split()
        part.append(m21key.Key(tonic, mode.lower()))
    except Exception:
        pass

    part.append(meter.TimeSignature('4/4'))
    part.append(m21tempo.MetronomeMark(number=bpm))

    beats_per_sec = bpm / 60.0
    prev_end_beat = 0.0

    for start_sec, dur_sec, midi_pitch in notes_list:
        start_beat = start_sec * beats_per_sec
        dur_beats  = dur_sec   * beats_per_sec

        gap = start_beat - prev_end_beat
        if gap > 0.12:
            q_gap = quantize(gap)
            if q_gap >= 0.0625:
                part.append(note.Rest(quarterLength=q_gap))

        q_dur = quantize(max(0.125, dur_beats))
        n = m21note.Note()
        n.pitch.midi = max(0, min(127, midi_pitch))
        n.duration.quarterLength = q_dur
        part.append(n)
        prev_end_beat = start_beat + dur_beats

    s.append(part)

    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tf:
        tmp_path = tf.name
    s.write('musicxml', fp=tmp_path)
    with open(tmp_path, 'r', encoding='utf-8') as f:
        xml_str = f.read()
    os.unlink(tmp_path)
    return xml_str

# ─────────────────────────────────────────────────────────────
#  Render MusicXML → SVG pages via verovio
# ─────────────────────────────────────────────────────────────
def render_sheet_music(xml_str):
    import verovio, cairosvg
    tk = verovio.toolkit()
    tk.setOptions({
        'pageWidth':        2100,
        'pageHeight':       2970,
        'scale':            45,
        'adjustPageHeight': True,
        'footer':           'none',
        'header':           'auto',
    })
    tk.loadData(xml_str)
    page_count = tk.getPageCount()
    png_pages = []
    for p in range(1, page_count + 1):
        svg_str   = tk.renderToSVG(p)
        png_bytes = cairosvg.svg2png(
            bytestring=svg_str.encode('utf-8'),
            background_color='white',
            dpi=150,
        )
        png_pages.append(png_bytes)
    return png_pages

# ─────────────────────────────────────────────────────────────
#  File upload
# ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your audio file",
    type=["mp3", "wav", "mp4", "ogg", "flac", "m4a"],
    help="Supported: MP3, WAV, MP4, OGG, FLAC, M4A"
)

# ─────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────
if uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:

        # 1 ── Save upload
        raw_path = os.path.join(tmpdir, uploaded.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded.read())

        # 2 ── ffmpeg → mono WAV
        wav_path = os.path.join(tmpdir, "input.wav")
        with st.spinner("🔄 Converting audio…"):
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path, "-ac", "1", "-ar", "22050", wav_path],
                capture_output=True, text=True
            )
        if result.returncode != 0:
            st.error("ffmpeg conversion failed.")
            st.code(result.stderr[-600:])
            st.stop()

        # 3 ── Load + analyse
        with st.spinner("📊 Analysing audio…"):
            y, sr = librosa.load(wav_path, sr=22050, mono=True)

        duration        = librosa.get_duration(y=y, sr=sr)
        tempo_arr, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val       = float(np.atleast_1d(tempo_arr)[0])
        key_str         = detect_key(y, sr)
        rms             = float(np.sqrt(np.mean(y**2)))
        zcr             = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        spec_centroid   = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_rolloff    = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

        # 4 ── pyin pitch detection
        with st.spinner("🎼 Detecting pitches…"):
            fmin = librosa.note_to_hz(FMIN_NOTE)
            fmax = librosa.note_to_hz(FMAX_NOTE)
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048
            )

        hop_length = 512
        frame_sec  = hop_length / sr

        # 5 ── Group frames → notes
        notes = []
        i = 0
        while i < len(f0):
            if voiced_flag[i] and not np.isnan(f0[i]):
                note_midi = hz_to_midi(f0[i])
                start_i   = i
                while (i < len(f0) and voiced_flag[i]
                       and not np.isnan(f0[i])
                       and abs(hz_to_midi(f0[i]) - note_midi) <= 1):
                    i += 1
                dur = (i - start_i) * frame_sec
                if dur >= MIN_DURATION:
                    notes.append((start_i * frame_sec, dur, note_midi))
            else:
                i += 1

        # 6 ── Build MIDI bytes
        midi_obj = MIDIFile(1)
        midi_obj.addTempo(0, 0, BPM)
        bps = BPM / 60.0
        for start_sec, dur_sec, pitch in notes:
            midi_obj.addNote(0, 0, max(0, min(127, pitch)),
                             start_sec * bps, dur_sec * bps, MIDI_VOLUME)
        midi_buf = io.BytesIO()
        midi_obj.writeFile(midi_buf)
        midi_bytes = midi_buf.getvalue()

        # 7 ── Build sheet music
        xml_str   = None
        svg_pages = []
        sheet_ok  = False
        sheet_err = ""
        with st.spinner("🎼 Engraving sheet music (this takes ~10 sec)…"):
            try:
                xml_str   = build_musicxml(notes, BPM, key_str, MAX_SHEET_NOTES)
                svg_pages = render_sheet_music(xml_str)
                sheet_ok  = True
            except Exception as e:
                sheet_err = str(e)

        # ════════════════════════════════════════════════════════
        #  DISPLAY
        # ════════════════════════════════════════════════════════

        st.success(f"✅ Detected **{len(notes)} notes** in {duration:.1f}s of audio")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🎵 Key",         key_str)
        c2.metric("🥁 Tempo",       f"{tempo_val:.1f} BPM")
        c3.metric("⏱ Duration",     f"{duration:.1f}s")
        c4.metric("🎹 Notes found", len(notes))
        c5.metric("🔊 RMS Energy",  f"{rms:.4f}")

        st.markdown("---")

        tab_sheet, tab_roll, tab_wave, tab_notes, tab_stats = st.tabs([
            "🎼 Sheet Music",
            "🎹 Piano Roll",
            "📈 Waveform & Spectrum",
            "📊 Note Analysis",
            "ℹ️ Full Stats",
        ])

        # ── Sheet Music ────────────────────────────────────────
        with tab_sheet:
            st.subheader(f"Classical Notation — {key_str}")
            st.caption(
                f"Showing first {min(len(notes), MAX_SHEET_NOTES)} of {len(notes)} notes  ·  "
                "increase 'Max notes in sheet music' in sidebar for more pages"
            )
            if not sheet_ok:
                st.error(f"Sheet music rendering failed: {sheet_err}")
                st.info("Make sure music21 and verovio are in requirements.txt")
            else:
                for page_idx, png in enumerate(svg_pages, 1):
                    if len(svg_pages) > 1:
                        st.markdown(f"**Page {page_idx} / {len(svg_pages)}**")
                    st.image(png, use_container_width=True)

                # Per-page PNG downloads
                st.markdown("**Download sheet music:**")
                n_dl_cols = min(len(svg_pages), 4)
                dl_cols   = st.columns(n_dl_cols)
                for page_idx, png in enumerate(svg_pages, 1):
                    col = dl_cols[(page_idx - 1) % n_dl_cols]
                    col.download_button(
                        label=f"⬇️ PNG Page {page_idx}",
                        data=png,
                        file_name=f"sheet_music_p{page_idx}.png",
                        mime="image/png",
                    )
                st.caption("💡 Open in any image viewer · Print · Share directly")

        # ── Piano Roll ─────────────────────────────────────────
        with tab_roll:
            st.subheader("Piano Roll — time vs. pitch")
            if notes:
                pitches = [n[2] for n in notes]
                pmin    = max(0,   min(pitches) - 3)
                pmax    = min(127, max(pitches) + 3)

                fig, ax = plt.subplots(figsize=(14, 5))
                fig.patch.set_facecolor("#0e0e1a")
                ax.set_facecolor("#13132a")
                black_keys = {1, 3, 6, 8, 10}
                for p in range(pmin, pmax + 1):
                    ax.axhspan(p - 0.5, p + 0.5,
                               color="#1a1a30" if (p % 12) in black_keys else "#16162a",
                               zorder=0)
                cmap = plt.cm.plasma
                for start_s, dur_s, pitch in notes:
                    norm = (pitch - pmin) / max(1, pmax - pmin)
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (start_s, pitch - 0.45), dur_s, 0.9,
                        boxstyle="round,pad=0.02",
                        facecolor=cmap(norm), edgecolor="none", alpha=0.9
                    ))
                ytick_range = range(pmin, pmax + 1, 2)
                ax.set_yticks(list(ytick_range))
                ax.set_yticklabels([midi_to_name(p) for p in ytick_range],
                                   color="#ccccdd", fontsize=8)
                total_dur = notes[-1][0] + notes[-1][1]
                ax.set_xlim(0, total_dur)
                ax.set_ylim(pmin - 0.5, pmax + 0.5)
                ax.set_xlabel("Time (seconds)", color="#aaaacc")
                ax.set_title("Piano Roll", color="#e8d5ff", fontsize=13)
                ax.tick_params(colors="#aaaacc")
                for spine in ax.spines.values(): spine.set_color("#2d2d4e")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No notes detected.")

        # ── Waveform + Spectrogram ─────────────────────────────
        with tab_wave:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
            fig.patch.set_facecolor("#0e0e1a")
            for ax in (ax1, ax2):
                ax.set_facecolor("#13132a")
                for spine in ax.spines.values(): spine.set_color("#2d2d4e")
                ax.tick_params(colors="#aaaacc")

            times = np.linspace(0, duration, len(y))
            ax1.plot(times, y, color="#7c3aed", linewidth=0.4, alpha=0.85)
            ax1.set_ylabel("Amplitude", color="#aaaacc")
            ax1.set_title("Waveform  (yellow = detected beats)", color="#e8d5ff")
            for bt in librosa.frames_to_time(beats, sr=sr):
                ax1.axvline(bt, color="#f59e0b", alpha=0.3, linewidth=0.8)

            S_db = librosa.power_to_db(
                librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis="time",
                                     y_axis="mel", ax=ax2, cmap="magma")
            ax2.set_title("Mel Spectrogram", color="#e8d5ff")
            ax2.set_ylabel("Frequency (Hz)", color="#aaaacc")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── Note Analysis ──────────────────────────────────────
        with tab_notes:
            if notes:
                pitches   = [n[2] for n in notes]
                durations = [n[1] for n in notes]
                unique_pitches, counts = np.unique(pitches, return_counts=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Pitch distribution**")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    fig.patch.set_facecolor("#0e0e1a"); ax.set_facecolor("#13132a")
                    ax.bar([midi_to_name(p) for p in unique_pitches], counts,
                           color="#7c3aed", edgecolor="#4f46e5", linewidth=0.5)
                    plt.xticks(rotation=60, ha="right", color="#aaaacc", fontsize=8)
                    ax.set_ylabel("Count", color="#aaaacc")
                    ax.set_title("Note Frequency", color="#e8d5ff")
                    ax.tick_params(colors="#aaaacc")
                    for spine in ax.spines.values(): spine.set_color("#2d2d4e")
                    st.pyplot(fig); plt.close(fig)

                with col_b:
                    st.markdown("**Note duration spread**")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    fig.patch.set_facecolor("#0e0e1a"); ax.set_facecolor("#13132a")
                    ax.hist(durations, bins=20, color="#4f46e5",
                            edgecolor="#7c3aed", linewidth=0.5)
                    ax.set_xlabel("Duration (sec)", color="#aaaacc")
                    ax.set_ylabel("Count", color="#aaaacc")
                    ax.set_title("Duration Distribution", color="#e8d5ff")
                    ax.tick_params(colors="#aaaacc")
                    for spine in ax.spines.values(): spine.set_color("#2d2d4e")
                    st.pyplot(fig); plt.close(fig)

                most_common = unique_pitches[np.argmax(counts)]
                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Highest Note",     midi_to_name(max(pitches)))
                m2.metric("Lowest Note",      midi_to_name(min(pitches)))
                m3.metric("Most Common Note", midi_to_name(most_common))
                m4.metric("Avg Duration",     f"{np.mean(durations):.2f}s")

        # ── Full Stats ─────────────────────────────────────────
        with tab_stats:
            st.markdown("### Audio Statistics")
            for k, v in {
                "Estimated Key":           key_str,
                "Estimated Tempo (BPM)":   f"{tempo_val:.2f}",
                "Duration (sec)":          f"{duration:.2f}",
                "Sample Rate (Hz)":        sr,
                "Total Samples":           len(y),
                "Notes Detected":          len(notes),
                "Notes in Sheet Music":    min(len(notes), MAX_SHEET_NOTES),
                "RMS Energy":              f"{rms:.6f}",
                "Zero-Crossing Rate":      f"{zcr:.6f}",
                "Spectral Centroid (Hz)":  f"{spec_centroid:.1f}",
                "Spectral Rolloff (Hz)":   f"{spec_rolloff:.1f}",
                "Output MIDI BPM":         BPM,
                "Min Note Duration (s)":   MIN_DURATION,
            }.items():
                st.markdown(f"**{k}:** `{v}`")

        # ── Downloads row ──────────────────────────────────────
        st.markdown("---")
        st.subheader("⬇️ Downloads")
        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            st.download_button(
                "⬇️ MIDI File (.mid)",
                data=midi_bytes,
                file_name="transcription.mid",
                mime="audio/midi",
                use_container_width=True,
            )
            st.caption("GarageBand · FL Studio · Ableton · MuseScore")

        with dl2:
            if sheet_ok and svg_pages:
                st.download_button(
                    "⬇️ Sheet Music PNG (Page 1)",
                    data=svg_pages[0],
                    file_name="sheet_music_p1.png",
                    mime="image/png",
                    use_container_width=True,
                )
                st.caption("High-res PNG · Open in any image viewer · Print directly")

        with dl3:
            if sheet_ok and svg_pages and len(svg_pages) > 1:
                for i, png in enumerate(svg_pages[1:], 2):
                    st.download_button(
                        f"⬇️ Sheet Music PNG (Page {i})",
                        data=png,
                        file_name=f"sheet_music_p{i}.png",
                        mime="image/png",
                        use_container_width=True,
                    )

else:
    st.info("👆 Upload an audio file above to get started.")
    st.markdown("""
    **What this app produces:**
    - 🎼 **Classical sheet music** — real staff notation with clef, key signature, time signature
    - 🎹 **MIDI file** — play in any DAW or notation software
    - 🗒️ **MusicXML** — fully editable score in MuseScore (free), Sibelius, or Finale
    - 📈 Waveform with beat markers + Mel Spectrogram
    - 📊 Key, tempo, pitch histogram, note duration distribution, audio stats
    """)

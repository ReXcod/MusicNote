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
import base64

st.set_page_config(page_title="🎵 Music Analyzer", page_icon="🎵", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e0e1a; }
    h1, h2, h3 { color: #e8d5ff; }
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎵 Audio Analyzer & MIDI Converter")
st.caption("Upload audio → get classical sheet music, MIDI, waveform, and full analysis")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    BPM             = st.slider("Output MIDI BPM", 60, 200, 120, step=5)
    MIDI_VOLUME     = st.slider("MIDI Velocity", 50, 127, 100)
    MIN_DURATION    = st.slider("Min note duration (sec)", 0.02, 0.20, 0.05, step=0.01)
    FMIN_NOTE       = st.selectbox("Lowest pitch", ["C1","C2","C3"], index=1)
    FMAX_NOTE       = st.selectbox("Highest pitch", ["C6","C7","C8"], index=1)
    MAX_SHEET_NOTES = st.slider("Max notes in sheet music", 50, 300, 150, step=25)
    st.markdown("---")
    st.markdown("**Tips**\n- Higher min-duration = fewer blip notes\n- Adjust BPM to match your song")

# ── Helpers ───────────────────────────────────────────────────
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def hz_to_midi(freq):
    return int(round(librosa.hz_to_midi(freq)))

def midi_to_name(n):
    return f"{NOTE_NAMES[n % 12]}{(n // 12) - 1}"

MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def detect_key(y, sr):
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    best_score, best_key = -np.inf, "Unknown"
    for i, name in enumerate(NOTE_NAMES):
        maj  = np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_mean)[0, 1]
        min_ = np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_mean)[0, 1]
        if maj  > best_score: best_score, best_key = maj,  f"{name} Major"
        if min_ > best_score: best_score, best_key = min_, f"{name} Minor"
    return best_key

STANDARD_DURATIONS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125, 0.0625]

def quantize(beats):
    return min(STANDARD_DURATIONS, key=lambda x: abs(x - beats))

# ── Build MusicXML ────────────────────────────────────────────
def build_musicxml(notes_list, bpm, key_str, max_notes):
    from music21 import stream, note, meter, tempo, key, clef, metadata

    notes_list = notes_list[:max_notes]
    s          = stream.Score()
    s.metadata = metadata.Metadata()
    s.metadata.title = f"MIDI Transcription — {key_str}"

    part      = stream.Part()
    avg_pitch = float(np.mean([n[2] for n in notes_list]))
    part.append(clef.BassClef() if avg_pitch < 60 else clef.TrebleClef())

    try:
        tonic, mode = key_str.split()
        part.append(key.Key(tonic, mode.lower()))
    except Exception:
        pass

    part.append(meter.TimeSignature('4/4'))
    part.append(tempo.MetronomeMark(number=bpm))

    beats_per_sec = bpm / 60.0
    prev_end      = 0.0

    for start_sec, dur_sec, midi_pitch in notes_list:
        start_beat = start_sec * beats_per_sec
        dur_beats  = dur_sec   * beats_per_sec

        gap = start_beat - prev_end
        if gap > 0.12:
            q_gap = quantize(gap)
            if q_gap >= 0.0625:
                part.append(note.Rest(quarterLength=q_gap))

        q_dur = quantize(max(0.125, dur_beats))
        n = note.Note()
        n.pitch.midi = max(0, min(127, midi_pitch))
        n.duration.quarterLength = q_dur
        part.append(n)
        prev_end = start_beat + dur_beats

    s.append(part)

    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tf:
        tmp = tf.name
    s.write('musicxml', fp=tmp)
    with open(tmp, 'r', encoding='utf-8') as f:
        xml = f.read()
    os.unlink(tmp)
    return xml

# ── Render SVG pages via verovio ──────────────────────────────
def render_svg_pages(xml_str):
    import verovio
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
    return [tk.renderToSVG(p) for p in range(1, tk.getPageCount() + 1)]

# ── SVG → PNG bytes (best-effort, falls back gracefully) ──────
def svg_to_png(svg_str):
    try:
        import cairosvg
        return cairosvg.svg2png(
            bytestring=svg_str.encode('utf-8'),
            background_color='white',
            dpi=150,
        )
    except Exception:
        return None   # PNG not available, SVG display will be used instead

# ── Display an SVG page as an image in Streamlit ─────────────
# Embeds the SVG as a base64 <img> tag — works in ALL browsers,
# no extra libraries, white background guaranteed.
def show_svg(svg_str):
    b64 = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{b64}" '
        f'style="width:100%;background:white;border-radius:8px;'
        f'padding:12px;display:block;margin-bottom:12px;" />',
        unsafe_allow_html=True,
    )

# ── Upload ────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your audio file",
    type=["mp3","wav","mp4","ogg","flac","m4a"],
)

# ── Main pipeline ─────────────────────────────────────────────
if uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:

        raw_path = os.path.join(tmpdir, uploaded.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded.read())

        wav_path = os.path.join(tmpdir, "input.wav")
        with st.spinner("🔄 Converting audio…"):
            res = subprocess.run(
                ["ffmpeg","-y","-i",raw_path,"-ac","1","-ar","22050",wav_path],
                capture_output=True, text=True
            )
        if res.returncode != 0:
            st.error("ffmpeg failed.")
            st.code(res.stderr[-600:])
            st.stop()

        with st.spinner("📊 Analysing audio…"):
            y, sr = librosa.load(wav_path, sr=22050, mono=True)

        duration         = librosa.get_duration(y=y, sr=sr)
        tempo_arr, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val        = float(np.atleast_1d(tempo_arr)[0])
        key_str          = detect_key(y, sr)
        rms              = float(np.sqrt(np.mean(y**2)))
        zcr              = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        spec_c           = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_r           = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

        with st.spinner("🎼 Detecting pitches…"):
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz(FMIN_NOTE),
                fmax=librosa.note_to_hz(FMAX_NOTE),
                sr=sr, frame_length=2048,
            )

        frame_sec = 512 / sr
        notes, i  = [], 0
        while i < len(f0):
            if voiced_flag[i] and not np.isnan(f0[i]):
                m, si = hz_to_midi(f0[i]), i
                while (i < len(f0) and voiced_flag[i]
                       and not np.isnan(f0[i])
                       and abs(hz_to_midi(f0[i]) - m) <= 1):
                    i += 1
                dur = (i - si) * frame_sec
                if dur >= MIN_DURATION:
                    notes.append((si * frame_sec, dur, m))
            else:
                i += 1

        # Build MIDI
        midi_obj = MIDIFile(1)
        midi_obj.addTempo(0, 0, BPM)
        bps = BPM / 60.0
        for s_sec, d_sec, pitch in notes:
            midi_obj.addNote(0, 0, max(0, min(127, pitch)),
                             s_sec*bps, d_sec*bps, MIDI_VOLUME)
        midi_buf = io.BytesIO()
        midi_obj.writeFile(midi_buf)
        midi_bytes = midi_buf.getvalue()

        # Build sheet music
        xml_str   = None
        svg_pages = []    # list of SVG strings
        png_pages = []    # list of PNG bytes (or None)
        sheet_ok  = False
        sheet_err = ""
        with st.spinner("🎼 Engraving sheet music…"):
            try:
                xml_str   = build_musicxml(notes, BPM, key_str, MAX_SHEET_NOTES)
                svg_pages = render_svg_pages(xml_str)
                png_pages = [svg_to_png(svg) for svg in svg_pages]
                sheet_ok  = True
            except Exception as e:
                sheet_err = str(e)

        # ── Summary metrics ───────────────────────────────────
        st.success(f"✅ Detected **{len(notes)} notes** in {duration:.1f}s")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("🎵 Key",         key_str)
        c2.metric("🥁 Tempo",       f"{tempo_val:.1f} BPM")
        c3.metric("⏱ Duration",     f"{duration:.1f}s")
        c4.metric("🎹 Notes found", len(notes))
        c5.metric("🔊 RMS Energy",  f"{rms:.4f}")
        st.markdown("---")

        # ── Tabs ──────────────────────────────────────────────
        tab_sheet, tab_roll, tab_wave, tab_notes, tab_stats = st.tabs([
            "🎼 Sheet Music",
            "🎹 Piano Roll",
            "📈 Waveform & Spectrum",
            "📊 Note Analysis",
            "ℹ️ Full Stats",
        ])

        # ════════════════════════════════════════════════════
        # TAB 1 — SHEET MUSIC
        # ════════════════════════════════════════════════════
        with tab_sheet:
            st.subheader(f"Classical Notation — {key_str}")
            st.caption(
                f"Showing first {min(len(notes), MAX_SHEET_NOTES)} of {len(notes)} notes  ·  "
                "change 'Max notes in sheet music' in sidebar for more/fewer pages"
            )

            if not sheet_ok:
                st.error(f"Sheet music rendering failed: {sheet_err}")
            else:
                # ── Display every page as an image ────────────
                for idx, svg in enumerate(svg_pages, 1):
                    if len(svg_pages) > 1:
                        st.markdown(f"**Page {idx} / {len(svg_pages)}**")
                    show_svg(svg)   # base64 <img> — always visible

                # ── Download buttons ──────────────────────────
                st.markdown("---")
                st.markdown("**⬇️ Download sheet music:**")

                has_png = any(p is not None for p in png_pages)
                n_cols  = max(1, min(len(svg_pages), 4)) + (1 if has_png else 0)
                n_cols  = min(n_cols, 4)
                dcols   = st.columns(n_cols)

                col_i = 0
                for idx, (svg, png) in enumerate(zip(svg_pages, png_pages), 1):
                    if png is not None:
                        dcols[col_i % n_cols].download_button(
                            label=f"🖼️ PNG — Page {idx}",
                            data=png,
                            file_name=f"sheet_music_p{idx}.png",
                            mime="image/png",
                        )
                        col_i += 1
                    else:
                        # PNG conversion unavailable — offer SVG instead
                        dcols[col_i % n_cols].download_button(
                            label=f"🖼️ SVG — Page {idx}",
                            data=svg.encode("utf-8"),
                            file_name=f"sheet_music_p{idx}.svg",
                            mime="image/svg+xml",
                        )
                        col_i += 1

                if has_png:
                    st.caption("PNG: open in any image viewer, print, or share directly.")
                else:
                    st.caption(
                        "SVG: open in any browser — File → Print → Save as PDF "
                        "for high-quality output. (PNG requires libcairo system package.)"
                    )

        # ════════════════════════════════════════════════════
        # TAB 2 — PIANO ROLL
        # ════════════════════════════════════════════════════
        with tab_roll:
            st.subheader("Piano Roll — time vs. pitch")
            if notes:
                pitches = [n[2] for n in notes]
                pmin    = max(0,   min(pitches) - 3)
                pmax    = min(127, max(pitches) + 3)
                fig, ax = plt.subplots(figsize=(14, 5))
                fig.patch.set_facecolor("#0e0e1a")
                ax.set_facecolor("#13132a")
                bk = {1,3,6,8,10}
                for p in range(pmin, pmax+1):
                    ax.axhspan(p-.5, p+.5,
                               color="#1a1a30" if p%12 in bk else "#16162a", zorder=0)
                cmap = plt.cm.plasma
                for s_s, d_s, pitch in notes:
                    norm = (pitch-pmin)/max(1,pmax-pmin)
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (s_s, pitch-.45), d_s, .9,
                        boxstyle="round,pad=0.02",
                        facecolor=cmap(norm), edgecolor="none", alpha=.9
                    ))
                trange = range(pmin, pmax+1, 2)
                ax.set_yticks(list(trange))
                ax.set_yticklabels([midi_to_name(p) for p in trange],
                                   color="#ccccdd", fontsize=8)
                td = notes[-1][0]+notes[-1][1]
                ax.set_xlim(0, td); ax.set_ylim(pmin-.5, pmax+.5)
                ax.set_xlabel("Time (sec)", color="#aaaacc")
                ax.set_title("Piano Roll", color="#e8d5ff")
                ax.tick_params(colors="#aaaacc")
                for sp in ax.spines.values(): sp.set_color("#2d2d4e")
                st.pyplot(fig); plt.close(fig)
            else:
                st.warning("No notes detected. Lower Min note duration in sidebar.")

        # ════════════════════════════════════════════════════
        # TAB 3 — WAVEFORM + SPECTROGRAM
        # ════════════════════════════════════════════════════
        with tab_wave:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
            fig.patch.set_facecolor("#0e0e1a")
            for ax in (ax1, ax2):
                ax.set_facecolor("#13132a")
                for sp in ax.spines.values(): sp.set_color("#2d2d4e")
                ax.tick_params(colors="#aaaacc")
            ax1.plot(np.linspace(0,duration,len(y)), y,
                     color="#7c3aed", linewidth=0.4, alpha=0.85)
            ax1.set_ylabel("Amplitude", color="#aaaacc")
            ax1.set_title("Waveform  (yellow = beats)", color="#e8d5ff")
            for bt in librosa.frames_to_time(beats, sr=sr):
                ax1.axvline(bt, color="#f59e0b", alpha=0.3, linewidth=0.8)
            S_db = librosa.power_to_db(
                librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis="time",
                                     y_axis="mel", ax=ax2, cmap="magma")
            ax2.set_title("Mel Spectrogram", color="#e8d5ff")
            ax2.set_ylabel("Frequency (Hz)", color="#aaaacc")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

        # ════════════════════════════════════════════════════
        # TAB 4 — NOTE ANALYSIS
        # ════════════════════════════════════════════════════
        with tab_notes:
            if notes:
                pitches   = [n[2] for n in notes]
                durations = [n[1] for n in notes]
                up, counts = np.unique(pitches, return_counts=True)

                ca, cb = st.columns(2)
                with ca:
                    fig, ax = plt.subplots(figsize=(7,4))
                    fig.patch.set_facecolor("#0e0e1a"); ax.set_facecolor("#13132a")
                    ax.bar([midi_to_name(p) for p in up], counts,
                           color="#7c3aed", edgecolor="#4f46e5", linewidth=0.5)
                    plt.xticks(rotation=60, ha="right", color="#aaaacc", fontsize=8)
                    ax.set_ylabel("Count", color="#aaaacc")
                    ax.set_title("Pitch Distribution", color="#e8d5ff")
                    ax.tick_params(colors="#aaaacc")
                    for sp in ax.spines.values(): sp.set_color("#2d2d4e")
                    st.pyplot(fig); plt.close(fig)

                with cb:
                    fig, ax = plt.subplots(figsize=(7,4))
                    fig.patch.set_facecolor("#0e0e1a"); ax.set_facecolor("#13132a")
                    ax.hist(durations, bins=20, color="#4f46e5",
                            edgecolor="#7c3aed", linewidth=0.5)
                    ax.set_xlabel("Duration (sec)", color="#aaaacc")
                    ax.set_ylabel("Count", color="#aaaacc")
                    ax.set_title("Duration Distribution", color="#e8d5ff")
                    ax.tick_params(colors="#aaaacc")
                    for sp in ax.spines.values(): sp.set_color("#2d2d4e")
                    st.pyplot(fig); plt.close(fig)

                st.markdown("---")
                m1,m2,m3,m4 = st.columns(4)
                mc = up[np.argmax(counts)]
                m1.metric("Highest Note",     midi_to_name(max(pitches)))
                m2.metric("Lowest Note",      midi_to_name(min(pitches)))
                m3.metric("Most Common Note", midi_to_name(mc))
                m4.metric("Avg Duration",     f"{np.mean(durations):.2f}s")

        # ════════════════════════════════════════════════════
        # TAB 5 — FULL STATS
        # ════════════════════════════════════════════════════
        with tab_stats:
            st.markdown("### Audio Statistics")
            for k, v in {
                "Estimated Key":          key_str,
                "Estimated Tempo (BPM)":  f"{tempo_val:.2f}",
                "Duration (sec)":         f"{duration:.2f}",
                "Sample Rate (Hz)":       sr,
                "Total Samples":          len(y),
                "Notes Detected":         len(notes),
                "Notes in Sheet Music":   min(len(notes), MAX_SHEET_NOTES),
                "RMS Energy":             f"{rms:.6f}",
                "Zero-Crossing Rate":     f"{zcr:.6f}",
                "Spectral Centroid (Hz)": f"{spec_c:.1f}",
                "Spectral Rolloff (Hz)":  f"{spec_r:.1f}",
                "Output MIDI BPM":        BPM,
                "Min Note Duration (s)":  MIN_DURATION,
            }.items():
                st.markdown(f"**{k}:** `{v}`")

        # ════════════════════════════════════════════════════
        # BOTTOM DOWNLOAD BAR
        # ════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("⬇️ Downloads")
        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "⬇️ MIDI File (.mid)",
                data=midi_bytes,
                file_name="transcription.mid",
                mime="audio/midi",
                use_container_width=True,
            )
            st.caption("GarageBand · FL Studio · Ableton · MuseScore")

        with d2:
            if sheet_ok and png_pages and png_pages[0] is not None:
                st.download_button(
                    "🖼️ Sheet Music PNG (Page 1)",
                    data=png_pages[0],
                    file_name="sheet_music.png",
                    mime="image/png",
                    use_container_width=True,
                )
                st.caption("High-res PNG · Print directly · Share as image")
            elif sheet_ok and svg_pages:
                st.download_button(
                    "🖼️ Sheet Music SVG (Page 1)",
                    data=svg_pages[0].encode("utf-8"),
                    file_name="sheet_music.svg",
                    mime="image/svg+xml",
                    use_container_width=True,
                )
                st.caption("Open in browser → Print → Save as PDF")

else:
    st.info("👆 Upload an audio file above to get started.")
    st.markdown("""
    **What this app produces:**
    - 🎼 **Classical sheet music** — displayed as an image, downloadable as PNG
    - 🎹 **MIDI file** download
    - 📈 Waveform + Mel Spectrogram + beat markers
    - 📊 Key, tempo, pitch & duration analysis
    """)

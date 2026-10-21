"""
app.py — Streamlit demo app for MotionForge pipeline
Yakubu Mohammed Abass | COMP 430 Final Project

Run:
  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pipeline import run_pipeline, DEFAULT_CFG

# Page config
st.set_page_config(
    page_title="MotionForge",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital@0;1&display=swap');

html, body, [class*="css"] { font-family: 'Space Mono', monospace; }

.main { background: #09070f; }

h1 { 
    font-family: 'Space Mono', monospace !important;
    font-size: 2.4rem !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #fff 30%, #a78bfa 70%, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.05em !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    color: #6b6080 !important;
    margin-bottom: 0.5rem !important;
}

.stage-badge {
    display: inline-block;
    padding: 2px 10px;
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 3px;
    font-size: 11px;
    color: #a78bfa;
    font-family: monospace;
    letter-spacing: 0.15em;
    margin-bottom: 12px;
}

.stProgress > div > div { background: linear-gradient(90deg, #a78bfa, #34d399); }

[data-testid="stSidebar"] {
    background: #0e0a1a !important;
    border-right: 1px solid #1e1830 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c4dff, #00bfa5) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.1em !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    margin-top: 1rem !important;
}

.stButton > button:hover { opacity: 0.88 !important; }

.output-label {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b6080;
    margin-bottom: 4px;
    font-family: monospace;
}

div[data-testid="stImage"] img { border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# SIDEBAR — ALL SETTINGS
with st.sidebar:
    st.markdown("### MOTIONFORGE")
    st.markdown("*One Frame, Many Moments*")
    st.markdown("---")

    # Upload 
    st.markdown("### 01 / VIDEO INPUT")
    uploaded = st.file_uploader(
        "Upload a short video (mp4, mov, avi)",
        type=["mp4", "mov", "avi", "mkv"],
        help="Record with a static camera. 3–10 seconds works best."
    )

    st.markdown("---")

    # Extraction 
    st.markdown("### 02 / EXTRACTION")
    frame_skip  = st.slider("Frame skip",       2, 10, 4,
                             help="Sample every N frames. Lower = more silhouettes")
    max_frames  = st.slider("Max silhouettes", 6, 24, 14,
                             help="How many poses appear in the composite")
    mog2_thresh = st.slider("MOG2 threshold",  20, 80, 35,
                             help="Motion sensitivity. Lower = more sensitive")
    out_width   = st.select_slider("Output width",
                                   options=[640, 960, 1280, 1920], value=1280)

    st.markdown("---")

    # Van Gogh
    st.markdown("### 03 / VAN GOGH STROKES")
    vg_enabled = st.toggle("Enable Van Gogh layer", value=True)

    if vg_enabled:
        vg_step   = st.slider("Stroke density (grid step px)", 4, 20, 10,
                               help="Lower = denser, painterly. Higher = sparser")
        vg_length = st.slider("Stroke length px",  10, 55, 28)
        vg_width  = st.slider("Stroke width px",    1,  5,  2)
        vg_curve  = st.slider("Curvature",        0.0, 1.0, 0.35, step=0.05,
                               help="0 = straight lines. 1 = very curvy/swirling")
        vg_blur   = st.slider("Flow blur (swirl amount)", 8, 50, 25,
                               help="Higher = more Starry Night turbulence")
    else:
        vg_step = vg_length = vg_width = vg_blur = 0
        vg_curve = 0.0

    st.markdown("---")

    # Paint Strokes
    st.markdown("### 04 / PAINT STROKES")
    paint_enabled = st.toggle("Enable paint stroke layer", value=True)

    if paint_enabled:
        paint_style   = st.radio("Style", ["watercolor", "oil"],
                                  help="Watercolor = soft + translucent. Oil = thick impasto")
        paint_density = st.slider("Stroke density",   3, 14, 6,
                                   help="Lower = more bristle coverage")
        paint_max_len = st.slider("Max stroke length px", 8, 48, 28)
        paint_bleed   = st.slider("Bleed / spread",  0.0, 0.8, 0.3, step=0.05,
                                   help="How far watercolor bleeds beyond silhouette edge")
        paint_alpha   = st.slider("Layer opacity",   0.3, 1.0, 0.75, step=0.05)
    else:
        paint_style   = "watercolor"
        paint_density = paint_max_len = 0
        paint_bleed = paint_alpha = 0.0

    st.markdown("---")

    # Contours
    st.markdown("### 05 / CONTOUR RINGS")
    contour_rings  = st.slider("Number of rings", 4, 20, 14)
    contour_glow   = st.toggle("Glow halo on contours", value=True)

    st.markdown("---")

    # Post-FX
    st.markdown("### 06 / POST-FX")
    glow_str  = st.slider("Bloom glow",   0, 40, 20)
    grain_int = st.slider("Film grain",   0, 30, 10)
    vignette  = st.slider("Vignette",   0.0, 1.0, 0.6, step=0.05)

    st.markdown("---")

    # Animation
    st.markdown("### 07 / ANIMATION")
    animate     = st.toggle("Generate animation", value=True)
    if animate:
        anim_fmt = st.radio("Format", ["gif", "mp4"], horizontal=True)
        anim_fps = st.slider("FPS", 4, 24, 8)
    else:
        anim_fmt = "gif"
        anim_fps = 8

    st.markdown("---")
    run_btn = st.button("▶  RUN PIPELINE")


# MAIN AREA
st.title("MotionForge")
st.markdown("*One Frame, Many Moments*")
st.markdown("**Yakubu Mohammed Abass · COMP 430 Final Project**")
st.markdown("---")

if not uploaded and not run_btn:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stage-badge">STAGE 01 → 02</div>', unsafe_allow_html=True)
        st.markdown("#### Frame Extraction + Optical Flow")
        st.markdown("MOG2 background subtraction isolates moving person. Farneback dense flow accumulates motion energy across every frame pair.")
    with col2:
        st.markdown('<div class="stage-badge">STAGE 03 → 04</div>', unsafe_allow_html=True)
        st.markdown("#### Van Gogh + Paint Strokes")
        st.markdown("Flow vectors rendered as curved brush strokes following Starry Night turbulence. Sobel-directed bristle strokes painted onto each silhouette.")
    with col3:
        st.markdown('<div class="stage-badge">STAGE 05 → 07</div>', unsafe_allow_html=True)
        st.markdown("#### Contours + Post-FX + Animation")
        st.markdown("Body outlines colored teal → white → gold across time. Bloom glow, film grain, vignette. Animated GIF/MP4 shows the composite building live.")

    st.info("⬅  Upload a video and configure settings in the sidebar, then click **RUN PIPELINE**.")
    st.stop()

if not uploaded:
    st.warning("Please upload a video file in the sidebar first.")
    st.stop()

# Build cfg
cfg = DEFAULT_CFG.copy()
cfg.update({
    "frame_skip"           : frame_skip,
    "max_silhouettes"      : max_frames,
    "mog2_threshold"       : mog2_thresh,
    "output_width"         : out_width,

    "vg_enabled"           : vg_enabled,
    "vg_stroke_step"       : vg_step   if vg_enabled else 10,
    "vg_stroke_length"     : vg_length if vg_enabled else 28,
    "vg_stroke_width"      : vg_width  if vg_enabled else 2,
    "vg_stroke_curve"      : vg_curve  if vg_enabled else 0.35,
    "vg_flow_blur"         : vg_blur   if vg_enabled else 25,

    "paint_enabled"        : paint_enabled,
    "paint_style"          : paint_style,
    "paint_stroke_density" : paint_density if paint_enabled else 6,
    "paint_stroke_max_len" : paint_max_len if paint_enabled else 28,
    "paint_bleed_alpha"    : paint_bleed   if paint_enabled else 0.3,
    "paint_layer_alpha"    : paint_alpha   if paint_enabled else 0.75,

    "contour_rings"        : contour_rings,
    "contour_glow"         : contour_glow,

    "glow_strength"        : glow_str,
    "grain_intensity"      : grain_int,
    "vignette_strength"    : vignette,

    "animate"              : animate,
    "anim_format"          : anim_fmt,
    "anim_fps"             : anim_fps,
})

# Show video preview
with st.expander("📹 Uploaded video preview", expanded=False):
    st.video(uploaded)

# Run pipeline
if run_btn:
    # Save upload to temp file
    suffix = os.path.splitext(uploaded.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    out_dir = tempfile.mkdtemp(prefix="motionforge_")

    # Progress UI
    st.markdown("---")
    st.markdown("### ⚙️ PIPELINE RUNNING")
    prog_bar  = st.progress(0.0)
    prog_text = st.empty()
    stage_col1, stage_col2, stage_col3, stage_col4 = st.columns(4)
    stage_indicators = {
        "extract" : stage_col1.empty(),
        "flow"    : stage_col2.empty(),
        "render"  : stage_col3.empty(),
        "animate" : stage_col4.empty(),
    }

    def _update_stages(frac):
        icons = ["⏳","⏳","⏳","⏳"]
        if frac >= 0.05: icons[0] = "✅"
        if frac >= 0.28: icons[1] = "✅"
        if frac >= 0.85: icons[2] = "✅"
        if frac >= 0.99: icons[3] = "✅"
        stage_indicators["extract"].markdown(f"{icons[0]} Extraction")
        stage_indicators["flow"].markdown(f"{icons[1]} Flow")
        stage_indicators["render"].markdown(f"{icons[2]} Rendering")
        stage_indicators["animate"].markdown(f"{icons[3]} Animation")

    def progress_cb(frac, msg):
        prog_bar.progress(min(frac, 1.0))
        prog_text.markdown(f"`{msg}`")
        _update_stages(frac)

    _update_stages(0)

    try:
        t0      = time.time()
        outputs = run_pipeline(tmp_path, cfg, out_dir, progress_cb)
        elapsed = time.time() - t0

        prog_bar.progress(1.0)
        prog_text.markdown(f"`Done in {elapsed:.1f}s`")
        _update_stages(1.0)

        st.success(f"✅ Pipeline complete in **{elapsed:.1f}s**")
        st.markdown("---")

        # Results
        st.markdown("### 🎨 RESULTS")

        # Pipeline stages in a 4-column grid
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        stages = [
            ("vangogh",   "01 Van Gogh Strokes"),
            ("paint",     "02 Paint Layer"),
            ("contours",  "03 Contour Rings"),
            ("final",     "04 ★ FINAL"),
        ]
        for col, (key, label) in zip(cols, stages):
            path = outputs.get(key)
            if path and os.path.exists(path):
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with col:
                    st.markdown(f'<div class="output-label">{label}</div>', unsafe_allow_html=True)
                    st.image(img, use_container_width=True)
                    with open(path, "rb") as f:
                        st.download_button(
                            f"⬇ Download",
                            data=f,
                            file_name=os.path.basename(path),
                            mime="image/png",
                            key=f"dl_{key}",
                        )

        # Final large view 
        st.markdown("---")
        st.markdown("### FINAL COMPOSITE")
        final_path = outputs.get("final")
        if final_path and os.path.exists(final_path):
            img = cv2.imread(final_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, use_container_width=True)

        # Animation
        anim_path = outputs.get("animation")
        if anim_path and os.path.exists(anim_path):
            st.markdown("---")
            st.markdown("### ANIMATION — Motion Building Over Time")

            if anim_path.endswith(".gif"):
                with open(anim_path, "rb") as f:
                    gif_bytes = f.read()
                st.image(gif_bytes, use_container_width=True)
                st.download_button(
                    "⬇ Download GIF",
                    data=gif_bytes,
                    file_name="motionforge.gif",
                    mime="image/gif",
                )
            else:
                with open(anim_path, "rb") as f:
                    mp4_bytes = f.read()
                st.video(mp4_bytes)
                st.download_button(
                    "⬇ Download MP4",
                    data=mp4_bytes,
                    file_name="motionforge.mp4",
                    mime="video/mp4",
                )

        # Config summary 
        with st.expander("⚙️ Configuration used"):
            st.json({k: v for k, v in cfg.items()
                     if not isinstance(v, (list, tuple, np.ndarray))})

    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        st.exception(e)

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

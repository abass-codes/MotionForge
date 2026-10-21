# MotionForge: One Frame, Many Moments
### Van Gogh Flow Field + Paint Strokes

---

## Project Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Core engine — shared by CLI and Streamlit |
| `run.py` | Command-line interface with full `--args` |
| `app.py` | Streamlit web app — upload video, tune settings, visualize |
| `requirements.txt` | Python dependencies |
| `./videos/` | Sample input videos to test with |

---

## Install

```bash
pip install -r requirements.txt
```

---

## Option A — Command Line

```bash
# Default: Van Gogh + watercolor paint + animation GIF
python run.py --video jump.mp4

# Van Gogh only (no paint strokes)
python run.py --video dance.mp4 --no-paint

# Paint only (no Van Gogh background)
python run.py --video run.mp4 --no-vangogh --paint-style oil

# Dense Van Gogh strokes + tight watercolor
python run.py --video jump.mp4 \
    --vg-step 6 --vg-length 35 --vg-curve 0.5 \
    --paint-style watercolor --paint-density 4

# Oil impasto, dark canvas, no animation
python run.py --video dance.mp4 \
    --paint-style oil --no-animate

# MP4 animation at higher FPS
python run.py --video jump.mp4 --anim-format mp4 --anim-fps 12

# All controls
python run.py --help
```

### All Arguments

**Extraction**
| Arg | Default | Description |
|-----|---------|-------------|
| `--frame-skip N` | 4 | Sample every N frames |
| `--max-frames N` | 16 | Max silhouettes in composite |
| `--mog2-thresh N` | 35 | MOG2 sensitivity (lower = more motion) |
| `--width N` | 1280 | Output width in pixels |

**Van Gogh Strokes**
| Arg | Default | Description |
|-----|---------|-------------|
| `--no-vangogh` | — | Disable Van Gogh background |
| `--vg-step N` | 10 | Grid spacing px — lower = denser |
| `--vg-length N` | 28 | Max stroke length px |
| `--vg-width N` | 2 | Stroke thickness px |
| `--vg-curve F` | 0.35 | Curvature 0=straight 1=very curvy |
| `--vg-blur N` | 25 | Flow smoothing — higher = more swirl |

**Paint Strokes**
| Arg | Default | Description |
|-----|---------|-------------|
| `--no-paint` | — | Disable paint stroke layer |
| `--paint-style` | watercolor | watercolor or oil |
| `--paint-density N` | 6 | Stroke grid inside silhouette |
| `--paint-max-len N` | 28 | Max paint stroke length |
| `--paint-bleed F` | 0.30 | Watercolor bleed opacity 0–1 |
| `--paint-alpha F` | 0.75 | Paint layer opacity 0–1 |

**Contours**
| Arg | Default | Description |
|-----|---------|-------------|
| `--contour-rings N` | 14 | Number of contour rings |
| `--no-contour-glow` | — | Disable glow halo on contours |

**Post-FX**
| Arg | Default | Description |
|-----|---------|-------------|
| `--glow N` | 20 | Bloom glow strength |
| `--grain N` | 10 | Film grain intensity |
| `--vignette F` | 0.6 | Vignette strength 0–1 |

**Animation**
| Arg | Default | Description |
|-----|---------|-------------|
| `--no-animate` | — | Skip animation |
| `--anim-format` | gif | gif or mp4 |
| `--anim-fps N` | 8 | Animation frames per second |

---

## Option B — Streamlit App (Recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

- Upload any video via the browser
- Tune every parameter with interactive sliders
- Watch the 4 pipeline stages render in real time
- Download final image + animation directly from the browser

---

## Pipeline Stages

```
Video -> MOG2 Extraction -> Farneback Optical Flow
      -> Van Gogh Stroke Canvas (flow-directed curved strokes)
      -> Paint Stroke Layer (Sobel-directed bristle strokes per silhouette)
      -> Contour Rings (body outline colored teal->white->gold by time)
      -> Post-FX (bloom glow · film grain · vignette · color grade)
      -> Animation (GIF/MP4 showing composite building frame by frame)
```

### Output Files

```
output/
  01_vangogh_strokes.png   Van Gogh flow field canvas
  02_paint_layer.png       Paint strokes composited on flow canvas
  03_contours.png          Contour rings added
  04_FINAL.png             Full post-FX final image  ★
  05_animation.gif         Animation of composite building up
```

---

## Tips for Best Results

- **Camera**: must be static (tripod or propped phone)
- **Background**: plain wall or door — less texture = cleaner silhouette
- **Lighting**: even lighting, avoid strong shadows that move
- **Motion**: 3–8 seconds, clear deliberate movement (jump, wave, dance move)
- **Dense painting**: lower `--vg-step` to 6 and `--paint-density` to 4
- **More swirl**: raise `--vg-curve` to 0.6 and `--vg-blur` to 40
- **More silhouettes**: lower `--mog2-thresh` to 25–28

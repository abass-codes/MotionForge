"""
pipeline.py — Core engine shared by CLI (run.py) and Streamlit app (app.py)
MotionForge: One Frame, Many Moments
Yakubu Mohammed Abass | COMP 430 Final Project
"""

import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter


# DEFAULTS
DEFAULT_CFG = {
    # Extraction
    "frame_skip"           : 4,
    "max_silhouettes"      : 16,
    "bg_history"           : 40,
    "mog2_threshold"       : 35,
    "morph_kernel"         : 9,
    "min_motion_pixels"    : 600,
    "output_width"         : 1280,

    # Van Gogh
    "vg_enabled"           : True,
    "vg_stroke_step"       : 10,
    "vg_stroke_length"     : 28,
    "vg_stroke_width"      : 2,
    "vg_stroke_curve"      : 0.35,
    "vg_flow_blur"         : 25,

    # Paint strokes
    "paint_enabled"        : True,
    "paint_style"          : "watercolor",   # watercolor | oil
    "paint_stroke_density" : 6,
    "paint_stroke_min_len" : 8,
    "paint_stroke_max_len" : 28,
    "paint_stroke_min_w"   : 1,
    "paint_stroke_max_w"   : 4,
    "paint_bleed_sigma"    : 5,
    "paint_bleed_alpha"    : 0.30,
    "paint_layer_alpha"    : 0.75,

    # Contours
    "contour_rings"        : 14,
    "contour_thickness"    : 2,
    "contour_glow"         : True,

    # Canvas
    "canvas_bg"            : (14, 10, 22),

    # Post-FX
    "glow_strength"        : 20,
    "grain_intensity"      : 10,
    "vignette_strength"    : 0.6,

    # Animation
    "animate"              : True,
    "anim_fps"             : 8,
    "anim_format"          : "gif",   # gif | mp4
}

VG_PALETTE = [
    (180,120, 60),(100,160,200),( 80,140, 80),
    (200,100, 80),(140, 80,180),(220,180, 80),
    ( 60,160,180),(200,140, 60),
]

PAINT_PALETTE = {
    "watercolor": [
        [(180,120,60),(140,90,180),(80,160,200)],
        [(200,140,60),(160,100,200),(100,180,140)],
        [(220,180,80),(200,90,110),(180,140,60)],
    ],
    "oil": [
        [(200,140,80),(80,120,200),(100,60,160)],
        [(220,100,60),(140,200,100),(60,160,200)],
        [(240,200,80),(200,60, 80),(80,200,160)],
    ],
}



# STAGE 1 — EXTRACTION
def extract_frames(video_path, cfg, progress_cb=None):
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=cfg["bg_history"],
        varThreshold=cfg["mog2_threshold"],
        detectShadows=False,
    )
    k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["morph_kernel"],   cfg["morph_kernel"]))
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["morph_kernel"]+4, cfg["morph_kernel"]+4))

    all_frames, silhouettes = [], []
    ref_frame = None
    fc = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cfg["output_width"]:
            h, w  = frame.shape[:2]
            s     = cfg["output_width"] / w
            frame = cv2.resize(frame, (cfg["output_width"], int(h * s)))
        if fc == cfg["bg_history"] // 2:
            ref_frame = frame.copy()

        fg = bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_o)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_c)
        fg = _feather(fg, sigma=4)

        all_frames.append(frame.copy())

        if fc > cfg["bg_history"] and fc % cfg["frame_skip"] == 0:
            mp = cv2.countNonZero((fg > 30).astype(np.uint8))
            if mp > cfg["min_motion_pixels"]:
                silhouettes.append({"idx": fc, "frame": frame.copy(), "mask": fg.copy(), "motion": mp})

        if progress_cb and fc % 15 == 0:
            progress_cb(fc / max(total, 1), f"Extracting frames {fc}/{total}")
        fc += 1

    cap.release()
    return all_frames, silhouettes, ref_frame, fps


# STAGE 2 — OPTICAL FLOW
def compute_flow(all_frames, cfg, progress_cb=None):
    h, w = all_frames[0].shape[:2]
    fu   = np.zeros((h, w), dtype=np.float32)
    fv   = np.zeros((h, w), dtype=np.float32)
    prev = cv2.cvtColor(all_frames[0], cv2.COLOR_BGR2GRAY)
    n    = len(all_frames)

    for i in range(1, n):
        curr = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag  = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        fu  += flow[:,:,0] * mag
        fv  += flow[:,:,1] * mag
        prev = curr
        if progress_cb and i % 20 == 0:
            progress_cb(i / n, f"Computing optical flow {i}/{n}")

    mag_total = np.sqrt(fu**2 + fv**2) + 1e-6
    fu /= mag_total
    fv /= mag_total
    flow_mag = gaussian_filter(mag_total, sigma=cfg["vg_flow_blur"])
    fu = gaussian_filter(fu, sigma=4)
    fv = gaussian_filter(fv, sigma=4)
    return fu, fv, flow_mag


# STAGE 3 — VAN GOGH STROKE CANVAS
def render_vangogh(flow_u, flow_v, flow_mag, h, w, cfg, progress_cb=None):
    if not cfg["vg_enabled"]:
        return np.full((h, w, 3), cfg["canvas_bg"], dtype=np.float32)

    canvas  = np.full((h, w, 3), cfg["canvas_bg"], dtype=np.float32)
    step    = cfg["vg_stroke_step"]
    palette = np.array(VG_PALETTE, dtype=np.float32)
    max_mag = flow_mag.max() + 1e-6
    ys      = np.arange(step, h - step, step)
    xs      = np.arange(step, w - step, step)
    total   = len(ys) * len(xs)
    done    = 0

    for y in ys:
        for x in xs:
            mn  = flow_mag[y, x] / max_mag
            if mn < 0.015:
                done += 1
                continue
            u     = flow_u[y, x]
            v     = flow_v[y, x]
            angle = np.arctan2(v, u)
            alpha = 0.3 + 0.6 * mn
            length = cfg["vg_stroke_length"] * (0.4 + 0.6 * mn)

            pp    = (angle + np.pi) / (2 * np.pi) * (len(palette) - 1)
            lo    = int(pp) % len(palette)
            hi    = (lo + 1) % len(palette)
            color = palette[lo] * (1 - (pp % 1)) + palette[hi] * (pp % 1)
            color = np.clip(color * (0.55 + 0.8 * mn), 0, 255)

            _draw_curved_stroke(canvas, x, y, angle, length, color, alpha,
                                flow_u, flow_v, cfg["vg_stroke_curve"],
                                cfg["vg_stroke_width"])
            done += 1

        if progress_cb and done % (len(xs) * 5) == 0:
            progress_cb(done / max(total, 1), "Painting Van Gogh strokes...")

    return canvas


def _draw_curved_stroke(canvas, x, y, angle, length, color, alpha, fu, fv, curve, width):
    h, w = canvas.shape[:2]
    segs  = 6
    seg_l = length / segs
    cx, cy, ca = float(x), float(y), angle
    pts   = [(int(cx), int(cy))]

    for _ in range(segs):
        nx = cx + np.cos(ca) * seg_l
        ny = cy + np.sin(ca) * seg_l
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            break
        pts.append((int(nx), int(ny)))
        lx = int(np.clip(nx, 0, w-1))
        ly = int(np.clip(ny, 0, h-1))
        la = np.arctan2(fv[ly, lx], fu[ly, lx])
        ca = ca * (1 - curve) + la * curve
        cx, cy = nx, ny

    if len(pts) < 2:
        return
    layer = np.zeros_like(canvas)
    for i in range(len(pts)-1):
        cv2.line(layer, pts[i], pts[i+1],
                 (float(color[0]), float(color[1]), float(color[2])),
                 max(1, int(width)), cv2.LINE_AA)
    mask = (layer.sum(axis=2) > 0).astype(np.float32)[:,:,np.newaxis]
    canvas += layer * mask * alpha


# STAGE 4 — PAINT STROKE LAYER
def render_paint_layer(silhouettes, canvas_base, cfg, progress_cb=None):
    if not cfg["paint_enabled"]:
        return canvas_base.copy()

    canvas  = canvas_base.copy()
    palette = PAINT_PALETTE[cfg["paint_style"]]
    n       = len(silhouettes)
    is_oil  = cfg["paint_style"] == "oil"

    for i, sil in enumerate(silhouettes):
        t = i / max(n-1, 1)

        pi   = min(int(t * (len(palette)-1)), len(palette)-2)
        pf   = t * (len(palette)-1) - pi
        clo  = palette[pi][i % len(palette[pi])]
        chi  = palette[min(pi+1, len(palette)-1)][i % len(palette[pi])]
        color = tuple(int(clo[c]*(1-pf) + chi[c]*pf) for c in range(3))

        # Watercolor bleed
        if not is_oil and cfg["paint_bleed_alpha"] > 0:
            canvas = _paint_bleed(canvas, sil["mask"], color, cfg["paint_bleed_alpha"], cfg["paint_bleed_sigma"])

        # Sobel-directed brush strokes
        stroke_layer = _paint_strokes(sil["frame"], sil["mask"], color, cfg, is_oil)
        smask = (stroke_layer.sum(axis=2) > 0).astype(np.float32)[:,:,np.newaxis]
        canvas = canvas * (1 - smask * cfg["paint_layer_alpha"]) + stroke_layer * cfg["paint_layer_alpha"]

        if progress_cb:
            progress_cb(i / n, f"Painting stroke layer {i+1}/{n}")

    return np.clip(canvas, 0, 255)


def _paint_bleed(canvas, mask, color, alpha, sigma):
    h, w      = canvas.shape[:2]
    bleed     = np.zeros((h, w, 3), dtype=np.float32)
    mask_f    = (mask > 30).astype(np.float32)
    for c in range(3):
        bleed[:,:,c] = mask_f * color[c]
    bleed_blur = cv2.GaussianBlur(bleed, (0,0), sigma)
    bmask      = (bleed_blur.sum(axis=2) > 0).astype(np.float32)[:,:,np.newaxis]
    return canvas + bleed_blur * bmask * alpha


def _paint_strokes(frame, mask, color, cfg, is_oil):
    h, w     = frame.shape[:2]
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx       = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy       = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag      = np.sqrt(gx**2 + gy**2)
    angle    = np.arctan2(gy, gx) + np.pi / 2
    angle    = gaussian_filter(angle, sigma=3)
    mag      = gaussian_filter(mag, sigma=2)
    max_mag  = mag.max() + 1e-6
    mask_bin = (mask > 30).astype(np.float32)
    mag     *= mask_bin

    step    = cfg["paint_stroke_density"]
    layer   = np.zeros((h, w, 3), dtype=np.float32)

    for y in range(step, h-step, step):
        for x in range(step, w-step, step):
            if mask[y, x] < 30:
                continue
            mn  = mag[y, x] / max_mag
            ang = angle[y, x]
            jx  = int(np.clip(x + np.random.uniform(-3, 3), 0, w-1))
            jy  = int(np.clip(y + np.random.uniform(-3, 3), 0, h-1))

            slen = cfg["paint_stroke_min_len"] + mn * (cfg["paint_stroke_max_len"] - cfg["paint_stroke_min_len"])
            sw   = cfg["paint_stroke_min_w"]   + mn * (cfg["paint_stroke_max_w"]   - cfg["paint_stroke_min_w"])
            if is_oil:
                sw *= 1.8

            tv   = np.random.uniform(0.88, 1.12)
            col  = tuple(int(np.clip(c * tv, 0, 255)) for c in color)
            p1   = (int(np.clip(jx - np.cos(ang)*slen/2, 0, w-1)),
                    int(np.clip(jy - np.sin(ang)*slen/2, 0, h-1)))
            p2   = (int(np.clip(jx + np.cos(ang)*slen/2, 0, w-1)),
                    int(np.clip(jy + np.sin(ang)*slen/2, 0, h-1)))
            cv2.line(layer, p1, p2,
                     (float(col[0]), float(col[1]), float(col[2])),
                     max(1, int(sw)), cv2.LINE_AA)

            if is_oil and mn > 0.5:
                hl = tuple(int(min(255, c*1.6)) for c in col)
                cv2.circle(layer, (jx,jy), max(1, int(sw*0.3)),
                           (float(hl[0]), float(hl[1]), float(hl[2])), -1)
    return layer


# STAGE 5 — CONTOUR RINGS
def render_contours(canvas, silhouettes, cfg, progress_cb=None):
    n   = len(silhouettes)
    out = canvas.astype(np.float32)
    h, w = canvas.shape[:2]

    for i, sil in enumerate(silhouettes):
        t = i / max(n-1, 1)
        if t < 0.5:
            t2    = t * 2
            color = (int(40+180*t2), int(180+75*t2), int(80+120*t2))
        else:
            t2    = (t-0.5)*2
            color = (int(220-180*t2), int(255-130*t2), int(200+55*t2))

        binary   = (sil["mask"] > 30).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        if cfg["contour_glow"]:
            gl = np.zeros((h, w, 3), dtype=np.float32)
            cv2.drawContours(gl, contours, -1, color, cfg["contour_thickness"]+6, cv2.LINE_AA)
            out = np.clip(out + cv2.GaussianBlur(gl, (0,0), 8) * 0.5, 0, 255)

        cv2.drawContours(out, contours, -1,
                         (float(color[0]), float(color[1]), float(color[2])),
                         cfg["contour_thickness"], cv2.LINE_AA)

        if progress_cb:
            progress_cb(i / n, f"Drawing contour rings {i+1}/{n}")

    return np.clip(out, 0, 255).astype(np.uint8)


# STAGE 6 — POST-FX
def post_fx(img, cfg):
    img = _bloom_glow(img, cfg["glow_strength"])
    img = _color_grade(img, cfg["paint_style"])
    img = _vignette(img, cfg["vignette_strength"])
    img = _grain(img, cfg["grain_intensity"])
    return img


def _bloom_glow(img, strength):
    blur  = cv2.GaussianBlur(img, (0,0), strength)
    glow  = cv2.addWeighted(img, 1.0, blur, 0.35, 0)
    return np.clip(glow, 0, 255).astype(np.uint8)


def _color_grade(img, style):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    if style == "watercolor":
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.15, 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.05, 0, 255)
    else:
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.35, 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.10, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _vignette(img, strength):
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X-w/2)/(w/2))**2 + ((Y-h/2)/(h/2))**2)
    vig  = 1 - np.clip(dist * strength, 0, 1)
    return np.clip(img.astype(np.float32) * vig[:,:,np.newaxis], 0, 255).astype(np.uint8)


def _grain(img, intensity):
    noise = np.random.normal(0, intensity, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# STAGE 7 — ANIMATION
def render_animation(silhouettes, flow_u, flow_v, flow_mag, cfg,
                     output_path, progress_cb=None):
    """
    Build an animation showing the composite constructing itself:
    - Start with just the Van Gogh flow canvas
    - Progressively add each silhouette's paint strokes + contour
    - End on the fully post-processed final frame
    Each frame in the animation = one more silhouette added.
    """
    h, w     = silhouettes[0]["frame"].shape[:2]
    n        = len(silhouettes)
    anim_frames = []

    # Base canvas: Van Gogh strokes (rendered once)
    if progress_cb:
        progress_cb(0.0, "Rendering base Van Gogh canvas for animation...")
    vg_canvas = render_vangogh(flow_u, flow_v, flow_mag, h, w, cfg)
    vg_uint8  = np.clip(vg_canvas, 0, 255).astype(np.uint8)

    palette   = PAINT_PALETTE[cfg["paint_style"]]
    is_oil    = cfg["paint_style"] == "oil"
    canvas    = vg_canvas.copy()

    for i, sil in enumerate(silhouettes):
        t = i / max(n-1, 1)

        # Paint bleed + strokes for this silhouette
        pi    = min(int(t*(len(palette)-1)), len(palette)-2)
        pf    = t*(len(palette)-1) - pi
        clo   = palette[pi][i % len(palette[pi])]
        chi   = palette[min(pi+1, len(palette)-1)][i % len(palette[pi])]
        color = tuple(int(clo[c]*(1-pf)+chi[c]*pf) for c in range(3))

        if not is_oil and cfg["paint_bleed_alpha"] > 0:
            canvas = _paint_bleed(canvas, sil["mask"], color,
                                  cfg["paint_bleed_alpha"], cfg["paint_bleed_sigma"])
        stroke_layer = _paint_strokes(sil["frame"], sil["mask"], color, cfg, is_oil)
        smask = (stroke_layer.sum(axis=2) > 0).astype(np.float32)[:,:,np.newaxis]
        canvas = canvas * (1-smask*cfg["paint_layer_alpha"]) + stroke_layer*cfg["paint_layer_alpha"]

        # Contour for this silhouette
        frame_canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        if t < 0.5:
            t2 = t*2; color_c = (int(40+180*t2), int(180+75*t2), int(80+120*t2))
        else:
            t2 = (t-0.5)*2; color_c = (int(220-180*t2), int(255-130*t2), int(200+55*t2))

        binary   = (sil["mask"] > 30).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            if cfg["contour_glow"]:
                gl = np.zeros((h, w, 3), dtype=np.float32)
                cv2.drawContours(gl, contours, -1, color_c, cfg["contour_thickness"]+6, cv2.LINE_AA)
                frame_canvas = np.clip(
                    frame_canvas.astype(np.float32) + cv2.GaussianBlur(gl,(0,0),8)*0.5,
                    0, 255).astype(np.uint8)
            cv2.drawContours(frame_canvas, contours, -1, color_c,
                             cfg["contour_thickness"], cv2.LINE_AA)

        # Light post-fx per animation frame (skip heavy grain for speed)
        frame_out = _vignette(frame_canvas, cfg["vignette_strength"] * 0.7)

        # Progress counter overlay
        label = f"{i+1:02d} / {n:02d}  MOMENTS"
        cv2.putText(frame_out, label, (20, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,160,220), 1, cv2.LINE_AA)

        anim_frames.append(frame_out)

        if progress_cb:
            progress_cb(i / n, f"Building animation frame {i+1}/{n}")

    # Final frame: fully post-processed (hold for longer)
    final = post_fx(np.clip(canvas, 0, 255).astype(np.uint8), cfg)
    for _ in range(4):   # hold final frame
        anim_frames.append(final)

    # Save
    fmt = cfg["anim_format"].lower()
    if fmt == "gif":
        _save_gif(anim_frames, output_path, cfg["anim_fps"])
    else:
        _save_mp4(anim_frames, output_path, cfg["anim_fps"])

    return anim_frames


def _save_gif(frames, path, fps):
    try:
        import imageio
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        duration   = 1.0 / fps
        imageio.mimsave(path, rgb_frames, duration=duration, loop=0)
    except ImportError:
        print("  imageio not installed — falling back to MP4")
        mp4_path = path.replace(".gif", ".mp4")
        _save_mp4(frames, mp4_path, fps)


def _save_mp4(frames, path, fps):
    h, w   = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


# FULL PIPELINE RUNNER
def run_pipeline(video_path, cfg=None, output_dir="output", progress_cb=None):
    """
    End-to-end pipeline. Returns dict of output paths.
    progress_cb(fraction: float, message: str) — optional callback for Streamlit.
    """
    if cfg is None:
        cfg = DEFAULT_CFG.copy()

    os.makedirs(output_dir, exist_ok=True)

    def _prog(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)
        else:
            print(f"  [{int(frac*100):3d}%] {msg}")

    # 1. Extract
    _prog(0.0, "Extracting frames + silhouettes...")
    all_frames, silhouettes, ref_frame, src_fps = extract_frames(video_path, cfg, _prog)
    if not silhouettes:
        raise ValueError("No motion detected. Try lowering mog2_threshold.")

    # Select evenly spaced subset
    n   = cfg["max_silhouettes"]
    idx = np.linspace(0, len(silhouettes)-1, min(n, len(silhouettes)), dtype=int)
    silhouettes = [silhouettes[i] for i in idx]
    h, w = silhouettes[0]["frame"].shape[:2]

    # 2. Optical flow
    _prog(0.1, "Computing optical flow...")
    flow_u, flow_v, flow_mag = compute_flow(all_frames, cfg, _prog)

    # 3. Van Gogh canvas
    _prog(0.3, "Rendering Van Gogh brush strokes...")
    vg_canvas = render_vangogh(flow_u, flow_v, flow_mag, h, w, cfg, _prog)

    # Intermediate save
    vg_path = os.path.join(output_dir, "01_vangogh_strokes.png")
    cv2.imwrite(vg_path, np.clip(vg_canvas, 0, 255).astype(np.uint8))

    # 4. Paint strokes
    _prog(0.55, "Rendering paint stroke layer...")
    painted = render_paint_layer(silhouettes, vg_canvas, cfg, _prog)

    paint_path = os.path.join(output_dir, "02_paint_layer.png")
    cv2.imwrite(paint_path, np.clip(painted, 0, 255).astype(np.uint8))

    # 5. Contour rings
    _prog(0.72, "Drawing contour rings...")
    with_contours = render_contours(painted.astype(np.uint8), silhouettes, cfg, _prog)

    contour_path = os.path.join(output_dir, "03_contours.png")
    cv2.imwrite(contour_path, with_contours)

    # 6. Post-FX
    _prog(0.85, "Applying post-FX...")
    final = post_fx(with_contours, cfg)

    final_path = os.path.join(output_dir, "04_FINAL.png")
    cv2.imwrite(final_path, final)

    # 7. Animation
    anim_path = None
    if cfg["animate"]:
        ext       = "gif" if cfg["anim_format"] == "gif" else "mp4"
        anim_path = os.path.join(output_dir, f"05_animation.{ext}")
        _prog(0.88, f"Rendering animation ({ext})...")
        render_animation(silhouettes, flow_u, flow_v, flow_mag, cfg,
                         anim_path, _prog)

    _prog(1.0, "Done!")

    return {
        "vangogh"  : vg_path,
        "paint"    : paint_path,
        "contours" : contour_path,
        "final"    : final_path,
        "animation": anim_path,
    }

# HELPER
def _feather(mask, sigma=3):
    return np.clip(gaussian_filter(mask.astype(np.float32), sigma=sigma), 0, 255).astype(np.uint8)

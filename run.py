"""
run.py — Command-line interface for MotionForge pipeline
Yakubu Mohammed Abass | COMP 430 Final Project

Usage examples:
  # Default (Van Gogh + watercolor paint + animation GIF)
  python run.py --video jump.mp4

  # Van Gogh only, no paint strokes
  python run.py --video dance.mp4 --no-paint

  # Paint only (no Van Gogh flow strokes in background)
  python run.py --video run.mp4 --no-vangogh --paint-style oil

  # Full control over Van Gogh args
  python run.py --video jump.mp4 \\
      --vg-step 6 --vg-length 35 --vg-curve 0.5 --vg-blur 30

  # Full control over paint args
  python run.py --video jump.mp4 \\
      --paint-style watercolor --paint-density 4 --paint-bleed 0.4

  # Oil paint on dark canvas, no animation
  python run.py --video dance.mp4 \\
      --paint-style oil --no-animate --output my_output/

  # MP4 animation instead of GIF, custom FPS
  python run.py --video run.mp4 --anim-format mp4 --anim-fps 12

  # Tune silhouette extraction
  python run.py --video jump.mp4 \\
      --frame-skip 3 --max-frames 20 --mog2-thresh 45
"""

import argparse
import sys
import os
from pipeline import run_pipeline, DEFAULT_CFG


def parse_args():
    p = argparse.ArgumentParser(
        description="MotionForge: Van Gogh Flow Field + Paint Strokes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input / Output 
    p.add_argument("--video", required=True,
                   help="Path to input video file")
    p.add_argument("--output", default="output",
                   help="Output directory (default: output/)")

    # Extraction 
    ex = p.add_argument_group("Extraction")
    ex.add_argument("--frame-skip", type=int, default=DEFAULT_CFG["frame_skip"],
                    help=f"Sample every N frames (default {DEFAULT_CFG['frame_skip']})")
    ex.add_argument("--max-frames", type=int, default=DEFAULT_CFG["max_silhouettes"],
                    help=f"Max silhouettes to composite (default {DEFAULT_CFG['max_silhouettes']})")
    ex.add_argument("--mog2-thresh", type=int, default=DEFAULT_CFG["mog2_threshold"],
                    help=f"MOG2 sensitivity — lower=more motion (default {DEFAULT_CFG['mog2_threshold']})")
    ex.add_argument("--width", type=int, default=DEFAULT_CFG["output_width"],
                    help=f"Resize width (default {DEFAULT_CFG['output_width']})")

    # Van Gogh 
    vg = p.add_argument_group("Van Gogh Flow Strokes")
    vg.add_argument("--no-vangogh", action="store_true",
                    help="Disable Van Gogh stroke background")
    vg.add_argument("--vg-step", type=int, default=DEFAULT_CFG["vg_stroke_step"],
                    help=f"Grid spacing between strokes px — lower=denser (default {DEFAULT_CFG['vg_stroke_step']})")
    vg.add_argument("--vg-length", type=int, default=DEFAULT_CFG["vg_stroke_length"],
                    help=f"Max stroke length px (default {DEFAULT_CFG['vg_stroke_length']})")
    vg.add_argument("--vg-width", type=int, default=DEFAULT_CFG["vg_stroke_width"],
                    help=f"Stroke thickness px (default {DEFAULT_CFG['vg_stroke_width']})")
    vg.add_argument("--vg-curve", type=float, default=DEFAULT_CFG["vg_stroke_curve"],
                    help=f"Stroke curvature 0=straight 1=very curvy (default {DEFAULT_CFG['vg_stroke_curve']})")
    vg.add_argument("--vg-blur", type=int, default=DEFAULT_CFG["vg_flow_blur"],
                    help=f"Flow field smoothing — higher=more swirl (default {DEFAULT_CFG['vg_flow_blur']})")

    # Paint Strokes
    pt = p.add_argument_group("Paint Strokes")
    pt.add_argument("--no-paint", action="store_true",
                    help="Disable paint stroke silhouette layer")
    pt.add_argument("--paint-style", default=DEFAULT_CFG["paint_style"],
                    choices=["watercolor", "oil"],
                    help=f"watercolor=soft translucent | oil=thick impasto (default {DEFAULT_CFG['paint_style']})")
    pt.add_argument("--paint-density", type=int, default=DEFAULT_CFG["paint_stroke_density"],
                    help=f"Stroke grid inside silhouette — lower=denser (default {DEFAULT_CFG['paint_stroke_density']})")
    pt.add_argument("--paint-max-len", type=int, default=DEFAULT_CFG["paint_stroke_max_len"],
                    help=f"Max paint stroke length px (default {DEFAULT_CFG['paint_stroke_max_len']})")
    pt.add_argument("--paint-bleed", type=float, default=DEFAULT_CFG["paint_bleed_alpha"],
                    help=f"Watercolor bleed opacity 0-1 (default {DEFAULT_CFG['paint_bleed_alpha']})")
    pt.add_argument("--paint-alpha", type=float, default=DEFAULT_CFG["paint_layer_alpha"],
                    help=f"Overall paint layer opacity 0-1 (default {DEFAULT_CFG['paint_layer_alpha']})")

    # Contours 
    ct = p.add_argument_group("Contour Rings")
    ct.add_argument("--contour-rings", type=int, default=DEFAULT_CFG["contour_rings"],
                    help=f"Number of contour rings (default {DEFAULT_CFG['contour_rings']})")
    ct.add_argument("--no-contour-glow", action="store_true",
                    help="Disable glow halo around contour lines")

    # Post-FX
    fx = p.add_argument_group("Post-FX")
    fx.add_argument("--glow", type=int, default=DEFAULT_CFG["glow_strength"],
                    help=f"Bloom glow strength (default {DEFAULT_CFG['glow_strength']})")
    fx.add_argument("--grain", type=int, default=DEFAULT_CFG["grain_intensity"],
                    help=f"Film grain intensity (default {DEFAULT_CFG['grain_intensity']})")
    fx.add_argument("--vignette", type=float, default=DEFAULT_CFG["vignette_strength"],
                    help=f"Vignette strength 0-1 (default {DEFAULT_CFG['vignette_strength']})")

    # Animation 
    an = p.add_argument_group("Animation")
    an.add_argument("--no-animate", action="store_true",
                    help="Skip animation output")
    an.add_argument("--anim-format", default=DEFAULT_CFG["anim_format"],
                    choices=["gif", "mp4"],
                    help=f"Animation format (default {DEFAULT_CFG['anim_format']})")
    an.add_argument("--anim-fps", type=int, default=DEFAULT_CFG["anim_fps"],
                    help=f"Animation frames per second (default {DEFAULT_CFG['anim_fps']})")

    return p.parse_args()


def build_cfg(args):
    cfg = DEFAULT_CFG.copy()

    # Extraction
    cfg["frame_skip"] = args.frame_skip
    cfg["max_silhouettes"] = args.max_frames
    cfg["mog2_threshold"] = args.mog2_thresh
    cfg["output_width"] = args.width

    # Van Gogh
    cfg["vg_enabled"] = not args.no_vangogh
    cfg["vg_stroke_step"] = args.vg_step
    cfg["vg_stroke_length"] = args.vg_length
    cfg["vg_stroke_width"] = args.vg_width
    cfg["vg_stroke_curve"] = args.vg_curve
    cfg["vg_flow_blur"] = args.vg_blur

    # Paint
    cfg["paint_enabled"] = not args.no_paint
    cfg["paint_style"] = args.paint_style
    cfg["paint_stroke_density"] = args.paint_density
    cfg["paint_stroke_max_len"] = args.paint_max_len
    cfg["paint_bleed_alpha"] = args.paint_bleed
    cfg["paint_layer_alpha"] = args.paint_alpha

    # Contours
    cfg["contour_rings"] = args.contour_rings
    cfg["contour_glow"] = not args.no_contour_glow

    # Post-FX
    cfg["glow_strength"] = args.glow
    cfg["grain_intensity"] = args.grain
    cfg["vignette_strength"] = args.vignette

    # Animation
    cfg["animate"] = not args.no_animate
    cfg["anim_format"] = args.anim_format
    cfg["anim_fps"] = args.anim_fps

    return cfg


def main():
    args = parse_args()
    cfg = build_cfg(args)

    print("MOTIONFORGE: One Frame, Many Moments")
    print("Van Gogh Flow Field + Paint Strokes")
    print(f"\n  Video  : {args.video}")
    print(f"  Output : {args.output}/")
    print(f"  VanGogh: {'ON  (step={vg_stroke_step}, len={vg_stroke_length}, curve={vg_stroke_curve})'.format(**cfg) if cfg['vg_enabled'] else 'OFF'}")
    print(f"  Paint  : {'ON  (style=' + cfg['paint_style'] + ', density=' + str(cfg['paint_stroke_density']) + ')' if cfg['paint_enabled'] else 'OFF'}")
    print(f"  Animate: {'ON  (' + cfg['anim_format'].upper() + ' @ ' + str(cfg['anim_fps']) + 'fps)' if cfg['animate'] else 'OFF'}")
    print()

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)

    outputs = run_pipeline(args.video, cfg, output_dir=args.output)

    print("OUTPUT FILES")
    labels = {
        "vangogh" : "Van Gogh stroke canvas",
        "paint" : "Paint stroke layer",
        "contours" : "Contour rings overlay",
        "final" : "  FINAL composite",
        "animation" : "Animation",
    }
    for key, path in outputs.items():
        if path:
            print(f"{labels.get(key, key):<26} -> {path}")


if __name__ == "__main__":
    main()

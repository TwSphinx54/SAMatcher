import argparse
import re
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
import colorsys

METHODS = {"BASE", "OETR", "SAMA"}
METRICS = [
    "AUC@5", "AUC@10", "AUC@20",
    "Acc@5", "Acc@10", "Acc@15", "Acc@20",
    "mAP@5", "mAP@10", "mAP@20",
    "Precision", "MS"
]


def parse_pose_comp(path: str) -> List[Dict]:
    """
    Parse a whitespace-separated pose_comp file.
    """
    rows: List[Dict] = []
    current_pipeline = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("//") or line.startswith("Pipeline"):
                continue

            toks = re.split(r"\s+", line)
            if not toks:
                continue

            if toks[0].upper() in METHODS:
                method = toks[0].upper()
                if current_pipeline is None:
                    continue
                vals_str = toks[1:]
                pipeline = current_pipeline
            else:
                current_pipeline = toks[0]
                pipeline = current_pipeline
                if len(toks) < 2:
                    continue
                method = toks[1].upper()
                vals_str = toks[2:]

            if method not in METHODS:
                continue

            vals = [float(v) for v in vals_str[:len(METRICS)] if re.match(r"^-?\d+(\.\d+)?$", v)]
            if len(vals) != len(METRICS):
                continue

            row = {"Pipeline": pipeline, "Method": method}
            row.update({m: v for m, v in zip(METRICS, vals)})
            rows.append(row)
    return rows


def gaussian_smooth(y: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Apply 1D Gaussian smoothing.
    """
    if kernel_size <= 1:
        return y
    # Build 1D Gaussian kernel
    k = np.arange(kernel_size) - (kernel_size - 1) / 2.0
    g = np.exp(-0.5 * (k / sigma) ** 2)
    g /= g.sum()
    return np.convolve(y, g, mode="same")


def adjust_color_saturation(color, sat_factor=1.0, lightness_shift=0.0):
    """
    Adjust saturation and lightness in HLS space.
    """
    r, g, b, *_ = mcolors.to_rgba(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, min(1.0, s * sat_factor))
    l = max(0.0, min(1.0, l + lightness_shift))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2, 1.0)


def plot_ridgeline(data: List[Dict], out_path: str, smooth: bool = True, scale: float = 2.0):
    """
    Plot per-pipeline ridgelines of method deltas relative to BASE.
    """
    # Group rows by pipeline and method.
    pipelines = []
    by_pipeline: Dict[str, Dict[str, np.ndarray]] = {}

    for row in data:
        p = row["Pipeline"]
        m = row["Method"]
        y = np.array([row[metric] for metric in METRICS], dtype=float)
        if smooth:
            y = gaussian_smooth(y, kernel_size=5, sigma=1.0)
        if p not in by_pipeline:
            by_pipeline[p] = {}
            pipelines.append(p)
        by_pipeline[p][m] = y

    # Keep layout stable even when a method is missing for a pipeline.
    for p in pipelines:
        for m in METHODS:
            if m not in by_pipeline[p]:
                by_pipeline[p][m] = np.zeros(len(METRICS), dtype=float)

    n_metrics = len(METRICS)
    x = np.arange(n_metrics, dtype=float)

    # Compute a global amplitude to control vertical spacing.
    amp_list = []
    for p in pipelines:
        base = by_pipeline[p]["BASE"]
        oetr_delta = by_pipeline[p]["OETR"] - base
        sama_delta = by_pipeline[p]["SAMA"] - base
        amp_list.append(np.max(np.abs(np.vstack([oetr_delta, sama_delta]))))
    global_amp = (max(amp_list) if amp_list else 1.0) * max(1.0, scale)

    n = len(pipelines)
    y_gap = global_amp * 1.2
    y_shift = -global_amp * 0.60

    height = max(2.2, 0.7 * n)
    width = max(8.0, 0.5 * n_metrics)

    fig, ax = plt.subplots(figsize=(width, height))

    # Color strategy: same base color per pipeline, method-specific saturation/lightness.
    pipeline_cmap = plt.get_cmap("Set2")
    oetr_alpha, sama_alpha = 0.78, 0.55

    for i, p in enumerate(pipelines):
        base_color = pipeline_cmap(i % 8)
        sama_color = adjust_color_saturation(base_color, sat_factor=0.65, lightness_shift=0.06)
        oetr_color = adjust_color_saturation(base_color, sat_factor=1.25, lightness_shift=-0.03)
        y0 = i * y_gap + y_shift

        base = by_pipeline[p]["BASE"]

        # Baseline reference.
        ax.plot([x.min(), x.max()], [y0, y0], color=base_color, linestyle="-", linewidth=0.8, alpha=0.35, zorder=0.5)

        # Draw SAMA first, then OETR on top.
        sama_delta = (by_pipeline[p]["SAMA"] - base) * max(1.0, scale)
        y_bottom = np.full_like(sama_delta, y0, dtype=float)
        y_top = y0 + sama_delta
        ax.fill_between(x, y_bottom, y_top, facecolor=sama_color, alpha=sama_alpha, linewidth=0, zorder=1)
        ax.plot(x, y_top, color=sama_color, linewidth=1.1, alpha=0.95, zorder=1.1)

        oetr_delta = (by_pipeline[p]["OETR"] - base) * max(1.0, scale)
        y_top2 = y0 + oetr_delta
        ax.fill_between(x, y_bottom, y_top2, facecolor=oetr_color, alpha=oetr_alpha, linewidth=0, zorder=1.3)
        ax.plot(x, y_top2, color=oetr_color, linewidth=1.2, alpha=0.98, zorder=1.4)

        # Pipeline label on the left.
        ax.text(-0.7, y0, p, ha="right", va="center", fontsize=10, color=base_color)

    # Axes style
    ax.set_xlim(-1.0, n_metrics - 0.1)
    ax.set_ylim(-global_amp * 0.6 + y_shift, (n - 1) * y_gap + global_amp * 1.1 + y_shift)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=32, ha="right", fontsize=9)

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Legend uses a sample base color with two method variants.
    sample_base = pipeline_cmap(1)
    sample_sama = adjust_color_saturation(sample_base, sat_factor=0.65, lightness_shift=0.06)
    sample_oetr = adjust_color_saturation(sample_base, sat_factor=1.25, lightness_shift=-0.03)
    legend_patches = [
        Patch(facecolor=sample_oetr, edgecolor="none", alpha=oetr_alpha, label="OETR"),
        Patch(facecolor=sample_sama, edgecolor="none", alpha=sama_alpha, label="SAMatcher"),
    ]

    # Title on axes; legend centered at figure level.
    ax.set_title("Pose Comparison Ridgeline: Delta vs BASE", fontsize=12, pad=8)

    # Reserve top space for title + legend.
    fig.subplots_adjust(top=0.88)
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        ncol=2,
        columnspacing=1.0,
        handlelength=1.4,
        handletextpad=0.5,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot ridgeline chart from pose_comp.csv")
    parser.add_argument("--csv", type=str, default="./pose_comp.csv")
    parser.add_argument("--out", type=str, default="./outputs/pose_comp_ridgeline.png")
    parser.add_argument("--no-smooth", action="store_true", help="Disable Gaussian smoothing")
    parser.add_argument("--scale", type=float, default=2.0, help="Magnification factor for deltas")
    args = parser.parse_args()

    data = parse_pose_comp(args.csv)
    if not data:
        raise SystemExit("No data parsed. Check CSV path/format.")
    plot_ridgeline(data, args.out, smooth=not args.no_smooth, scale=args.scale)
    print(f"Saved ridgeline to: {args.out}")


if __name__ == "__main__":
    main()

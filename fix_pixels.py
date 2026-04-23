#!/usr/bin/env python
"""
fix_pixels.py - Un-wobble an AI-generated pixel-art image.

Pipeline:
  [0] Load image.
  [1] Gradient magnitude — pixel borders light up.
  [2] Project gradient onto X and Y axes — peaks mark grid lines.
  [3] FFT of each projection — strongest periodic frequency is the pixel pitch P.
  [4] Phase search — find where the first grid line starts.
  [5] Per-line local snap — for each predicted line, slide to the nearest
      actual gradient peak (this is what fixes the wobble).
  [6] Median color per cell.
  [7] Rebuild image at source resolution (for A/B comparison).
  [8] Clean nearest-neighbor upscale of the final pixel grid.

Every step dumps an artifact to .out/ so you can see the logic work.
"""
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter, gaussian_filter1d

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "crow.jpg"
MERGE_RADIUS = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0  # Lab distance threshold
OUT_DIR = ".out"
os.makedirs(OUT_DIR, exist_ok=True)


def save(name, img):
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, img)
    print(f"  -> {path}")


def save_plot(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  -> {path}")


# ---- [0] load ---------------------------------------------------------------
print("[0] load", IN_PATH)
bgr = cv2.imread(IN_PATH, cv2.IMREAD_COLOR)
if bgr is None:
    sys.exit(f"could not read {IN_PATH}")
H, W = bgr.shape[:2]
print(f"    size = {W} x {H}")
save("00_original.png", bgr)


# ---- [1] gradient magnitude -------------------------------------------------
print("[1] gradient magnitude")
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))  # strong at vertical edges
gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))  # strong at horizontal edges
grad = gx + gy
save("01_gradient.png", np.clip(grad / grad.max() * 255, 0, 255).astype(np.uint8))


# ---- [2] axis projections ---------------------------------------------------
# A vertical pixel border is a column of high |dI/dx|, so summing gx down the
# columns gives a 1D signal whose peaks are vertical-border x-positions.
print("[2] axis projections")
col_sig = gx.sum(axis=0)  # length W, peaks at vertical borders
row_sig = gy.sum(axis=1)  # length H, peaks at horizontal borders

fig, ax = plt.subplots(2, 1, figsize=(12, 5))
ax[0].plot(col_sig)
ax[0].set_title("gx summed down columns -> peaks at vertical pixel borders")
ax[0].set_xlabel("x")
ax[1].plot(row_sig)
ax[1].set_title("gy summed across rows -> peaks at horizontal pixel borders")
ax[1].set_xlabel("y")
save_plot("02_projections.png")


# ---- [3] pitch via autocorrelation -----------------------------------------
# FFT alone is fooled by harmonics: a square-edged signal has strong content
# at 2f and 3f, so the FFT can lock onto a half-period. Autocorrelation does
# not have this problem -- the first strong peak after lag 0 is the true
# fundamental period, regardless of harmonic content.
def detect_pitch(sig, min_P=4, max_P=80):
    """Return pixel pitch (in source pixels) using autocorrelation."""
    s = sig - sig.mean()
    # Full autocorrelation, then keep only non-negative lags.
    ac_full = np.correlate(s, s, mode="full")
    ac = ac_full[len(ac_full) // 2 :]
    # Normalize so the bias-from-overlap doesn't favor short lags.
    n = len(ac)
    ac = ac / np.arange(n, 0, -1)
    # Look for peaks in the candidate-period window.
    window = ac[min_P : max_P + 1]
    peaks, props = find_peaks(window, prominence=window.max() * 0.05)
    if len(peaks) == 0:
        # Fallback: just argmax in the window.
        k = int(np.argmax(window))
    else:
        # First (smallest-lag) prominent peak = the fundamental period.
        k = int(peaks[0])
    P_int = min_P + k
    # Sub-pixel refinement around the chosen peak.
    if 0 < P_int < len(ac) - 1:
        y1, y2, y3 = ac[P_int - 1], ac[P_int], ac[P_int + 1]
        denom = y1 - 2 * y2 + y3
        delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    P = P_int + delta
    return P, ac, P_int


print("[3] pitch via autocorrelation")
P_x, ac_x, k_x = detect_pitch(col_sig)
P_y, ac_y, k_y = detect_pitch(row_sig)
print(f"    autocorr says P_x = {P_x:.3f}   P_y = {P_y:.3f}")
P_init = (P_x + P_y) / 2.0


# Autocorrelation can still lock onto a sub-pixel period when each AI tile
# has internal anti-aliased ramps. Validate using an ANOVA-style metric:
#
#   F = var(cell mean colours) / mean(within-cell variance)
#
# At the *true* P, neighbouring cells are different colours (high numerator)
# while each cell is internally near-uniform (low denominator). At P/2, both
# halves of a true pixel share a colour -> numerator collapses and F drops.
def f_score(img, P):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    P_int = int(round(P))
    if P_int < 2 or P_int > min(H, W) // 4:
        return -np.inf, (0, 0)
    mean = cv2.boxFilter(g, -1, (P_int, P_int), normalize=True)
    mean_sq = cv2.boxFilter(g * g, -1, (P_int, P_int), normalize=True)
    var = np.maximum(mean_sq - mean * mean, 0)
    half = P_int // 2
    best, best_off = -np.inf, (0, 0)
    for oy in range(P_int):
        yc = np.arange(oy + half, H - half, P_int)
        if len(yc) < 2:
            continue
        for ox in range(P_int):
            xc = np.arange(ox + half, W - half, P_int)
            if len(xc) < 2:
                continue
            cm = mean[np.ix_(yc, xc)]
            cv_ = var[np.ix_(yc, xc)]
            within = cv_.mean() + 1e-6
            between = cm.var()
            F = between / within
            if F > best:
                best, best_off = F, (ox, oy)
    return best, best_off


print("[3b] validate against P, 2P, 3P via F-statistic")
candidates = [P_init, 2 * P_init, 3 * P_init]
scores = []
for cand in candidates:
    F, off = f_score(bgr, cand)
    scores.append((cand, F, off))
    print(f"    P={cand:6.2f}  F={F:8.3f}  best phase={off}")
P, F_chosen, _ = max(scores, key=lambda t: t[1])
print(f"    chose P = {P:.3f}  (F = {F_chosen:.3f})")

# Sweep for debug plot.
sweep_Ps = list(range(4, 50))
sweep_scores = [f_score(bgr, p)[0] for p in sweep_Ps]

# Plot autocorrelations + the chosen P, plus the cell-uniformity sweep so
# the validation step is visible.
fig, ax = plt.subplots(3, 1, figsize=(12, 8))
ax[0].plot(np.arange(len(ac_x)), ac_x)
ax[0].axvline(P, color="g", alpha=0.7, label=f"chosen P={P:.2f}")
ax[0].axvline(P_x, color="r", ls="--", alpha=0.5, label=f"autocorr P_x={P_x:.2f}")
ax[0].set_xlim(0, 100)
ax[0].set_title("Autocorrelation of column projection")
ax[0].set_xlabel("lag (px)")
ax[0].legend()
ax[1].plot(np.arange(len(ac_y)), ac_y)
ax[1].axvline(P, color="g", alpha=0.7, label=f"chosen P={P:.2f}")
ax[1].axvline(P_y, color="r", ls="--", alpha=0.5, label=f"autocorr P_y={P_y:.2f}")
ax[1].set_xlim(0, 100)
ax[1].set_title("Autocorrelation of row projection")
ax[1].set_xlabel("lag (px)")
ax[1].legend()
ax[2].plot(sweep_Ps, sweep_scores, "o-")
ax[2].axvline(P, color="g", alpha=0.7, label=f"chosen P={P:.2f}")
ax[2].set_title("F-statistic (higher = more pixel-arty) vs candidate P")
ax[2].set_xlabel("candidate pitch P (source px)")
ax[2].set_ylabel("between-cell var / within-cell var")
ax[2].legend()
save_plot("03_autocorr.png")


# ---- [4] global phase -------------------------------------------------------
# Slide a Dirac comb of period P across sig and pick the offset with max response.
def detect_phase(sig, P):
    best, best_val = 0.0, -np.inf
    for off in np.linspace(0, P, int(np.ceil(P)) * 4, endpoint=False):
        idx = np.round(np.arange(off, len(sig), P)).astype(int)
        idx = idx[idx < len(sig)]
        v = sig[idx].sum()
        if v > best_val:
            best_val, best = v, off
    return best


x0 = detect_phase(col_sig, P)
y0 = detect_phase(row_sig, P)
print(f"[4] global phase: x0 = {x0:.3f}   y0 = {y0:.3f}")

xs_uniform = np.arange(x0, W + 0.5, P)
ys_uniform = np.arange(y0, H + 0.5, P)


def draw_grid(bgr, xs, ys, color=(0, 255, 0), thickness=1):
    out = bgr.copy()
    for x in xs:
        xi = int(round(x))
        if 0 <= xi < out.shape[1]:
            cv2.line(out, (xi, 0), (xi, out.shape[0] - 1), color, thickness)
    for y in ys:
        yi = int(round(y))
        if 0 <= yi < out.shape[0]:
            cv2.line(out, (0, yi), (out.shape[1] - 1, yi), color, thickness)
    return out


save("04_grid_uniform.png", draw_grid(bgr, xs_uniform, ys_uniform, (0, 255, 0)))


# ---- [5] local snap (wobble correction) ------------------------------------
# A uniform grid will drift in a wobbly image. For each predicted line, search
# a small window for the true gradient peak. Two safeguards:
#  * search radius < P/2, so a line can't snap onto its neighbour's peak;
#  * after snapping, enforce monotonicity & a minimum gap so lines never cross.
def snap_lines(sig, predicted, search=None, min_gap_frac=0.5):
    if search is None:
        # Half a pixel of search radius is plenty for typical AI-art wobble
        # and keeps us safely inside the neighbour's territory.
        search = max(2, int(round(P / 4)))
    snapped = []
    for p in predicted:
        lo = max(0, int(round(p - search)))
        hi = min(len(sig), int(round(p + search + 1)))
        if hi - lo <= 1:
            snapped.append(p)
            continue
        local = sig[lo:hi]
        k = int(np.argmax(local))
        if 0 < k < len(local) - 1:
            y1, y2, y3 = local[k - 1], local[k], local[k + 1]
            denom = y1 - 2 * y2 + y3
            delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
            snapped.append(lo + k + delta)
        else:
            snapped.append(lo + k)
    snapped = np.array(snapped, dtype=float)
    # Enforce monotonicity with a minimum gap of P*min_gap_frac.
    min_gap = P * min_gap_frac
    for i in range(1, len(snapped)):
        if snapped[i] < snapped[i - 1] + min_gap:
            snapped[i] = snapped[i - 1] + min_gap
    return snapped


print("[5] local snap")
xs_raw = snap_lines(col_sig, xs_uniform)
ys_raw = snap_lines(row_sig, ys_uniform)
dx_raw = xs_raw - xs_uniform[: len(xs_raw)]
dy_raw = ys_raw - ys_uniform[: len(ys_raw)]
print(f"    raw x displacement: std={dx_raw.std():.2f} max|d|={np.abs(dx_raw).max():.2f}")
print(f"    raw y displacement: std={dy_raw.std():.2f} max|d|={np.abs(dy_raw).max():.2f}")

# True wobble is locally coherent (neighbouring grid lines drift together).
# Independent per-line snapping is noisy: a single line can latch onto an
# internal feature edge (a feather, a flower) instead of the real cell border.
# Median filter kills those outliers; Gaussian smooth recovers the underlying
# slow-varying wobble curve.
def smooth_displacement(d, median_size=5, gauss_sigma=2.5):
    d_med = median_filter(d, size=median_size)
    d_smooth = gaussian_filter1d(d_med, sigma=gauss_sigma)
    return d_smooth


dx = smooth_displacement(dx_raw)
dy = smooth_displacement(dy_raw)
xs = xs_uniform[: len(dx)] + dx
ys = ys_uniform[: len(dy)] + dy
print(f"    smoothed x displacement: std={dx.std():.2f} max|d|={np.abs(dx).max():.2f}")
print(f"    smoothed y displacement: std={dy.std():.2f} max|d|={np.abs(dy).max():.2f}")


# Sanity check: a real wobble correction should never produce a "pixel" cell
# that is 1.5x larger or 0.5x smaller than its neighbours. If it does, the
# snap latched onto a spurious feature. Bump up smoothing until clean, then
# warn if still bad (likely means our pitch P is wrong).
def gap_ratio_range(arr):
    g = np.diff(arr)
    m = np.median(g)
    return g.min() / m, g.max() / m


def enforce_gap_sanity(d_raw, predicted, max_ratio=1.5, min_ratio=0.5, max_sigma=20.0):
    """Re-smooth with growing sigma until all gaps lie in [min_ratio, max_ratio] x median."""
    sigma = 2.5
    while sigma <= max_sigma:
        d = smooth_displacement(d_raw, gauss_sigma=sigma)
        lines = predicted[: len(d)] + d
        lo, hi = gap_ratio_range(lines)
        if lo >= min_ratio and hi <= max_ratio:
            return d, sigma, (lo, hi), True
        sigma *= 1.5
    return d, sigma, (lo, hi), False


dx, sigma_x, (lox, hix), ok_x = enforce_gap_sanity(dx_raw, xs_uniform)
dy, sigma_y, (loy, hiy), ok_y = enforce_gap_sanity(dy_raw, ys_uniform)
xs = xs_uniform[: len(dx)] + dx
ys = ys_uniform[: len(dy)] + dy
print(f"    gap sanity: x ratio range=[{lox:.2f}, {hix:.2f}] sigma={sigma_x:.1f} ok={ok_x}")
print(f"    gap sanity: y ratio range=[{loy:.2f}, {hiy:.2f}] sigma={sigma_y:.1f} ok={ok_y}")
if not (ok_x and ok_y):
    print("    !! WARNING: gap ratios out of [0.5, 1.5] even after max smoothing.")
    print("    !! This usually means the detected pitch P is wrong (try forcing 2*P).")
print(f"    col gaps: min={np.diff(xs).min():.2f} max={np.diff(xs).max():.2f} median={np.median(np.diff(xs)):.2f}")
print(f"    row gaps: min={np.diff(ys).min():.2f} max={np.diff(ys).max():.2f} median={np.median(np.diff(ys)):.2f}")

save("05_grid_snapped.png", draw_grid(bgr, xs, ys, (0, 0, 255)))

# Overlay green=uniform vs red=snapped vs yellow=raw-snap for visual diff.
both = draw_grid(bgr, xs_uniform, ys_uniform, (0, 200, 0))
both = draw_grid(both, xs_raw, ys_raw, (0, 200, 200))
both = draw_grid(both, xs, ys, (0, 0, 255))
save("05b_grid_both.png", both)

# Plot raw vs smoothed displacement so the smoothing effect is visible.
fig, ax = plt.subplots(2, 1, figsize=(12, 5))
ax[0].plot(dx_raw, color="tab:orange", alpha=0.6, label="raw snap")
ax[0].plot(dx, color="tab:blue", lw=2, label="smoothed")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_title("vertical line displacement vs uniform grid")
ax[0].set_xlabel("column index")
ax[0].set_ylabel("pixels")
ax[0].legend()
ax[1].plot(dy_raw, color="tab:orange", alpha=0.6, label="raw snap")
ax[1].plot(dy, color="tab:blue", lw=2, label="smoothed")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_title("horizontal line displacement vs uniform grid")
ax[1].set_xlabel("row index")
ax[1].set_ylabel("pixels")
ax[1].legend()
save_plot("05c_displacement.png")


# ---- [6] median color per cell ---------------------------------------------
print("[6] median color per cell")
xs_full = np.unique(np.clip(np.concatenate([[0.0], xs, [W]]), 0, W))
ys_full = np.unique(np.clip(np.concatenate([[0.0], ys, [H]]), 0, H))
nx = len(xs_full) - 1
ny = len(ys_full) - 1
print(f"    output grid: {nx} cols x {ny} rows")

pixels = np.zeros((ny, nx, 3), dtype=np.uint8)
for i in range(ny):
    y0c, y1c = int(round(ys_full[i])), int(round(ys_full[i + 1]))
    if y1c <= y0c:
        continue
    for j in range(nx):
        x0c, x1c = int(round(xs_full[j])), int(round(xs_full[j + 1]))
        if x1c <= x0c:
            continue
        block = bgr[y0c:y1c, x0c:x1c].reshape(-1, 3)
        # Median per channel is robust to anti-aliasing bleed at cell edges.
        pixels[i, j] = np.median(block, axis=0).astype(np.uint8)

save("06_pixels.png", pixels)


# ---- [6b] conservative palette merge ---------------------------------------
# Forced k-means merges rare accent colours into larger buckets. Here we only
# merge colours that are already close to the same local anchor. This avoids
# "chain merging" where A~B and B~C would incorrectly collapse a long colour
# ramp even when A and C are not actually close.
print(f"[6b] conservative palette merge within Lab radius {MERGE_RADIUS:.1f}")
cell_bgr = pixels.reshape(-1, 3)
unique_bgr, inverse, counts_unique = np.unique(cell_bgr, axis=0, return_inverse=True, return_counts=True)
unique_lab = cv2.cvtColor(unique_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
print(f"    unique cell colors before merge: {len(unique_bgr)}")

seed_order = np.argsort(-counts_unique)
labels_unique = np.full(len(unique_bgr), -1, dtype=np.int32)
palette_lab_list = []
palette_counts_list = []
radius2 = MERGE_RADIUS * MERGE_RADIUS

for seed_idx in seed_order:
    if labels_unique[seed_idx] != -1:
        continue
    seed_lab = unique_lab[seed_idx]
    unassigned = labels_unique == -1
    d = unique_lab[unassigned] - seed_lab
    close_mask = np.sum(d * d, axis=1) <= radius2
    member_indices = np.flatnonzero(unassigned)[close_mask]
    cluster_idx = len(palette_lab_list)
    labels_unique[member_indices] = cluster_idx
    w = counts_unique[member_indices].astype(np.float32)
    palette_lab_list.append((unique_lab[member_indices] * w[:, None]).sum(axis=0) / w.sum())
    palette_counts_list.append(int(w.sum()))

palette_lab = np.array(palette_lab_list, dtype=np.float32)
palette_counts = np.array(palette_counts_list, dtype=np.int32)
K = len(palette_lab)
print(f"    unique palette colors after merge: {K}")

palette_bgr = cv2.cvtColor(
    np.clip(np.round(palette_lab), 0, 255).astype(np.uint8).reshape(-1, 1, 3),
    cv2.COLOR_LAB2BGR,
).reshape(-1, 3)

order = np.argsort(-palette_counts)
remap = np.zeros(K, dtype=np.int32)
remap[order] = np.arange(K)
labels_unique = remap[labels_unique]
palette_bgr = palette_bgr[order]
palette_counts = palette_counts[order]

pixel_labels = labels_unique[inverse]
pixels_q = palette_bgr[pixel_labels].reshape(ny, nx, 3).astype(np.uint8)
save("06b_pixels_quantized.png", pixels_q)

swatch_h = 32
swatch_w = 48
palette_strip = np.zeros((swatch_h, swatch_w * K, 3), dtype=np.uint8)
for i, color in enumerate(palette_bgr):
    palette_strip[:, i * swatch_w : (i + 1) * swatch_w] = color
save("06c_palette.png", palette_strip)

fig, ax = plt.subplots(figsize=(max(8, K * 0.28), 2.5))
ax.bar(np.arange(K), palette_counts, color=palette_bgr[:, ::-1] / 255.0)
ax.set_title(f"Palette usage after conservative Lab merge (radius={MERGE_RADIUS:.1f}, colors={K})")
ax.set_xlabel("palette index (sorted by use)")
ax.set_ylabel("cell count")
save_plot("06d_palette_usage.png")


# ---- [7] rebuild at source resolution --------------------------------------
# Paint each source-space cell with its chosen solid color for A/B compare.
print("[7] rebuild at source resolution")
rebuilt = np.zeros_like(bgr)
for i in range(ny):
    y0c, y1c = int(round(ys_full[i])), int(round(ys_full[i + 1]))
    for j in range(nx):
        x0c, x1c = int(round(xs_full[j])), int(round(xs_full[j + 1]))
        rebuilt[y0c:y1c, x0c:x1c] = pixels[i, j]
save("07_rebuilt_source_res.png", rebuilt)

rebuilt_q = np.zeros_like(bgr)
for i in range(ny):
    y0c, y1c = int(round(ys_full[i])), int(round(ys_full[i + 1]))
    for j in range(nx):
        x0c, x1c = int(round(xs_full[j])), int(round(xs_full[j + 1]))
        rebuilt_q[y0c:y1c, x0c:x1c] = pixels_q[i, j]
save("07b_rebuilt_quantized.png", rebuilt_q)


# ---- [8] clean nearest-neighbor upscale ------------------------------------
print("[8] clean upscale")
scale = int(round(P))
clean_up = cv2.resize(pixels, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)
save("08_clean_upscaled.png", clean_up)

clean_up_q = cv2.resize(pixels_q, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)
save("08b_clean_upscaled_quantized.png", clean_up_q)

print("\ndone. inspect .out/ to follow the pipeline.")

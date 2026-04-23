# pixeled-image

Recover a clean pixel grid from an AI-generated "almost pixel art" image.

The main script, `fix_pixels.py`, detects the underlying pixel pitch, estimates the grid phase, snaps wobbling grid lines back onto strong image edges, extracts a representative color per cell, and writes intermediate debug artifacts for each stage.

## What it does

- Detects vertical and horizontal edge energy
- Estimates the source pixel pitch from autocorrelation
- Finds a global grid phase
- Locally snaps predicted grid lines to nearby edge peaks
- Extracts median cell colors
- Optionally merges very similar colors in Lab space
- Rebuilds both source-resolution and clean nearest-neighbor outputs

## Files

- `fix_pixels.py`: main pipeline
- `crow.jpg`: sample input image
- `example-out/`: committed example outputs from a successful run
- `.out/`: ignored local output directory used when you run the script yourself

## Requirements

Python packages used by `fix_pixels.py`:

- `opencv-python`
- `numpy`
- `matplotlib`
- `scipy`

`package.json` is present, but the image reconstruction pipeline itself is Python-based.

## Usage

```bash
python fix_pixels.py
python fix_pixels.py crow.jpg
python fix_pixels.py crow.jpg 3.0
```

Arguments:

- `argv[1]`: input image path, default `crow.jpg`
- `argv[2]`: Lab merge radius for conservative palette merging, default `3.0`

The script writes all generated artifacts into `.out/`.

## Example output

Input:

![Original input](example-out/00_original.png)

Grid detection and correction:

![Uniform vs snapped grid](example-out/05b_grid_both.png)

Projection diagnostics:

![Axis projections](example-out/02_projections.png)

Pitch search:

![Autocorrelation and F-statistic](example-out/03_autocorr.png)

Per-line displacement before and after smoothing:

![Displacement plots](example-out/05c_displacement.png)

Extracted pixel grid:

![Quantized pixel grid](example-out/06b_pixels_quantized.png)

Palette usage:

![Palette usage graph](example-out/06d_palette_usage.png)

Final clean upscale:

![Final upscaled result](example-out/08b_clean_upscaled_quantized.png)

The checked-in `example-out/` directory also contains:

- gradient and projection diagnostics
- detected grid overlays
- displacement plots
- raw and quantized pixel grids
- palette visualization
- rebuilt source-resolution images
- clean upscaled outputs

## Branch

This project is published on the `pixeled-image` branch of:

<https://github.com/Dimava/misc/tree/pixeled-image>

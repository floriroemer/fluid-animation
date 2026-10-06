# fluid-animation

A slow, hypnotic fluid animation — the look of milk swirling in coffee, or two
colored liquids you stopped stirring a moment too early. Colors stretch into
ribbons, fold around each other, and creep toward one another, but they
**never fully blend** into a single flat color.

Pure Python. No GPU, no compiled extensions, no extra dependencies beyond
numpy and matplotlib.

## Run it

```bash
python fluid.py
```

A window opens and the animation starts. That's it.

### Controls

| Input | Effect |
|---|---|
| **Click** anywhere | Splat a random color into the pool with a kick of momentum |
| **Space** | Pause / resume |
| **Q** or **Esc** | Quit |

### Options

```bash
python fluid.py --grid 192        # finer detail (slower)
python fluid.py --grid 96         # coarser, faster
python fluid.py --exposure 2.2    # brighter colors
python fluid.py --fps 60          # smoother motion
python fluid.py --save out.mp4    # render to video instead of a window
python fluid.py --selftest        # headless check, no window
```

## How it works

This is **Jos Stam's "Stable Fluids"** (SIGGRAPH 1999) — the same technique
behind a lot of film and game fluid effects, and a genuinely good match for
this aesthetic.

The simulation keeps two things on a square grid:

- a **velocity field** `(u, v)` — the invisible motion of the liquid
- a set of **dye layers**, one per color

Each frame does four things:

1. **Advect** — every cell traces backwards along the flow to find where its
   contents came from, and takes that value. This is what stretches dye into
   long, smooth ribbons.
2. **Diffuse** — velocity only, and very slightly. Dye diffusion is set to
   **zero**, which is the key trick: with no diffusion, colors can be stretched
   and folded forever without ever actually mixing into each other.
3. **Project** — solve a Poisson equation for pressure so the velocity field is
   divergence-free. This is what makes it look like an *incompressible* liquid
   rather than a gas.
4. **Fade** — a slow uniform decay on all dye. Without it, injected dye
   accumulates until the whole pool washes out to white. Because every layer
   fades at the same rate, the color *ratios* — which is all the eye actually
   reads — are preserved.

The stirring comes from five invisible "paddles" that wander along slow sine
paths, each pushing the fluid tangentially (a smooth vortex) and dribbling in
its own color.

### Tuning the look

The interesting knobs are all near the top of `fluid.py`:

| Knob | Effect |
|---|---|
| `PALETTE` | The colors. Any list of RGB triples in 0..1. |
| `BACKGROUND` | The dark pool color. |
| `FORCE_STRENGTH` / `DAMPING` | Flow speed. Equilibrium speed ≈ `FORCE_STRENGTH / DAMPING`. |
| `DYE_RATE` | How fast fresh color enters. |
| `dye_diffusion` | **Leave at 0** for the never-fully-mixing effect. Raise it slightly for softer, blurrier colors. |
| `dye_dissipation` | How fast colors fade out. |
| `STIRRERS` | The paddles: positions, paths, radii, spin direction, colors. |

### A note on `dt` and CFL

`dt` is deliberately small (0.015). The advection step moves each cell
`|u| * dt * grid` cells per step — the **CFL number**. If that exceeds ~1, the
advection skips over structure and the image turns to mush. `--selftest` prints
the CFL number so you can check before trusting a result. If you raise
`FORCE_STRENGTH`, expect to lower `dt` to compensate.

## Performance

Roughly, on a typical laptop CPU (single-threaded numpy):

| Grid | ms/frame | FPS |
|---|---|---|
| 96² | ~16 | ~60 |
| 144² | ~30 | ~33 |
| 192² | ~55 | ~18 |

The window runs two simulation substeps per displayed frame, so a `--grid 144`
window at `--fps 30` is doing ~60 solver steps per second.

## Requirements

- Python 3.8+
- numpy
- matplotlib

```bash
pip install -r requirements.txt
```

## License

MIT

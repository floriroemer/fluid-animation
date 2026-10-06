#!/usr/bin/env python3
"""
fluid.py -- slow, hypnotic "coffee & cream" fluid animation.

Colors are stirred into a dark pool by a handful of invisible paddles.
They stretch into ribbons, fold around each other, and creep toward one
another -- but with dye diffusion switched off they never actually blend
into a single flat color. Exactly like milk in coffee that you stop
stirring a moment too early.

Physics: Jos Stam's "Stable Fluids" (SIGGRAPH 1999).
    * semi-Lagrangian advection  -> unconditional stability
    * Gauss-Seidel pressure solve -> makes the velocity field divergence free
    * diffusion = 0 for dye       -> colors stay distinct forever

Run it:
    python fluid.py                 # default, opens a window
    python fluid.py --grid 192      # finer detail, slower
    python fluid.py --save out.mp4  # render to video instead of a window
    python fluid.py --selftest      # headless smoke test, no window
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Palette -- muted, dusty tones on a near-black pool.
# Swap these out for your own; the sim will happily use any list of RGB triples.
# ---------------------------------------------------------------------------
PALETTE = [
    (0.98, 0.76, 0.52),   # warm apricot
    (0.93, 0.42, 0.44),   # soft coral
    (0.58, 0.36, 0.64),   # dusty violet
    (0.30, 0.55, 0.70),   # muted steel blue
    (0.45, 0.74, 0.62),   # sage green
    (0.96, 0.91, 0.80),   # cream
]

BACKGROUND = (0.05, 0.05, 0.07)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------
class StableFluid:
    """A square, incompressible 2D fluid on a uniform grid."""

    def __init__(
        self,
        n: int = 144,
        n_dye: int = 6,
        viscosity: float = 2e-5,
        dye_diffusion: float = 0.0,
        dye_dissipation: float = 0.55,
        iterations: int = 16,
        dt: float = 0.015,
    ) -> None:
        self.n = n
        self.n_dye = n_dye
        self.viscosity = viscosity
        self.dye_diffusion = dye_diffusion
        self.dye_dissipation = dye_dissipation
        self.iterations = iterations
        self.dt = dt

        size = n + 2                      # +2 for the boundary ring
        self.u = np.zeros((size, size))   # x velocity
        self.v = np.zeros((size, size))   # y velocity
        self.u0 = np.zeros((size, size))
        self.v0 = np.zeros((size, size))
        self.p = np.zeros((size, size))   # pressure
        self.div = np.zeros((size, size)) # divergence

        self.dye = [np.zeros((size, size)) for _ in range(n_dye)]
        self.dye0 = [np.zeros((size, size)) for _ in range(n_dye)]

        # Cached index grids, reused by the advection step every frame.
        idx = np.arange(1, n + 1, dtype=np.float64)
        self._jj, self._ii = np.meshgrid(idx, idx, indexing="ij")

        # Normalised 0..1 coordinates of every interior cell (for forcing).
        self.gx = (self._ii - 0.5) / n
        self.gy = (self._jj - 0.5) / n

    # -- boundary conditions -------------------------------------------------
    def _set_bnd(self, b: int, x: np.ndarray) -> None:
        """b=0 scalar (dye/pressure), b=1 x-velocity, b=2 y-velocity.

        A velocity component flips sign across a wall (no-slip-ish), a scalar
        just mirrors.
        """
        x[0, 1:-1] = -x[1, 1:-1] if b == 2 else x[1, 1:-1]
        x[-1, 1:-1] = -x[-2, 1:-1] if b == 2 else x[-2, 1:-1]
        x[1:-1, 0] = -x[1:-1, 1] if b == 1 else x[1:-1, 1]
        x[1:-1, -1] = -x[1:-1, -2] if b == 1 else x[1:-1, -2]
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

    # -- linear solve --------------------------------------------------------
    def _lin_solve(self, x: np.ndarray, x0: np.ndarray, a: float, c: float) -> None:
        """Gauss-Seidel relaxation of  (I - a*Laplacian) x = x0."""
        for _ in range(self.iterations):
            x[1:-1, 1:-1] = (
                x0[1:-1, 1:-1]
                + a
                * (
                    x[:-2, 1:-1]
                    + x[2:, 1:-1]
                    + x[1:-1, :-2]
                    + x[1:-1, 2:]
                )
            ) / c

    # -- advection -----------------------------------------------------------
    def _advect(self, d: np.ndarray, d0: np.ndarray,
                u: np.ndarray, v: np.ndarray, dt: float) -> None:
        """Semi-Lagrangian: trace each cell backwards along the flow."""
        n = self.n
        dt0 = dt * n

        x = self._ii - dt0 * u[1:-1, 1:-1]
        y = self._jj - dt0 * v[1:-1, 1:-1]
        np.clip(x, 0.5, n + 0.5, out=x)
        np.clip(y, 0.5, n + 0.5, out=y)

        i0 = x.astype(np.int32)
        j0 = y.astype(np.int32)
        i1 = i0 + 1
        j1 = j0 + 1

        s1 = x - i0
        s0 = 1.0 - s1
        t1 = y - j0
        t0 = 1.0 - t1

        d[1:-1, 1:-1] = (
            s0 * (t0 * d0[j0, i0] + t1 * d0[j1, i0])
            + s1 * (t0 * d0[j0, i1] + t1 * d0[j1, i1])
        )

    # -- projection ----------------------------------------------------------
    def _project(self) -> None:
        """Remove divergence: solve a Poisson equation for pressure."""
        n = self.n
        h = 1.0 / n
        u, v, p, div = self.u, self.v, self.p, self.div

        div[1:-1, 1:-1] = -0.5 * h * (
            u[1:-1, 2:] - u[1:-1, :-2] + v[2:, 1:-1] - v[:-2, 1:-1]
        )
        p.fill(0.0)
        self._set_bnd(0, div)
        self._set_bnd(0, p)
        self._lin_solve(p, div, 1.0, 4.0)

        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        self._set_bnd(1, u)
        self._set_bnd(2, v)

    # -- one full step -------------------------------------------------------
    def velocity_step(self) -> None:
        dt, n, visc = self.dt, self.n, self.viscosity

        # Velocity drag. Without this the stirrers pump energy in forever and
        # the flow accelerates until advection is hopelessly under-resolved.
        # With it, the flow relaxes to a steady speed of ~FORCE_STRENGTH/DAMPING.
        self.u *= float(np.exp(-DAMPING * dt))
        self.v *= float(np.exp(-DAMPING * dt))

        self._set_bnd(1, self.u)
        self._set_bnd(2, self.v)

        if visc > 0.0:
            self.u0[:] = self.u
            self.v0[:] = self.v
            a = dt * visc * n * n
            self._lin_solve(self.u, self.u0, a, 1.0 + 4.0 * a)
            self._lin_solve(self.v, self.v0, a, 1.0 + 4.0 * a)
            self._set_bnd(1, self.u)
            self._set_bnd(2, self.v)

        self._project()

        self.u0[:] = self.u
        self.v0[:] = self.v
        self._advect(self.u, self.u0, self.u0, self.v0, dt)
        self._advect(self.v, self.v0, self.u0, self.v0, dt)
        self._project()

    def dye_step(self) -> None:
        dt, n, diff = self.dt, self.n, self.dye_diffusion
        # Slow fade so injected dye reaches a steady state instead of piling
        # up until the whole pool is washed out. Colour *ratios* are what the
        # eye reads, so fading everything together keeps the structure intact.
        decay = float(np.exp(-self.dye_dissipation * dt))
        for k in range(self.n_dye):
            d = self.dye[k]
            d0 = self.dye0[k]

            if diff > 0.0:
                d0[:] = d
                a = dt * diff * n * n
                self._lin_solve(d, d0, a, 1.0 + 4.0 * a)
                self._set_bnd(0, d)

            d0[:] = d
            self._advect(d, d0, self.u, self.v, dt)
            self._set_bnd(0, d)

            d *= decay

    # -- user-facing helpers -------------------------------------------------
    def add_force(self, gx: float, gy: float, fx: float, fy: float,
                  radius: float) -> None:
        """Push the fluid at normalised position (gx, gy)."""
        dx = self.gx - gx
        dy = self.gy - gy
        falloff = np.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius))
        self.u[1:-1, 1:-1] += fx * falloff
        self.v[1:-1, 1:-1] += fy * falloff

    def add_dye(self, gx: float, gy: float, color: int,
                radius: float, amount: float) -> None:
        """Drop a soft blob of one palette color into the pool."""
        dx = self.gx - gx
        dy = self.gy - gy
        falloff = np.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius))
        self.dye[color][1:-1, 1:-1] += amount * falloff

    # -- rendering -----------------------------------------------------------
    def composite(self, exposure: float = 1.7) -> np.ndarray:
        """Blend all dye layers into an RGB image with a soft filmic curve."""
        acc = np.zeros((self.n, self.n, 3))
        for k in range(self.n_dye):
            acc += self.dye[k][1:-1, 1:-1, None] * np.asarray(PALETTE[k])
        rgb = 1.0 - np.exp(-exposure * acc)
        bg = np.asarray(BACKGROUND)
        return bg + (1.0 - bg) * rgb


# ---------------------------------------------------------------------------
# The paddles that stir the pool
# ---------------------------------------------------------------------------
STIRRERS = [
    # cx, cy  centre of the wander path;  ax, ay  how far it wanders
    # wx, wy  wander speed;  ph  phase;  r  influence radius;  s  spin (+/-)
    dict(cx=0.32, cy=0.34, ax=0.20, ay=0.17, wx=0.13, wy=0.09,
         ph=0.0, r=0.17, s=+1.0, color=1),
    dict(cx=0.68, cy=0.30, ax=0.16, ay=0.22, wx=0.07, wy=0.11,
         ph=2.1, r=0.19, s=-1.0, color=3),
    dict(cx=0.50, cy=0.70, ax=0.24, ay=0.14, wx=0.10, wy=0.06,
         ph=4.3, r=0.20, s=+1.0, color=4),
    dict(cx=0.30, cy=0.72, ax=0.14, ay=0.18, wx=0.05, wy=0.08,
         ph=1.2, r=0.16, s=-1.0, color=2),
    dict(cx=0.74, cy=0.68, ax=0.18, ay=0.16, wx=0.09, wy=0.12,
         ph=5.5, r=0.18, s=+1.0, color=5),
]

FORCE_STRENGTH = 4.0     # paddle acceleration (equilibrium speed = this / DAMPING)
DAMPING = 10.0           # velocity drag; keeps the flow at a steady, calm speed
DYE_RATE = 0.05          # how fast fresh color enters
DYE_RADIUS = 0.055       # blob size at the injection point


def drive(sim: StableFluid, t: float) -> None:
    """Advance the paddles to time t and let them stir + inject color."""
    for st in STIRRERS:
        px = st["cx"] + st["ax"] * np.sin(st["wx"] * t + st["ph"])
        py = st["cy"] + st["ay"] * np.sin(st["wy"] * t + st["ph"] * 1.7)

        dx = sim.gx - px
        dy = sim.gy - py
        # Smooth analytic vortex. The tangential direction is (-dy, dx)/r, but
        # dividing by r blows up at the core, so fold the r into the falloff
        # instead:  speed ~ (r/r0) * exp(-r^2/2r0^2)  ->  0 at the centre and
        # at infinity, smooth everywhere. No grid-scale noise.
        r0 = st["r"]
        falloff = np.exp(-(dx * dx + dy * dy) / (2.0 * r0 * r0))
        fx = -dy / r0 * st["s"] * FORCE_STRENGTH * falloff
        fy = dx / r0 * st["s"] * FORCE_STRENGTH * falloff
        sim.u[1:-1, 1:-1] += fx * sim.dt
        sim.v[1:-1, 1:-1] += fy * sim.dt

        sim.add_dye(px, py, st["color"], DYE_RADIUS, DYE_RATE)


# ---------------------------------------------------------------------------
# Front ends
# ---------------------------------------------------------------------------
def run_selftest(grid: int, frames: int) -> int:
    """Headless smoke test -- no window, just verify the physics behaves."""
    sim = StableFluid(n=grid)
    t0 = time.perf_counter()
    for f in range(frames):
        drive(sim, f * sim.dt)
        sim.velocity_step()
        sim.dye_step()
    elapsed = time.perf_counter() - t0

    img = sim.composite()
    dye_mass = sum(float(d[1:-1, 1:-1].sum()) for d in sim.dye)
    speed = float(np.abs(sim.u[1:-1, 1:-1]).mean())

    # CFL: how many cells a parcel of dye travels per step. Must stay well
    # below 1 or semi-Lagrangian advection skips over structure and the
    # result turns to mush.
    cfl = float(np.abs(sim.u[1:-1, 1:-1]).max()) * sim.dt * grid
    # Roughness: mean absolute Laplacian of the image. High = speckle/grain.
    lap = np.abs(
        -4 * img[1:-1, 1:-1]
        + img[:-2, 1:-1] + img[2:, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:]
    ).mean()

    print(f"grid          : {grid}x{grid}")
    print(f"frames        : {frames}  ({elapsed:.2f}s, {elapsed / frames * 1000:.1f} ms/frame)")
    print(f"fps estimate  : {frames / elapsed:.1f}")
    print(f"image range   : {img.min():.4f} .. {img.max():.4f}")
    print(f"finite        : {np.isfinite(img).all()}")
    print(f"dye mass      : {dye_mass:.1f}")
    print(f"mean |u|      : {speed:.3f}")
    print(f"CFL           : {cfl:.3f}")
    print(f"roughness     : {lap:.5f}")

    if not np.isfinite(img).all():
        print("FAIL: simulation diverged to NaN/Inf", file=sys.stderr)
        return 1
    if img.max() - img.min() < 1e-3:
        print("FAIL: image is flat -- no visible structure", file=sys.stderr)
        return 1
    if cfl > 1.0:
        print(f"WARN: CFL {cfl:.2f} > 1 -- advection is under-resolved", file=sys.stderr)
    print("OK")
    return 0


def run_window(grid: int, fps: int, exposure: float) -> int:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    sim = StableFluid(n=grid)
    interval = 1000.0 / fps

    fig = plt.figure(figsize=(7.5, 7.5), facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(sim.composite(exposure), interpolation="bilinear",
                   animated=True, vmin=0.0, vmax=1.0)

    state = {"frame": 0, "paused": False}

    def update(_):
        if not state["paused"]:
            for _ in range(2):                      # 2 substeps per shown frame
                drive(sim, state["frame"] * sim.dt)
                sim.velocity_step()
                sim.dye_step()
                state["frame"] += 1
        im.set_data(sim.composite(exposure))
        return (im,)

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        gx = event.xdata / (grid - 1)
        gy = event.ydata / (grid - 1)
        # Splat a random color at the click, with a kick to the flow.
        color = int(np.random.randint(0, len(PALETTE)))
        for _ in range(3):
            sim.add_dye(gx, gy, color, 0.07, 0.5)
        ang = np.random.uniform(0, 2 * np.pi)
        sim.add_force(gx, gy, 40 * np.cos(ang), 40 * np.sin(ang), 0.08)

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    anim = FuncAnimation(fig, update, interval=interval, blit=True,
                         cache_frame_data=False)
    fig._anim = anim                                # keep a reference alive
    plt.show()
    return 0


def run_save(grid: int, frames: int, out_path: str, fps: int,
             exposure: float) -> int:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    sim = StableFluid(n=grid)
    fig = plt.figure(figsize=(7.5, 7.5), facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(sim.composite(exposure), interpolation="bilinear",
                   vmin=0.0, vmax=1.0)

    def update(i):
        drive(sim, i * sim.dt)
        sim.velocity_step()
        sim.dye_step()
        im.set_data(sim.composite(exposure))
        return (im,)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps,
                         blit=False, cache_frame_data=False)
    print(f"rendering {frames} frames -> {out_path} ...")
    anim.save(out_path, fps=fps, dpi=100,
              savefig_kwargs={"facecolor": BACKGROUND})
    print("done")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=144,
                    help="simulation resolution (default 144; try 96 for speed, 192 for detail)")
    ap.add_argument("--fps", type=int, default=30, help="target frames per second")
    ap.add_argument("--exposure", type=float, default=1.7,
                    help="brightness of the colors (default 1.7)")
    ap.add_argument("--save", metavar="PATH",
                    help="render to a video file instead of opening a window (.mp4 or .gif)")
    ap.add_argument("--frames", type=int, default=600,
                    help="frames to render when using --save")
    ap.add_argument("--selftest", action="store_true",
                    help="run a headless check and exit (no window)")
    ap.add_argument("--selftest-frames", type=int, default=40)
    args = ap.parse_args(argv)

    if args.selftest:
        return run_selftest(args.grid, args.selftest_frames)
    if args.save:
        return run_save(args.grid, args.frames, args.save, args.fps, args.exposure)
    return run_window(args.grid, args.fps, args.exposure)


if __name__ == "__main__":
    raise SystemExit(main())

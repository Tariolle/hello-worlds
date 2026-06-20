"""Manim 3-D energy-landscape animation for the live pitch.

Loads the real TUAB latent energy grid + points (energy_data.npz, exported by
examples/eeg/energy_export.py) and renders: the energy surface E=-log p_hat forming, the
recordings settling into its basins (normal=blue, abnormal=red), then a camera orbit.

Render (720p):  manim -qm results/energy/energy_manim.py EnergyLandscape
"""
import os

import numpy as np
from manim import (ThreeDScene, ThreeDAxes, Surface, Dot3D, VGroup, Text, Create, FadeIn,
                   Write, DEGREES, OUT, DOWN, BLUE, RED, GREY_B, BLUE_E, PURPLE, ORANGE, YELLOW)
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "energy_data.npz"))
GX, GY, E = D["gx"], D["gy"], D["energy"]
EMB, Y, PE = D["emb"], D["y"], D["point_energy"]
gxv, gyv = GX[:, 0], GY[0, :]
INTERP = RegularGridInterpolator((gxv, gyv), E, bounds_error=False, fill_value=float(E.max()))

# subsample points so the rotating scene stays light
rng = np.random.default_rng(0)
idx = rng.permutation(len(EMB))[:180]
EMB, Y, PE = EMB[idx], Y[idx], PE[idx]


def sx(x):
    return float(np.interp(x, [GX.min(), GX.max()], [-3.5, 3.5]))


def sy(y):
    return float(np.interp(y, [GY.min(), GY.max()], [-3.5, 3.5]))


def sz(e):
    return float(np.interp(e, [E.min(), E.max()], [0.0, 2.6]))


class EnergyLandscape(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=62 * DEGREES, theta=-55 * DEGREES, zoom=0.85)
        axes = ThreeDAxes(x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[0, 3, 1],
                          x_length=7, y_length=7, z_length=3)

        def func(u, v):
            e = float(INTERP([[u, v]])[0])
            return np.array([sx(u), sy(v), sz(e)])

        surface = Surface(func, u_range=[gxv.min(), gxv.max()], v_range=[gyv.min(), gyv.max()],
                          resolution=(46, 46), fill_opacity=0.9, stroke_width=0.15)
        surface.set_fill_by_value(axes=axes, colorscale=[BLUE_E, PURPLE, ORANGE, YELLOW], axis=2)

        title = Text("TUAB latent energy landscape", font_size=30)
        sub = Text("E = -log density (proxy)  -  SIGReg-ambient encoder  -  normal / abnormal",
                   font_size=18, color=GREY_B)
        title.to_corner(np.array([-1, 1, 0]))
        sub.next_to(title, DOWN, aligned_edge=np.array([-1, 0, 0]))
        self.add_fixed_in_frame_mobjects(title, sub)

        self.play(Write(title), FadeIn(sub), run_time=1.2)
        self.play(Create(surface), run_time=3.5)

        dots = VGroup(*[
            Dot3D(point=[sx(px), sy(py), sz(pez) + 0.06], radius=0.05,
                  color=(BLUE if yy == 0 else RED), resolution=(6, 6))
            for (px, py), yy, pez in zip(EMB, Y, PE)
        ])
        self.play(FadeIn(dots, shift=OUT * 0.6), run_time=1.6)

        self.begin_ambient_camera_rotation(rate=0.22)
        self.wait(8)
        self.stop_ambient_camera_rotation()
        self.wait(0.4)

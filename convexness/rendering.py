"""
Inspired from https://github.com/rnd-team-dev/plotoptix/blob/master/examples/1_basics/1_scatter_plot_3d.py.
"""

import numpy as np
from plotoptix import TkOptiX, DenoiserKind
from plotoptix.materials import m_shadow_catcher


def render_points(points, geometry, colors, r=0.02):
    # Create plots:
    rt = TkOptiX()  # create and configure, show the window later

    #
    rt.set_param(max_accumulation_frames=500)

    # bounding box
    max_p, min_p = np.max(points[0], axis=0), np.min(points[0], axis=0)
    for p in points[1:]:
        max_p = np.maximum(max_p, np.max(p, axis=0))
        min_p = np.minimum(min_p, np.min(p, axis=0))

    # add plots, ParticleSet geometry with variable radius is default
    for idx, (p, g, c) in enumerate(zip(points, geometry, colors)):
        if p.shape[0] == 0:
            continue

        rt.set_data("particles " + str(idx), pos=p, r=r, c=c)

    # lighting
    rt.setup_light("light1", pos=[8, 20, 5], color=[0.3, 0.3, 0.3], radius=1.5)
    rt.setup_light("light2", pos=[4, 15, -5], color=[0.6, 0.6, 0.6], radius=1.5)

    # add shadow catcher at the bottom
    # center = (max_p + min_p) / 2
    # bottom_pos = [-50.0, min_p[1], -50.0]

    # rt.setup_material("shadow", m_shadow_catcher)
    # rt.set_data("plane", pos=bottom_pos, u=[100, 0, 0], v=[0, 0, 100], c=1, geom="Parallelograms", mat="shadow")

    # show coordinates box
    # rt.set_coordinates()

    # postprocessing
    rt.set_float("denoiser_blend", 0.25)
    rt.set_int("denoiser_kind", DenoiserKind.RgbAlbedoNormal.value)
    rt.add_postproc("Denoiser")

    # show the UI window here - this method is calling some default
    # initialization for us, e.g. creates camera, so any modification
    # of these defaults should come below (or we provide on_initialization
    # callback)
    rt.show()

    # camera and lighting configured by hand
    eye = (max_p + min_p) / 2 + (max_p - min_p) * 1.5
    rt.update_camera(eye=eye)

    # ambient colors
    rt.set_ambient(1.0)
    rt.set_background(1)

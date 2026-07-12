# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "pymol-open-source-whl",
# ]
# ///

import os
import sys

# Set environment variable for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pymol # pytype: disable=import-error
pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()

from pymol import cmd # pytype: disable=import-error

cmd.load("cg_trajectory.pdb", "trajectory")
# Hide lines, show spheres for the CG beads
cmd.hide("lines")
cmd.show("spheres")
# Adjust sphere scale
cmd.set("sphere_scale", 0.5)

# Color by B-factor or just a simple color
cmd.color("cyan", "trajectory")

# Setup animation playback speed
cmd.mset("1 -500")
cmd.mplay()

cmd.orient()
cmd.set("ray_opaque_background", 1)
cmd.png("trajectory_frame1.png", width=1200, height=900, dpi=150)
cmd.save("animation.pse")
cmd.quit()

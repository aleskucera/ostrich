"""Helhest endpoint optimization using Genesis (Taichi-based differentiable physics).

NOT FUNCTIONAL — Genesis backward pass hangs for articulated robots.

Genesis loss.backward() hangs indefinitely for any articulated robot where a
free-floating root body (freejoint) has child joints (revolute, prismatic, etc.).
The backward pass blocks inside the ABD backward kernel. This affects all tested
Genesis versions (v0.3.8, v0.3.9, v0.4.1).

Single free-floating bodies (freejoint only, no children) work correctly — see
ball_throw/genesis_sim.py for a working example. The hang is triggered
exclusively by kinematic trees of depth >= 2 with a free root.

The original version used manual gradient injection via robot.set_pos_grad() +
scene._backward(), which also hangs (deadlock in kernel_step_2.grad /
kernel_forward_dynamics_without_qacc.grad).

This script is kept as a placeholder for when Genesis fixes this limitation.
"""

import os
import tempfile
import time

import genesis as gs
import torch

os.environ.setdefault("DISPLAY", ":1")
os.environ.pop("WAYLAND_DISPLAY", None)

DT = 2e-3
DURATION = 3.0
T = int(DURATION / DT)

TARGET_CTRL = [1.0, 6.0, 0.0]
INIT_CTRL = [2.0, 5.0, 0.0]

WHEEL_DOFS = [6, 7, 8]  # freejoint=6 DOFs [0..5], wheels at [6,7,8]
CHASSIS_LINK_IDX = 1    # link 0 = world, link 1 = chassis

HELHEST_MJCF = """
<mujoco model="helhest">
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="0.7 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0"
                diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
      <body name="battery" pos="-0.302 0.165 0">
        <inertial mass="2.0" pos="0 0 0" diaginertia="0.00768 0.0164 0.01208"/>
        <geom type="box" size="0.125 0.05 0.095" rgba="0.3 0.3 0.8 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="left_motor" pos="-0.09 0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="right_motor" pos="-0.09 -0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="rear_motor" pos="-0.22 -0.04 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" rgba="0.8 0.3 0.3 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel_holder" pos="-0.477 0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" rgba="0.6 0.6 0.6 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="right_wheel_holder" pos="-0.477 -0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" rgba="0.6 0.6 0.6 0.3"
              contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.7 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.7 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="0.35 0.1 0.01" rgba="0.15 0.15 0.15 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    print("ERROR: Genesis backward pass hangs for articulated robots (freejoint + child joints).")
    print("This is a known Genesis bug affecting all tested versions (v0.3.8, v0.3.9, v0.4.1).")
    print("See docstring for details. Skipping.")
    return


if __name__ == "__main__":
    main()

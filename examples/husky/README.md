# Clearpath Husky A200

A Husky A200 that drives in the clark_paper campaign worlds the way the Helhest
Junior does. Four-wheel skid-steer, keyboard teleop, same terrain treatment and
same CLI as `examples/helhest_junior/control_campaign.py`.

## Run it

Viewer (drive with `I` / `J` / `K` / `L`):

```bash
cd ~/projects/ostrich
.venv/bin/python examples/husky/control_campaign.py world=curb_or_mound variant=box30
```

Any world and variant the campaign built, and any pose from the spec:

```bash
.venv/bin/python examples/husky/control_campaign.py world=shadowed_trench variant=hidden
.venv/bin/python examples/husky/control_campaign.py world=curb_or_mound variant=box30 pose=decision
```

Headless, which is how it is smoke-tested:

```bash
.venv/bin/python examples/husky/control_campaign.py \
    rendering=headless world=curb_or_mound variant=box30 simulation.duration_seconds=2
```

The worlds come from the sibling `clark_paper` checkout
(`clark_paper/sim_worlds/results/<world>_<variant>.npz`), found by
`load_campaign_world`, which this file imports from the Helhest campaign script
rather than duplicating. Set `CLARK_PAPER=/path/to/clark_paper` if the checkout
is not `../clark_paper`.

Files: `common.py` (`create_husky_model`, `HuskyConfig`), `control_campaign.py`
(teleop + world), `../assets/husky.urdf`, `../conf/husky_campaign.yaml`.

## Where the numbers come from

Source: <https://github.com/husky/husky>, package `husky_description`, files
`urdf/husky.urdf.xacro` and `urdf/wheel.urdf.xacro`, read at the `melodic-devel`
revision. `examples/assets/husky.urdf` is the hand-expanded plain-URDF copy;
nothing in ostrich parses it (the same status as `examples/assets/helhest.urdf`),
it exists so every number in `HuskyConfig` can be traced to a line of the
upstream description.

| Quantity | Value | Source |
| --- | --- | --- |
| chassis box, X x Y x Z | 0.98740 x 0.57090 x 0.24750 m | `base_x_size`, `base_y_size`, `base_z_size` |
| lower collision box | 0.98740 x 0.57090 x 0.12375 m at z = 0.061875 | `base_link` collision 1: full size, `base_z_size/2` tall, centred `base_z_size/4` |
| upper collision box | 0.78992 x 0.57090 x 0.10375 m at z = 0.175625 | `base_link` collision 2: `base_x_size*4/5`, `base_z_size/2 - 0.02`, centred `base_z_size*3/4 - 0.01` |
| chassis mass | 46.034 kg | `inertial_link` |
| chassis CoM | (-0.00065, -0.085, 0.062) m from `base_link` | `inertial_link` origin |
| chassis inertia | ixx 0.6022, ixy -0.02364, ixz -0.1197, iyy 1.7386, iyz -0.001544, izz 2.0296 | `inertial_link` |
| wheel radius | 0.1651 m | `wheel_radius` |
| wheel width | 0.1143 m | `wheel_length` |
| wheel mass | 2.637 kg | `husky_wheel` macro |
| wheel inertia | ixx 0.02467, iyy 0.04411, izz 0.02467 | `husky_wheel` macro |
| wheelbase | 0.5120 m | `wheelbase` |
| track | 0.5708 m | `track` |
| wheel mounts | (+-0.2560, +-0.2854, 0.03282) m | `(+-wheelbase/2, +-track/2, wheel_vertical_offset)` |
| wheel joints | continuous, axis (0, 1, 0), velocity interface | `husky_wheel` macro |
| ground clearance | 0.13228 m | derived: `wheel_radius - wheel_vertical_offset`, and it is exactly where `base_footprint` sits |

Total model mass is 46.034 + 4 x 2.637 = 56.582 kg.

Two source notes. First, `track` is 0.5708 on `kinetic-devel` and
`melodic-devel` but 0.555 on `noetic-devel`; 0.5708 is used here. Every other
number above is identical across those three branches. Second, the wheel inertia
is not that of a uniform cylinder: a solid cylinder of 2.637 kg and r = 0.1651 m
gives 0.0359 axial / 0.0208 transverse against the description's 0.04411 /
0.02467, so the description's rim-heavy values are used as written.

### Assumed, because the source does not give it

* **Friction**: lateral (skid, along the axle) 0.5, longitudinal (rolling) 0.9,
  rolling resistance 0.7. The description only carries Gazebo's `mu1 = mu2 = 1.0`
  with `fdir1 = "1 0 0"`, an isotropic pair in a different contact model, so it
  does not transfer. These are the Helhest campaign's front-wheel values, applied
  to all four wheels. **Untuned**: nothing here was fitted to a Husky.
* **Motor gains**: `control_mode = "velocity"`, `k_p = 250.0`, `k_d = 0.0`,
  copied from `helhest_campaign.yaml`. In velocity mode `k_p` is the only gain
  the engine reads: the control constraint's compliance is `1 / (dt * k_p)`, so a
  larger `k_p` tracks the commanded wheel speed harder. `k_d` is read in position
  mode only. **Untuned.**
* **Teleop speeds**: 5.0 rad/s forward, 3.0 rad/s differential for a turn, ramped
  at 10 rad/s^2 up and 20 rad/s^2 down. Taken unchanged from the Helhest teleop
  so a drive feels the same. On the Husky's smaller wheel 5.0 rad/s is 0.83 m/s,
  which is under the A200's 1.0 m/s rated top speed, so it is a usable number,
  but it is not a Husky spec.
* **Contact stiffness** (`ke`/`kd`/`kf`) is left at the builder default, as the
  Helhest campaign leaves it.

## Model choices

* **Body frame is the wheel-axle plane, not `base_link`.** The description hangs
  the wheels 0.03282 m above `base_link` (the centre of the bottom plate). Every
  chassis z in `HuskyConfig` is the URDF value minus that offset, which puts all
  four axles at z = 0. That is the Helhest convention and it is what makes the
  campaign spawn rule `ground + WHEEL_RADIUS + 0.02` correct.
* **Chassis mass properties are the measured ones, not box densities.** The two
  chassis boxes are added at `density = 0.0` and the link carries the
  description's mass, CoM and full inertia tensor directly, the same way the
  Helhest wheels do. The 8.5 cm CoM offset in -Y (the battery) is kept; it is
  what the real vehicle does, and it makes the robot very slightly asymmetric.
* **Wheels are cylinders, not capsules**, for the reason set out in the Helhest
  Junior's `common.py`: a capsule's hemispherical caps bulge a wheel radius past
  each rim and catch on obstacles the real tread never touches.
* **Wheel inertia is placed on the spin axis.** `WHEEL_I` is `diag(ixx, iyy, izz)`
  with the axial term `iyy = 0.04411` on Y, which is the revolute axis. (The
  Helhest Junior's `WHEEL_I` puts its axial term on Z while its joint axis is Y;
  that file is not touched here.)
* **Skid-steer**: both wheels of a side share one command, so the four joint dofs
  6, 7, 8, 9 (front left, front right, rear left, rear right, after the free
  base joint's 6) are driven as two pairs.
* **Self-collision is off** by the Helhest's mechanism: every robot shape is in
  `collision_group = -1`, and newton's rule is that two shapes in the same
  negative group never collide. This matters here because the description's own
  collision boxes overlap the wheels: the box half width 0.28545 m reaches past
  the wheel inner face at 0.22825 m. Terrain is in the default positive group and
  still collides with everything.
* **Visuals are the collision primitives themselves** (`is_visible = True` on the
  boxes and cylinders). There is no Husky mesh in `examples/assets`, and the
  description's visuals are `.dae` meshes.

## Smoke test

`rendering=headless world=curb_or_mound variant=box30 simulation.duration_seconds=2`
(67 steps of 0.0299 s, ostrich engine, one world), chassis pose read from
`body_pose` in the HDF5 log:

* **Model builds**: 5 bodies (chassis + 4 wheels), 10 joint dofs (free base + 4
  revolutes), terrain 181x101 cells at 0.10 m plus 6 solids, no warnings.
* **Settles to the wheel radius.** Spawned at z = 0.18510
  (`ground + 0.1651 + 0.02`) on flat ground at the `curb_or_mound/box30` start
  pose (0.00, 3.20), the chassis frame rests at **z = 0.16511 m**, i.e. 0.011 mm
  above the 0.1651 m wheel radius, from step 1 onward. Peak-to-peak over the last
  20 steps is 0.00000 m. All four wheel centres land within 1.2e-5 m of the same
  height.
* **Drives forward.** With both sides commanded to 5.0 rad/s through the same
  rate limiter the keyboard uses, the chassis travels **1.4609 m in 2.0 s**
  (final linear speed 0.80 m/s). The no-slip prediction for that ramped command,
  `sum(target_vel) * dt * r`, is 1.4578 m, so tracking is within 0.2%. Lateral
  drift is -6.7 mm and yaw drift -0.24 deg over the run. The forced-command
  driver used for this is a scratch script, not a file in the repo; the keyboard
  path is what ships.
* **Only `control.mode=velocity` was exercised.** The position-mode branch
  mirrors the Helhest teleop's angle integrator and is untested here.

## Not matched

* **Bumpers.** The front and rear bumpers sit at x = +-0.48 m, z = 0.091 m, and
  are visual-only in the description (no `<collision>` element), so they are
  omitted. The model is therefore 0.9874 m long where the real vehicle with
  bumpers is closer to 1.1 m. The chassis box's front face at x = +0.4937 m does
  land just ahead of the bumper mount at 0.48 m, so the error is in the bumper's
  own depth, not its position.
* **Wheel tread.** The wheel is a smooth cylinder. The real Husky's lugged tyre
  is what bites into loose ground, and the anisotropic mu pair is a stand-in for
  it, not a model of it.
* **Suspension.** Neither the real A200 nor this model has any: the vehicle's
  only vertical compliance is tyre deflection, and a rigid cylinder has none, so
  the model is stiffer over a curb edge than the vehicle is.
* **Top plate, user rail, sensor arch, sensors, PACS.** All omitted. They are
  separate links in the description, and their mass is not inside the 46.034 kg
  either, so leaving them out is consistent, but a loaded Husky is heavier and
  higher-CoM than this.
* **Wheel wells.** The description's chassis collision box spans the full
  `base_y_size` and does not cut out the wheels, so the box overlaps them. That
  overlap is filtered (see above) rather than fixed, exactly as upstream leaves
  it for Gazebo.
* **Motor model.** A velocity-target constraint with one stiffness, not the
  A200's gearbox, current limit or the differential-less coupling of the two
  wheels on one side, which on the real vehicle are chain-driven from one motor.

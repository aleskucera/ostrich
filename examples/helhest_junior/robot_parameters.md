# The Helhest Junior Robot — A Complete Description

This document describes the **Helhest Junior**, a small mobile robot, in enough
detail that someone who has never seen it can understand what it is, how it moves,
and reproduce its exact physical parameters in a simulator.

Everything here is taken from the model definition in
`examples/helhest_junior/common.py` (the `HelhestJuniorConfig` class), which is
the single source of truth for the simulated robot.

> **Read this first — not every number is a measurement.** The parameters below
> come from three different sources, and it matters which is which:
>
> - 📏 **Measured** — taken from the real robot (a ruler or a scale).
> - 🧮 **Assumed / derived** — computed from the measured numbers under an
>   idealizing assumption (e.g. "treat the wheel as a uniform solid cylinder").
>   These are *not* measured; if the real mass is distributed differently, the
>   true value differs.
> - 🎛️ **Fine-tuned** — chosen so the *simulation behaves like the real robot*,
>   not measured at all. These are knobs adjusted during calibration.
>
> The **inertia tensors are all 🧮 assumed**, and the **motor gains and friction
> are 🎛️ fine-tuned**. See the provenance table in §5 and the dedicated
> discussion in §6, §8, and §9.

---

## 1. What is it?

The Helhest Junior is a **three-wheeled ground robot** that drives over flat
ground and small obstacles (it has been tested climbing a 16 cm box). It is the
smaller sibling of a larger robot called "Helhest"; "Junior" has the same basic
layout but its own dimensions and weights.

Think of it as a sturdy, heavy little rover:

- A **rigid body** ("chassis") that carries all the weight.
- **Three wheels**, each driven by its own motor. There is **no steering rack**
  and the wheels do not pivot — the robot turns the way a tank does, by spinning
  the left and right wheels at different speeds. This is called **skid-steering**.
- It weighs about **106 kg** in total, so it is small in footprint but quite
  dense (roughly the weight of a large adult).

It is used here as a "digital twin": the real robot was driven around and its
wheel commands + trajectory recorded, and the simulator is tuned to reproduce
that same motion (see `replay_real.py` and the `experiments/` folder).

---

## 2. How it is built (layout)

```
                 TOP VIEW (looking down)
                 X = forward ↑,  Y = left ←

                    front of robot
                 ┌──────────────────┐
        left     │                  │     (front box: wide,
       wheel ◯───┤   front box      │      heavy — holds most
    (Y=+0.365)   │   0.48 × 0.56    │      of the mass)
                 │                  │
        right    │                  │
       wheel ◯───┤                  │
    (Y=−0.365)   └─────┬──────┬─────┘
                       │ rear │
                       │ box  │        (rear box: narrow, light)
                       │ 0.48 │
                       │×0.24 │
                  ◯────┤      │
              rear     └──────┘
             wheel
          (X=−0.75)        back of robot
```

- The **front of the robot is +X**, **left is +Y**, **up is +Z**.
- The two **front wheels** (left and right) sit side by side on a common axle
  line, which is defined as **X = 0**.
- The **single rear wheel** sits on the centerline, 0.75 m behind the front
  axle. So the wheels form a **triangle**: two in front, one in back.
- The body is made of **two rectangular boxes** fixed rigidly together: a big
  heavy front box and a smaller lighter rear box.

---

## 3. Coordinate frame and units

- **All units are SI**: metres (m), kilograms (kg), kg·m² for inertia.
- **Axes** (right-handed):
  - **X** — longitudinal, pointing **forward** (front of robot = +X)
  - **Y** — lateral, pointing **left** (left wheel = +Y)
  - **Z** — vertical, pointing **up**
- **Origin** of the chassis frame: on the front-wheel axle line (X = 0),
  centred laterally (Y = 0), at axle height.

---

## 4. How it moves (locomotion)

The robot is a **skid-steer** (a.k.a. differential-drive) vehicle:

- Each of the three wheels has its own motor and is commanded by a **target
  wheel velocity** (how fast that wheel should spin). Positive velocity = the
  robot rolls **forward**.
- **Driving straight:** all wheels turn at the same speed.
- **Turning:** the left and right wheels turn at *different* speeds. If the
  right wheels turn faster than the left, the robot yaws to the left, and vice
  versa. Turning therefore requires the wheels to **skid/scrub** sideways
  against the ground — which is why ground friction (and especially the
  resistance to twisting, "torsional friction") strongly affects how it turns.
- There is **no separate steering mechanism**; heading is a pure consequence of
  the three wheel speeds and the ground contact.

In the simulator the wheel joints are revolute (hinge) joints with their axis
along **Y**, driven in `TARGET_VELOCITY` mode.

---

## 5. Where each parameter comes from (measured vs. assumed vs. tuned)

This is the most important table in the document. **Only the dimensions and the
total/wheel masses are grounded in the real robot.** The inertia tensors are
analytic idealizations, and the actuation and friction are calibration knobs.

| Parameter | Provenance | How it was obtained |
|---|---|---|
| Wheel radius, width | 📏 measured | ruler on the real wheel |
| Wheel mass (5.5 kg) | 📏 measured | scale |
| Wheel positions / track / wheelbase | 📏 measured | geometry of the real frame |
| Chassis box dimensions | 📏 measured | ruler on the real body |
| Total robot mass (106.2 kg) | 📏 measured | scale |
| Per-wheel load (≈39.1 / 28.0 kg) | 📏 measured | weighed each wheel's contact |
| Front/rear box masses | 🧮 derived | back-solved from the per-wheel loads + geometry (uniform-density assumption per box) |
| **Wheel inertia tensor** | 🧮 **assumed** | analytic **uniform solid cylinder** formula — *not measured* |
| **Chassis inertia tensor** | 🧮 **assumed** | each box treated as a **uniform-density solid box**; the engine computes the tensor from mass+size — *not measured* |
| Robot / chassis centre of mass | 🧮 derived | from the box masses + positions above |
| **Motor gains (TARGET_KE/KD)** | 🎛️ **fine-tuned** | chosen so wheels track commands stably; *not a measured motor constant* |
| **Wheel friction (front/rear)** | 🎛️ **fine-tuned** | calibrated to match real motion; geometry-file defaults differ from the values actually used in experiments |
| **Rolling friction** | 🎛️ **fine-tuned** | calibration knob |
| Contact stiffness ke/kd/kf | 🎛️ fine-tuned | engine-dependent penalty knob (ignored by the Ostrich/Axion solver) |

**Bottom line:** trust the **geometry and masses**. Treat the **inertia tensors
as modelling assumptions** (uniform density), and treat **gains + friction as
tuned parameters** that exist to reproduce observed behaviour, not as physical
measurements of the real robot.

---

## 5b. Mass and weight distribution

The robot's total mass is **106.2 kg**, split between the body and the wheels:

| Component | Mass (kg) | Count | Subtotal (kg) |
|---|---|---|---|
| Front chassis box | 78.8375 | 1 | 78.8375 |
| Rear chassis box | 10.8625 | 1 | 10.8625 |
| **Chassis (body) total** | | | **89.7** |
| Wheel | 5.5 | 3 | 16.5 |
| **Robot total** | | | **106.2** |

The body is **front-heavy**: the front box holds ~88% of the chassis mass.
Centres of mass (along the X axis, behind the front axle):

- Whole-robot centre of mass: **X = −0.198 m**
- Chassis-only centre of mass: **X = −0.188 m**

(These were not guessed — the box masses were back-solved from per-wheel scale
measurements of the real robot: ~39.1 kg on each front wheel, ~28.0 kg on the
rear.)

---

## 6. Wheels (detailed)

All three wheels are identical:

| Property | Value | Plain-language meaning |
|---|---|---|
| Radius | 0.35 m | 35 cm — a fairly large wheel |
| Width | 0.10 m | 10 cm thick |
| Mass (each) | 5.5 kg | |
| Spin inertia (I_axial) 🧮 | 0.336875 kg·m² | resistance to spinning about its own axle (*assumed*) |
| Tilt inertia (I_transverse) 🧮 | 0.173021 kg·m² | resistance to tipping/tumbling (*assumed*) |
| Collision shape | cylinder, r = 0.35, half-height = 0.05 | what the physics engine actually collides |
| Visual mesh | `assets/helhest/wheel2.obj`, scaled 0.7583 | what you see in the viewer |

> 🧮 **The wheel inertia tensor is assumed, not measured.** It is the analytic
> inertia of a *uniform solid cylinder* of the measured mass and size. A real
> wheel (hub, tyre, spokes, motor) does not have uniform density, so the true
> tensor differs — but in the absence of a measurement this idealization is used.

As a diagonal tensor (about the wheel's own centre, axle = the third axis):

```
WHEEL_I = [ 0.173021      0           0       ]   kg·m²
          [    0       0.173021       0       ]
          [    0          0        0.336875   ]   ← spin (axle) axis
```

The values are exactly the uniform-solid-cylinder formula for the measured
m = 5.5 kg, r = 0.35 m, h = 0.10 m:
- spin axis: I = ½·m·r² = ½·5.5·0.35² = **0.336875**
- transverse: I = 1/12·m·(3r² + h²) = **0.173021**

### Wheel positions (in the chassis frame)

| Wheel | Position (x, y, z) in metres |
|---|---|
| Left front | (0.0, **+0.365**, 0.0) |
| Right front | (0.0, **−0.365**, 0.0) |
| Rear | (**−0.75**, 0.0, 0.0) |

- **Track** (distance between the left and right wheels): **0.73 m**
- **Wheelbase** (front axle to rear wheel): **0.75 m**

So the robot's wheel "footprint" is roughly a 0.73 m × 0.75 m triangle.

---

## 7. The body / chassis (detailed)

The chassis is modelled as **two boxes rigidly bolted together** (they never
move relative to each other — together they form one rigid body).

Important orientation note: each box's **0.48 m side runs front-to-back (X)**,
and the wider 0.56 m / 0.24 m side runs **left-to-right (Y)**. Heights are
0.20 m (Z).

| Box | Centre (x, y, z) m | Size (x, y, z) m | Mass (kg) | Role |
|---|---|---|---|---|
| Front box | (−0.13, 0, 0) | 0.48 × 0.56 × 0.20 | 78.8375 | wide, heavy main body |
| Rear box | (−0.61, 0, 0) | 0.48 × 0.24 × 0.20 | 10.8625 | narrow, light tail |

Reference points along the length:

- Front box **front edge**: X = +0.11 m (11 cm ahead of the front-wheel axle)
- Front and rear boxes **meet** at X = −0.37 m (flush, no gap)
- Rear box **back edge**: X = −0.85 m (10 cm behind the rear wheel)

Overall body length ≈ 0.96 m (from +0.11 to −0.85), width 0.56 m at the front
tapering to 0.24 m at the tail, height 0.20 m.

> 🧮 **The chassis inertia tensor is assumed, not measured.** Each box is added
> to the simulator with a *uniform density* (mass ÷ volume), and the engine then
> computes the box's inertia tensor analytically from that. So the chassis
> rotational inertia is whatever two uniform-density solid boxes of these sizes
> and masses would have — it is **not** a measured property of the real body.
> The box **dimensions** (📏) and **masses** (🧮, back-solved from the measured
> per-wheel loads) are well grounded; the *distribution* of that mass within
> each box is the idealization.

---

## 8. Motors / actuation

- **3 driven wheel joints** — revolute (hinge) joints, one per wheel, all with
  their rotation axis along **Y**. Commanding a positive joint velocity drives
  the robot forward.
- **1 free joint** ("base_joint") — this is not a motor; it is how the chassis
  connects to the world, giving the whole robot its 6 degrees of freedom
  (3 translation + 3 rotation) so it can move and tip freely.
- **Motor gains** (how stiffly a wheel tracks its commanded motion):
  - `TARGET_KE` (stiffness / proportional gain) = **150.0**
  - `TARGET_KD` (damping / derivative gain) = **0.0**
  - (The model-builder helper uses k_p = 50, k_d = 0.1 by default for position
    mode; velocity mode is the one used in the experiments.)

> 🎛️ **These gains are fine-tuned, not measured.** They are not a datasheet motor
> constant or a measured controller gain — they are values chosen so the
> simulated wheels follow their velocity commands stably and reproduce the
> observed motion. Different solvers/timesteps may use different gains.

---

## 9. Friction / contact parameters

These describe how the wheels grip the ground. The values below are the
**defaults baked into the geometry file**.

> 🎛️ **All friction values are fine-tuned, not measured.** They are calibration
> knobs, not measured coefficients of friction for the real tyres. The numbers
> below are merely the file defaults; the experiments override them.

| Parameter | Default | Meaning |
|---|---|---|
| Front-wheel friction (left & right) | 0.7 | grip of the two front wheels |
| Rear-wheel friction | 0.4 | grip of the rear wheel |
| Rolling friction (`mu_rolling`) | 0.7 | resistance to rolling |

**Two important caveats:**

1. **These defaults are not the values used in the experiments.** The sim-to-real
   studies *calibrate* friction to match the real robot. For example, the MuJoCo
   reference model uses a friction of ~1.2 with a specific torsional-friction
   setting to reproduce the real turning. See
   `experiments/1_sim_to_real_box/README.md` for the calibrated values.
2. **Contact stiffness/damping (`ke`/`kd`/`kf`) depend on the physics engine.**
   The Ostrich/Axion solver uses its own internal compliance model and *ignores*
   these knobs; they only matter for the penalty-based "Semi-Implicit" engine.

---

## 10. Quick reference (all numbers in one place)

Legend:  📏 measured   🧮 assumed/derived   🎛️ fine-tuned

```
GEOMETRY & MASS  (📏 measured, except box-mass split 🧮)
  TOTAL MASS            106.2 kg     📏  (chassis 89.7 + 3×5.5 wheels)
  ROBOT CoM (X)         −0.198 m     🧮
  CHASSIS CoM (X)       −0.188 m     🧮

  WHEELS (×3, identical)
    radius              0.35 m       📏
    width               0.10 m       📏
    mass                5.5 kg each  📏
    left  position      (0.0,  0.365, 0.0)   📏
    right position      (0.0, −0.365, 0.0)   📏
    rear  position      (−0.75, 0.0,  0.0)   📏
    track               0.73 m       📏
    wheelbase           0.75 m       📏

  CHASSIS (2 fixed boxes)
    front box  center (−0.13, 0, 0)  size 0.48×0.56×0.20   📏 size
    rear  box  center (−0.61, 0, 0)  size 0.48×0.24×0.20   📏 size
    front box mass 78.8375 kg   🧮 (back-solved from per-wheel loads)
    rear  box mass 10.8625 kg   🧮
    body length ≈ 0.96 m  (front edge +0.11, back edge −0.85)

INERTIA TENSORS  (🧮 ASSUMED — uniform-density idealization, NOT measured)
  wheel I_axial (spin)  0.336875 kg·m²   🧮 uniform solid cylinder
  wheel I_transverse    0.173021 kg·m²   🧮 uniform solid cylinder
  chassis inertia       computed by engine from box mass+size 🧮 uniform box

ACTUATION  (🎛️ FINE-TUNED — not measured motor constants)
  3 revolute wheel joints, axis = Y, velocity-controlled
  1 free base joint (chassis ↔ world)
  TARGET_KE = 150.0   🎛️
  TARGET_KD = 0.0     🎛️

FRICTION  (🎛️ FINE-TUNED — calibration knobs, file defaults shown)
  front wheels  0.7   🎛️
  rear wheel    0.4   🎛️
  rolling       0.7   🎛️
  (experiments override these — see experiments/1_sim_to_real_box/README.md)

FRAME:  X = forward,  Y = left,  Z = up;  origin on front axle
```

Source: `examples/helhest_junior/common.py` (`HelhestJuniorConfig`).

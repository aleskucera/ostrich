# Ostrich

A GPU-accelerated physics simulator with non-smooth contact and exact friction.

---

Ostrich is a rigid body simulator built for robotics research where contact accuracy matters. Unlike simulators that smooth or approximate the friction cone, Ostrich solves the **non-penetration contact constraints exactly** and handles **non-smooth friction** — making it particularly well-suited for skid-steer locomotion, flipper robots, and any scenario where slip behavior drives the dynamics.

**Key properties:**

- **Stable at large timesteps** — 5×10⁻² s works well in most scenes; some contact-heavy scenes remain stable at 1×10⁻¹ s
- **Parallel worlds** — run thousands of independent simulations simultaneously on a single GPU
- **Maximal coordinates** — makes it straightforward to model unconventional robot morphologies accurately
- **CUDA-accelerated** — built on [NVIDIA Warp](https://github.com/NVIDIA/warp) and [Newton](https://github.com/newton-physics/newton)

---

## Helhest — 3-Wheeled Skid-Steer Robot

Helhest is a three-wheeled skid-steer robot that can also flip itself upright. Accurate friction is critical here: skid-steer turning depends entirely on the difference in lateral friction forces between wheels, and the flip maneuver requires correct contact dynamics throughout.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
  <div>
    <video src="https://github.com/user-attachments/assets/dd1f1bc5-c91b-4d57-999b-4fda29c69596" autoplay loop muted playsinline width="100%"></video>
    <p align="center"><b>Skid-steer turning</b></p>
  </div>
  <div>
    <video src="https://github.com/user-attachments/assets/53e5079e-fe8d-4476-a2ec-e027f61fe2a6" autoplay loop muted playsinline width="100%"></video>
    <p align="center"><b>Self-righting flip</b></p>
  </div>
  <div>
    <video src="https://github.com/user-attachments/assets/363bcfca-ea37-4adb-8e50-8ff673c296e6" autoplay loop muted playsinline width="100%"></video>
    <p align="center"><b>Dense obstacle field</b></p>
  </div>
  <div>
    <video src="https://github.com/user-attachments/assets/3fb658a2-933c-46d7-9c4f-bd35bef470a6" autoplay loop muted playsinline width="100%"></video>
    <p align="center"><b>Motor realism on incline</b> — P-velocity controller drifts under load as expected</p>
  </div>
</div>

---

## Marv — Flipper Tracked Robot

Marv is a tracked robot with four articulated flippers — each flipper carries its own track. The combination of track contacts, flipper joints, and varying terrain makes this a demanding test for any simulator. Maximal coordinates make the morphology easy to express without hacks.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
  <div>
    <video src="https://github.com/user-attachments/assets/873f6c1c-9e41-4aed-a379-b4f9a37be361" autoplay loop muted playsinline width="100%"></video>
    <p align="center"><b>Skid-steer turning</b></p>
  </div>
  <div>
    <video src="https://github.com/user-attachments/assets/10f61094-0d38-4aee-97e8-01e133aa5c21" autoplay loop muted playsinline width="100%"></video>
    <p align="center"><b>Flipper articulation</b></p>
  </div>
  <div>
    <video src="https://github.com/user-attachments/assets/bc3e81c9-e839-44f7-aa6a-4796e2ed4b31" autoplay loop muted playsinline width="100%"></video>
    <p align="center"><b>Dense obstacle field</b></p>
  </div>
  <div>
    <video src="https://github.com/user-attachments/assets/386a8857-a66a-4715-bb1e-2572b9cfc598" autoplay loop muted playsinline width="100%"></video>
    <p align="center"><b>Large obstacle traversal</b></p>
  </div>
</div>

---

## Getting Started

<div class="grid cards" markdown>

- :material-rocket-launch: **[Installation](getting-started/installation.md)**

    Set up Ostrich and its dependencies

- :material-book-open-variant: **[Quick Start](getting-started/quickstart.md)**

    Run your first simulation

</div>

---

**Aleš Kučera** — [kuceral4@fel.cvut.cz](mailto:kuceral4@fel.cvut.cz)
Czech Technical University in Prague

"""helhest_junior open-loop replay in Project Chrono (pychrono 9.0.1).

Self-contained on purpose: this runs under the chrono-env python
(~/.local/opt/chrono-env/bin/python), NOT the ostrich venv. The host talks to it
through JSON job/result batches (see engines/bridge.py):

    <chrono-env>/bin/python chrono_replay_junior.py --jobs jobs.json --out out.json

The robot model is ported from helhest_stack-tier1/scripts/chrono_vehicle_model.py
(same masses/inertias/geometry as ostrich examples/helhest_junior/common.py) with
three deliberate changes: per-wheel-set contact materials so front (0.7) and rear
(0.4) friction differ; no baked-in motor lag or rolling-resistance torque (the
bridge pre-filters commands with the measured actuator lag, and rolling resistance
is a sweep parameter); and a generic scene list. Landmines inherited from tier1,
measured not reasoned: NSC SetRollingFriction locks chassis yaw — never use it;
each motor keeps ONE ChFunctionConst mutated in place; wheels are 48-gon
circumscribing hull prisms because Bullet cylinder-on-plane is a degenerate
single contact point; envelope/margin 5e-4 or the robot rides ~1.4 mm high.

The logged pose is the chassis reference frame = front-axle center at hub height,
which equals the real robot's base_link (verified against helhest_stack
RobotParams: wheels at [0, ±0.365, 0], rear [-0.75, 0, 0]).
"""

import argparse
import json
import math
import time

import numpy as np
import pychrono as chrono

# --- model constants (tier1 donor, matches ostrich common.py) ---
CHASSIS_MASS = 89.7
CHASSIS_COM_X = -0.188127
CHASSIS_INERTIA = (2.4114, 4.2209, 6.0343)
WHEEL_MASS = 5.5
WHEEL_I_AXIS = 0.336875
WHEEL_I_DIAM = 0.173021
WHEEL_RADIUS = 0.35
HALF_WIDTH = 0.05
HALF_TRACK = 0.365
REAR_OFFSET = 0.75
TOTAL_MASS = 106.2
TOTAL_COM_X = -0.198
HULL_FACETS = 48

WHEELS = {  # sim command order is [left, right, rear]
    "left": (0.0, HALF_TRACK, 0.0),
    "right": (0.0, -HALF_TRACK, 0.0),
    "rear": (-REAR_OFFSET, 0.0, 0.0),
}

DEFAULT_PARAMS = {
    "terrain": "rigid",         # rigid | scm (scm needs the source build: run via chrono_env.sh)
    "system": "nsc",            # nsc | smc
    "solver": "default",        # default(PSOR) | psor | bb | apgd | jacobi
    "iterations": 0,            # 0 = engine default
    "tolerance": 0.0,           # 0 = engine default
    "wheel_shape": "hull48",    # hull48 | cylinder
    "envelope": 5e-4,
    "margin": 5e-4,
    "mu_front": 0.7,            # target PAIR friction wheel<->world
    "mu_rear": 0.4,
    "rolling_resistance": 0.0,  # torque-based, fraction of normal load
    "threads": 1,
    "torque_limit": 0.0,        # 0 = unlimited (motors are ideal velocity servos)
}


def hull_points(radius, half_width, n):
    rr = radius / math.cos(math.pi / n)
    pts = chrono.vector_ChVector3d()
    for i in range(n):
        a = 2.0 * math.pi * i / n
        for s in (-half_width, half_width):
            pts.push_back(chrono.ChVector3d(rr * math.cos(a), s, rr * math.sin(a)))
    return pts


def make_material(system_kind, mu):
    if system_kind == "smc":
        m = chrono.ChContactMaterialSMC()
        m.SetYoungModulus(2e7)
        m.SetPoissonRatio(0.3)
    else:
        m = chrono.ChContactMaterialNSC()
    m.SetFriction(mu)
    m.SetRestitution(0.0)
    return m


def build_robot(sys_, params):
    """Chassis + three speed-driven wheels. Returns (chassis, wheels, motor_fns)."""
    kind = params["system"]
    chassis = chrono.ChBodyAuxRef()
    chassis.SetMass(CHASSIS_MASS)
    chassis.SetFrameCOMToRef(
        chrono.ChFramed(chrono.ChVector3d(CHASSIS_COM_X, 0.0, 0.0), chrono.QUNIT))
    chassis.SetInertiaXX(chrono.ChVector3d(*CHASSIS_INERTIA))
    chassis.SetFrameRefToAbs(
        chrono.ChFramed(chrono.ChVector3d(0, 0, WHEEL_RADIUS), chrono.QUNIT))
    chassis.EnableCollision(False)
    sys_.Add(chassis)

    wheels, motor_fns = {}, {}
    for name, (wx, wy, wz) in WHEELS.items():
        mu = params["mu_rear"] if name == "rear" else params["mu_front"]
        mat = make_material(kind, mu)
        w = chrono.ChBody()
        w.SetMass(WHEEL_MASS)
        w.SetInertiaXX(chrono.ChVector3d(WHEEL_I_DIAM, WHEEL_I_AXIS, WHEEL_I_DIAM))
        w.SetPos(chrono.ChVector3d(wx, wy, wz + WHEEL_RADIUS))
        w.EnableCollision(True)
        if params["wheel_shape"] == "cylinder":
            shape = chrono.ChCollisionShapeCylinder(mat, WHEEL_RADIUS, 2 * HALF_WIDTH)
            # ChCollisionShapeCylinder axis is Z; rotate onto body Y.
            frame = chrono.ChFramed(chrono.ChVector3d(0, 0, 0),
                                    chrono.QuatFromAngleX(math.pi / 2))
        else:
            shape = chrono.ChCollisionShapeConvexHull(
                mat, hull_points(WHEEL_RADIUS, HALF_WIDTH, HULL_FACETS))
            frame = chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT)
        w.AddCollisionShape(shape, frame)
        sys_.Add(w)
        # Motor about its frame Z; Rx(-90) puts that on body +Y so positive
        # omega drives forward (tier1 convention, verified there).
        m = chrono.ChLinkMotorRotationSpeed()
        m.Initialize(w, chassis,
                     chrono.ChFramed(chrono.ChVector3d(wx, wy, wz + WHEEL_RADIUS),
                                     chrono.QuatFromAngleX(-math.pi / 2)))
        fn = chrono.ChFunctionConst(0.0)
        m.SetSpeedFunction(fn)
        sys_.Add(m)
        wheels[name], motor_fns[name] = w, fn

    # Startup self-check against the spec (composed totals).
    total = CHASSIS_MASS + 3 * WHEEL_MASS
    com_x = (CHASSIS_MASS * CHASSIS_COM_X + WHEEL_MASS * (0 + 0 - REAR_OFFSET)) / total
    assert abs(total - TOTAL_MASS) < 1e-6, f"total mass {total} != {TOTAL_MASS}"
    assert abs(com_x - TOTAL_COM_X) < 5e-3, f"com_x {com_x} != {TOTAL_COM_X}"
    return chassis, wheels, motor_fns


def build_scm_terrain(sys_, wheels, params):
    """Bekker-Wong + Janosi-Hanamoto deformable terrain (tier1 parameter set:
    Bekker preset from Chrono's own demo held fixed, Janosi K measured on the
    robot). Moving patches restrict the active soil region to each wheel.
    `scm_phi` (Mohr friction angle, deg) scales the shear strength — the lever
    for the turn gain alpha; `scm_janosi_k` defaults to the measured value."""
    import pychrono.vehicle as veh
    terrain = veh.SCMTerrain(sys_)
    terrain.SetSoilParameters(
        0.2e6,   # Bekker Kphi [Pa/m^n]
        0.0,     # Bekker Kc
        1.1,     # Bekker n
        0.0,     # Mohr cohesion [Pa]
        params.get("scm_phi", 30.0),        # Mohr friction angle [deg]
        params.get("scm_janosi_k", 0.0125),  # Janosi shear K [m] — measured
        4e7,     # elastic stiffness [Pa/m]
        3e4,     # damping [Pa s/m]
    )
    terrain.SetPlane(chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0), chrono.QUNIT))
    terrain.Initialize(100.0, 100.0, 0.05)
    for w in wheels.values():
        terrain.AddMovingPatch(w, chrono.ChVector3d(0, 0, 0),
                               chrono.ChVector3d(0.5, 4 * HALF_WIDTH, 2 * WHEEL_RADIUS))
    return terrain


def build_world(sys_, job, params):
    """Ground plane (optionally tilted about Y) + scene bodies."""
    kind = params["system"]
    # Pair friction under Chrono's default material composition is verified by
    # the tilt-probe check in verify_runner.py; ground gets the front-wheel mu
    # so wheel materials dominate under min-composition.
    ground_mat = make_material(kind, max(params["mu_front"], params["mu_rear"]))
    ground = chrono.ChBodyEasyBox(120, 120, 1.0, 1000, True, True, ground_mat)
    ground.SetPos(chrono.ChVector3d(0, 0, -0.5))
    ground.SetFixed(True)
    sys_.Add(ground)

    scene_bodies = []
    for i, body in enumerate(job.get("scene", [])):
        mat = make_material(kind, body.get("mu", 0.5))
        he = body["half_extents"]
        box = chrono.ChBodyEasyBox(2 * he[0], 2 * he[1], 2 * he[2],
                                   body.get("density", 400.0), True, True, mat)
        box.SetPos(chrono.ChVector3d(*body["pos"]))
        box.SetFixed(body["type"] == "static_box")
        sys_.Add(box)
        scene_bodies.append(box)
    return scene_bodies


def configure_solver(sys_, params):
    name = params["solver"]
    if name in ("default",):
        solver = None
    elif name == "psor":
        solver = chrono.ChSolverPSOR()
    elif name == "bb":
        solver = chrono.ChSolverBB()
    elif name == "apgd":
        solver = chrono.ChSolverAPGD()
    elif name == "jacobi":
        solver = chrono.ChSolverPJacobi()
    else:
        raise ValueError(f"unknown solver {name}")
    if solver is not None:
        if params["iterations"]:
            solver.SetMaxIterations(int(params["iterations"]))
        if params["tolerance"]:
            solver.SetTolerance(params["tolerance"])
        sys_.SetSolver(solver)
    elif params["iterations"]:
        sys_.GetSolver().AsIterative().SetMaxIterations(int(params["iterations"]))


def run_job(job):
    params = {**DEFAULT_PARAMS, **job.get("params", {})}
    dt = job["dt"]

    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(params["envelope"])
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(params["margin"])
    sys_ = chrono.ChSystemSMC() if params["system"] == "smc" else chrono.ChSystemNSC()
    # Incline tests tilt GRAVITY over a flat ground — identical friction physics
    # to a tilted plane, but with no spawn-pose mismatch transient.
    tilt = math.radians(job["ground"].get("tilt_deg", 0.0))
    sys_.SetGravitationalAcceleration(
        chrono.ChVector3d(-9.81 * math.sin(tilt), 0, -9.81 * math.cos(tilt)))
    sys_.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    if params["threads"] > 1:
        sys_.SetNumThreads(int(params["threads"]))
    configure_solver(sys_, params)

    chassis, wheels, motor_fns = build_robot(sys_, params)
    if params["terrain"] == "scm":
        terrain = build_scm_terrain(sys_, wheels, params)
        scene_bodies = []
    else:
        terrain = None
        scene_bodies = build_world(sys_, job, params)
    log_scene = bool(job.get("log_scene_bodies")) and scene_bodies

    order = ["left", "right", "rear"]  # command column order
    cmds = np.asarray(job["control"]["lrr"], dtype=np.float64)
    n_settle = int(round(job.get("settle_time_s", 1.0) / dt))
    n_steps = cmds.shape[0]

    for _ in range(n_settle):
        if terrain is not None:
            terrain.Advance(dt)
        sys_.DoStepDynamics(dt)

    pose = np.empty((n_steps, 7), dtype=np.float64)
    scene_pose = (np.empty((n_steps, len(scene_bodies), 4), dtype=np.float64)
                  if log_scene else None)
    omega = np.empty((n_steps, 3), dtype=np.float64) if job.get("log_wheel_omega") else None
    stable, diverged_at = True, None
    t_start = time.perf_counter()
    for i in range(n_steps):
        for c, name in enumerate(order):
            motor_fns[name].SetConstant(cmds[i, c])
        if params["rolling_resistance"] > 0.0:
            for name, w in wheels.items():
                w.EmptyAccumulators()
                n_force = abs(float(w.GetContactForce().z))
                spin = float(w.GetAngVelLocal().y)
                if n_force > 1.0 and abs(spin) > 1e-6:
                    t_roll = -math.copysign(
                        params["rolling_resistance"] * n_force * WHEEL_RADIUS, spin)
                    w.AccumulateTorque(chrono.ChVector3d(0.0, t_roll, 0.0), True)
        if terrain is not None:
            terrain.Advance(dt)
        sys_.DoStepDynamics(dt)
        f = chassis.GetFrameRefToAbs()
        p, q = f.GetPos(), f.GetRot()  # ChQuaterniond is (w, x, y, z)
        pose[i] = (p.x, p.y, p.z, q.e1, q.e2, q.e3, q.e0)
        if omega is not None:
            omega[i] = [float(wheels[n].GetAngVelLocal().y) for n in order]
        if log_scene:
            for k, b in enumerate(scene_bodies):
                bp, bq = b.GetPos(), b.GetRot()  # yaw from (w,x,y,z)
                yaw = math.atan2(2 * (bq.e0 * bq.e3 + bq.e1 * bq.e2),
                                 1 - 2 * (bq.e2 * bq.e2 + bq.e3 * bq.e3))
                scene_pose[i, k] = (bp.x, bp.y, bp.z, yaw)
        if not np.all(np.isfinite(pose[i])) or abs(p.z) > 5.0:
            stable, diverged_at = False, i * dt
            pose = pose[: i + 1]
            break
    wall = time.perf_counter() - t_start

    return {
        "id": job["id"],
        "dt": dt,
        "pose": pose.tolist(),
        "scene_body_pose": scene_pose[: pose.shape[0]].tolist() if log_scene else None,
        "wheel_omega": omega[: pose.shape[0]].tolist() if omega is not None else None,
        "wall_clock_s": wall,
        "n_steps": int(pose.shape[0]),
        "threads": params["threads"],
        "stable": stable,
        "diverged_at_s": diverged_at,
        "params_effective": params,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.jobs) as f:
        jobs = json.load(f)["jobs"]
    results = []
    for job in jobs:
        print(f"[chrono] {job['id']} ...", flush=True)
        results.append(run_job(job))
    with open(args.out, "w") as f:
        json.dump({"results": results, "engine": "chrono",
                   "engine_version": chrono.CHRONO_VERSION
                   if hasattr(chrono, "CHRONO_VERSION") else "9.0.1"}, f)
    print(f"[chrono] wrote {args.out} ({len(results)} results)")


if __name__ == "__main__":
    main()

"""helhest_junior open-loop replay in AGX Dynamics.

Runs ONLY inside the `agx-env` distrobox container (python 3.14 with the AGX
bindings on PYTHONPATH via setup_env.bash); the host invokes it through
engines/bridge.py:

    distrobox enter agx-env -- bash -lc \\
        'source /opt/Algoryx/AGX-2.42.1.0/setup_env.bash && \\
         python3 agx_replay_junior.py --jobs jobs.json --out out.json'

The robot comes from the existing demo (agx_helhest/demos/helhest_junior.agxPy,
class HelhestJunior — spec-checked masses/geometry identical to ostrich's
common.py). This runner adds what the demo lacks: explicit per-pair contact
materials with configurable friction MODEL (AGX's default IterativeProjectedCone
SPLIT is documented to give viscous sliding friction on wheels; DIRECT is the
manual's recommendation for vehicles — that's a sweep axis), solver iteration
counts, and open-loop command replay with pose logging.

Motor sign: Motor1D drives body2 (wheel) relative to body1 (chassis) about the
hinge +Y axis, so commands are NEGATED (demo convention) to make positive =
forward; the host-side verify checks +X motion. Logged pose = chassis model
frame = front-axle center at hub height = real base_link. agx.Quat is (x,y,z,w).
"""

import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time
from importlib.machinery import SourceFileLoader

import agx
import agxCollide
import agxSDK

DEMO = pathlib.Path.home() / "projects" / "agx_helhest" / "demos" / "helhest_junior.agxPy"

MOTOR_SIGN = -1.0

DEFAULT_PARAMS = {
    "terrain": "rigid",               # rigid | soil (agxTerrain deformable, library preset)
    "soil": "dirt_1",                 # agxTerrain library material (terrain == soil)
    # Oriented (anisotropic) friction: mu_front/mu_rear become the LONGITUDINAL
    # (rolling-direction, chassis +X) coefficients and mu_lat_* the lateral ones.
    "oriented_friction": False,
    "mu_lat_front": 0.3,
    "mu_lat_rear": 0.2,
    "friction_solve_type": "split",   # split | direct | iterative | direct_and_iterative
    "exact_cone_projection": False,   # only meaningful for direct
    "joint_solve_type": "default",    # default | direct | iterative | direct_and_iterative
    "num_resting_iterations": 0,      # 0 = engine default (16)
    "num_friction_iterations": 0,     # 0 = engine default (7)
    "youngs_modulus": 1e9,
    "mu_front": 0.7,                  # pair friction wheel<->world (explicit CM, exact)
    "mu_rear": 0.4,
    "rolling_resistance": 0.0,
    "torque_limit": 250.0,
    "threads": 1,
}

_SOLVE_TYPES = {
    "split": agx.FrictionModel.SPLIT,
    "direct": agx.FrictionModel.DIRECT,
    "iterative": agx.FrictionModel.ITERATIVE,
    "direct_and_iterative": agx.FrictionModel.DIRECT_AND_ITERATIVE,
}


def load_demo_module():
    loader = SourceFileLoader("agx_junior", str(DEMO))
    spec = importlib.util.spec_from_loader("agx_junior", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def make_friction_model(params):
    fm = agx.IterativeProjectedConeFriction()
    fm.setSolveType(_SOLVE_TYPES[params["friction_solve_type"]])
    if params["exact_cone_projection"]:
        fm.setEnableDirectExactConeProjection(True)
    return fm


def configure_contact(cm, mu, params, oriented_frame=None, mu_lat=None):
    cm.setRestitution(0.0)
    cm.setYoungsModulus(params["youngs_modulus"])
    if params["rolling_resistance"] > 0.0:
        cm.setRollingResistanceCoefficient(params["rolling_resistance"])
    if oriented_frame is not None:
        # Anisotropic: primary = chassis local +X (rolling direction; the frame
        # rotates with the body so it tracks heading), secondary = lateral.
        # Same pattern as AGX's own excavator_drivetrain.agxPy tutorial.
        cm.setFrictionCoefficient(mu, agx.ContactMaterial.PRIMARY_DIRECTION)
        cm.setFrictionCoefficient(mu_lat, agx.ContactMaterial.SECONDARY_DIRECTION)
        fm = agx.OrientedIterativeProjectedConeFrictionModel(
            oriented_frame, agx.Vec3.X_AXIS())
        fm.setSolveType(_SOLVE_TYPES[params["friction_solve_type"]])
        cm.setFrictionModel(fm)
    else:
        cm.setFrictionCoefficient(mu)
        cm.setFrictionModel(make_friction_model(params))


def run_job(job, mod):
    params = {**DEFAULT_PARAMS, **job.get("params", {})}
    dt = job["dt"]

    sim = agxSDK.Simulation()
    sim.setTimeStep(dt)
    # Incline tests tilt GRAVITY over a flat ground — identical friction physics
    # to a tilted plane, but with no spawn-pose mismatch transient.
    tilt = math.radians(job["ground"].get("tilt_deg", 0.0))
    sim.setUniformGravity(
        agx.Vec3(-9.81 * math.sin(tilt), 0, -9.81 * math.cos(tilt)))
    if params["threads"] >= 1:
        agx.setNumThreads(int(params["threads"]))
    if params["num_resting_iterations"]:
        sim.getSolver().setNumRestingIterations(int(params["num_resting_iterations"]))
    if params["num_friction_iterations"]:
        sim.getSolver().setNumDryFrictionIterations(int(params["num_friction_iterations"]))

    robot = mod.HelhestJunior(sim, root=None,
                              position=agx.Vec3(0.0, 0.0, mod.JuniorSpec.WHEEL_RADIUS))
    if params["joint_solve_type"] != "default":
        st = {"direct": agx.Constraint.DIRECT,
              "iterative": agx.Constraint.ITERATIVE,
              "direct_and_iterative": agx.Constraint.DIRECT_AND_ITERATIVE}[
                  params["joint_solve_type"]]
        for hinge in robot.motors.values():
            hinge.setSolveType(st)
    for hinge in robot.motors.values():
        hinge.getMotor1D().setForceRange(-params["torque_limit"], params["torque_limit"])

    if params["terrain"] == "soil":
        # agxTerrain deformable soil with a library material preset. Rigid
        # bodies (the wheels) interact with the height field by default — no
        # Shovel needed for driving. Particles/dynamic-mass off: we want
        # sinkage + compaction + shear, not excavation debris.
        import agxTerrain
        terrain = agxTerrain.Terrain(801, 801, 0.1, 0.5)  # 80x80 m, 10 cm cells
        terrain.loadLibraryMaterial(params["soil"])
        if params.get("soil_fast"):
            # Perf switches — but disabling dynamic mass also killed wheel
            # traction entirely in the first smoke test, so default is OFF.
            tp = terrain.getProperties()
            tp.setCreateParticles(False)
            tp.setEnableCreateDynamicMass(False)
        sim.add(terrain)
        ground_mat = terrain.getMaterial()
    else:
        # Ground: large static box.
        ground_mat = agx.Material("ground")
        ground_body = agx.RigidBody("ground")
        ground_body.setMotionControl(agx.RigidBody.STATIC)
        geom = agxCollide.Geometry(agxCollide.Box(agx.Vec3(60, 60, 0.5)))
        geom.setMaterial(ground_mat)
        ground_body.add(geom)
        ground_body.setPosition(agx.Vec3(0, 0, -0.5))
        sim.add(ground_body)

    oriented = robot.chassis.getFrame() if params["oriented_friction"] else None
    mm = sim.getMaterialManager()
    cm_front = mm.getOrCreateContactMaterial(robot.front_material, ground_mat)
    cm_rear = mm.getOrCreateContactMaterial(robot.rear_material, ground_mat)
    configure_contact(cm_front, params["mu_front"], params,
                      oriented_frame=oriented, mu_lat=params["mu_lat_front"])
    configure_contact(cm_rear, params["mu_rear"], params,
                      oriented_frame=oriented, mu_lat=params["mu_lat_rear"])
    if params["terrain"] == "soil":
        # The manual marks this as essential for wheels on agxTerrain: installs
        # the terramechanics TerrainWheelForceModel on the wheel-terrain CM
        # (plain friction models produced zero traction on the soil surface).
        import agxTerrain
        for wheel in robot.wheels.values():
            tw = agxTerrain.TerrainWheel(wheel)
            sim.add(tw)
        agxTerrain.TerrainWheel.configureContactMaterial(cm_front)
        agxTerrain.TerrainWheel.configureContactMaterial(cm_rear)

    scene_bodies = []
    for body in job.get("scene", []):
        mat = agx.Material(f"scene_{id(body)}")
        mat.getBulkMaterial().setDensity(body.get("density", 400.0))
        b = agx.RigidBody()
        if body["type"] == "static_box":
            b.setMotionControl(agx.RigidBody.STATIC)
        he = body["half_extents"]
        g = agxCollide.Geometry(agxCollide.Box(agx.Vec3(he[0], he[1], he[2])))
        g.setMaterial(mat)
        b.add(g)
        b.setPosition(agx.Vec3(*body["pos"]))
        sim.add(b)
        scene_bodies.append(b)
        cm = mm.getOrCreateContactMaterial(mat, ground_mat)
        cm.setFrictionCoefficient(body.get("mu", 0.5))
        cm.setRestitution(0.0)
        for wheel_mat, mu in ((robot.front_material, params["mu_front"]),
                              (robot.rear_material, params["mu_rear"])):
            cmw = mm.getOrCreateContactMaterial(mat, wheel_mat)
            configure_contact(cmw, mu, params)

    order = ["left_wheel", "right_wheel", "rear_wheel"]  # command column order
    cmds = job["control"]["lrr"]
    n_settle = int(round(job.get("settle_time_s", 1.0) / dt))
    n_steps = len(cmds)

    for _ in range(n_settle):
        sim.stepForward()

    chassis = robot.chassis
    pose = []
    omega = [] if job.get("log_wheel_omega") else None
    log_scene = bool(job.get("log_scene_bodies")) and scene_bodies
    scene_pose = [] if log_scene else None
    stable, diverged_at = True, None
    t_start = time.perf_counter()
    for i in range(n_steps):
        row = cmds[i]
        for c, name in enumerate(order):
            robot.motors[name].getMotor1D().setSpeed(MOTOR_SIGN * row[c])
        sim.stepForward()
        p = chassis.getPosition()
        q = chassis.getRotation()
        rec = (p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w())
        pose.append(rec)
        if omega is not None:
            omega.append([MOTOR_SIGN * robot.motors[n].getCurrentSpeed() for n in order])
        if log_scene:
            frame = []
            for b in scene_bodies:
                bp, bq = b.getPosition(), b.getRotation()
                yaw = math.atan2(2 * (bq.w() * bq.z() + bq.x() * bq.y()),
                                 1 - 2 * (bq.y() ** 2 + bq.z() ** 2))
                frame.append((bp.x(), bp.y(), bp.z(), yaw))
            scene_pose.append(frame)
        if not all(math.isfinite(v) for v in rec) or abs(p.z()) > 5.0:
            stable, diverged_at = False, i * dt
            break
    wall = time.perf_counter() - t_start
    sim.cleanup(agxSDK.Simulation.CLEANUP_ALL)

    return {
        "id": job["id"],
        "dt": dt,
        "pose": pose,
        "scene_body_pose": scene_pose,
        "wheel_omega": omega,
        "wall_clock_s": wall,
        "n_steps": len(pose),
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
    init = agx.AutoInit()  # noqa: F841 -- keeps AGX alive for the process
    mod = load_demo_module()
    with open(args.jobs) as f:
        jobs = json.load(f)["jobs"]
    results = []
    for job in jobs:
        print(f"[agx] {job['id']} ...", flush=True)
        results.append(run_job(job, mod))
    with open(args.out, "w") as f:
        json.dump({"results": results, "engine": "agx",
                   "engine_version": agx.agxGetVersion()}, f)
    print(f"[agx] wrote {args.out} ({len(results)} results)")


if __name__ == "__main__":
    main()

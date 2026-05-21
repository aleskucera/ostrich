"""
Stacked-boxes stability benchmark — Dojo (Julia differentiable physics).

Scene: 3-box inverted pyramid (Z-up convention, ground at z=0).
  Box 1 (bottom): hx=0.2 m, density=1500 kg/m³  →   96 kg
  Box 2 (middle): hx=0.8 m, density=1500 kg/m³  → 6144 kg
  Box 3 (top):    hx=1.6 m, density=1500 kg/m³  → 49152 kg
  Mass ratio top/bottom ≈ 512:1.

Contacts are modelled as corner-point contacts:
  • box1 bottom (4 corners) on the implicit ground plane (z = 0)
  • box1 top (4 corners) touching box2's bottom face
  • box2 top (4 corners) touching box3's bottom face

Maximal state per body: [x(3), v(3), q(4), ω(3)] = 13 elements.
Full mechanism state: [box1(13), box2(13), box3(13)] = 39 elements.
  box3 z-position: state[27 + 2] = state[29]  (1-indexed)

Usage:
    ~/.juliaup/bin/julia +1.10 --startup-file=no \\
        examples/comparison/stacked_boxes/dojo.jl \\
        --save examples/comparison/stacked_boxes/results/dojo.json
"""

using Dojo
using LinearAlgebra
using Printf
using JSON

# ======================== Constants ========================

const DURATION       = 3.0
const DENSITY        = 500.0    # kg/m³ — must match config.py

const HX1 = 0.3
const HX2 = 0.6
const HX3 = 1.2

const Z1 = HX1
const Z2 = 2*HX1 + HX2
const Z3 = 2*HX1 + 2*HX2 + HX3   # = 3.6 m

const STABILITY_TOL  = 0.1        # m — allowed Z deviation of top box
const MU             = 0.1        # friction coefficient

const BSEARCH_TOL    = 0.001
const BSEARCH_MAX    = 2.0
const DT_VALUES      = [0.001, 0.005, 0.01, 0.05, 0.1]

# ======================== Mass / inertia helpers ========================

mass(hx)    = DENSITY * (2*hx)^3
inertia(hx) = mass(hx) * (2*hx)^2 / 6.0   # solid cube, per axis

# ======================== Build mechanism ========================

"""
Build a Dojo Mechanism for the stacked-boxes scene.

Contact layout:
  - 4 corners of box1 bottom  → ground (z=0)
  - 4 corners of box1 top     → bottom face of box2  (body-body)
  - 4 corners of box2 top     → bottom face of box3  (body-body)
"""
function build_mechanism(; timestep=0.01, gravity=-9.81)
    origin = Origin{Float64}()

    # --- Bodies ---
    box1 = Box(HX1, HX1, HX1, mass(HX1); name=:box1)
    box2 = Box(HX2, HX2, HX2, mass(HX2); name=:box2)
    box3 = Box(HX3, HX3, HX3, mass(HX3); name=:box3)

    # --- Joints (all free-floating) ---
    j1 = JointConstraint(Floating(origin, box1; spring=0.0, damper=0.0), name=:j1)
    j2 = JointConstraint(Floating(origin, box2; spring=0.0, damper=0.0), name=:j2)
    j3 = JointConstraint(Floating(origin, box3; spring=0.0, damper=0.0), name=:j3)

    # --- Contacts ---
    contacts = Vector{ContactConstraint{Float64}}()

    # box1 bottom 4 corners → ground (z=0 implicit plane, normal [0,0,1])
    for (sx, sy) in [(1,1), (-1,1), (-1,-1), (1,-1)]
        push!(contacts, contact_constraint(box1, [0.0, 0.0, 1.0];
            friction_coefficient = MU,
            contact_origin       = [sx*HX1, sy*HX1, -HX1],
            contact_radius       = 0.0,
            contact_type         = :nonlinear,
            name                 = Symbol("box1_g_$(sx)_$(sy)")))
    end

    # box1 top 4 corners → box2 (SphereBoxCollision: corner of box1 = sphere r=0, box2 = box)
    friction_param = [1.0 0.0; 0.0 1.0]
    for (sx, sy) in [(1,1), (-1,1), (-1,-1), (1,-1)]
        collision = SphereBoxCollision{Float64,2,3,6}(
            [sx*HX1, sy*HX1, HX1],   # contact origin on box1 (top corner, body frame)
            2*HX2, 2*HX2, 2*HX2,     # full dimensions of box2
            0.0,                       # point contact (radius = 0)
        )
        model = NonlinearContact{Float64,8}(MU, friction_param, collision)
        push!(contacts, ContactConstraint((model, box1.id, box2.id);
            name=Symbol("box1_box2_$(sx)_$(sy)")))
    end

    # box2 top 4 corners → box3 (SphereBoxCollision: corner of box2 = sphere r=0, box3 = box)
    for (sx, sy) in [(1,1), (-1,1), (-1,-1), (1,-1)]
        collision = SphereBoxCollision{Float64,2,3,6}(
            [sx*HX2, sy*HX2, HX2],   # contact origin on box2 (top corner, body frame)
            2*HX3, 2*HX3, 2*HX3,     # full dimensions of box3
            0.0,
        )
        model = NonlinearContact{Float64,8}(MU, friction_param, collision)
        push!(contacts, ContactConstraint((model, box2.id, box3.id);
            name=Symbol("box2_box3_$(sx)_$(sy)")))
    end

    return Mechanism(origin, [box1, box2, box3], [j1, j2, j3], contacts;
        gravity, timestep)
end

# ======================== State helpers ========================

"""
Build the 39-element maximal initial state for 3 boxes at rest.
Layout per body: [x(3), v(3), q(4), ω(3)] (quaternion: [w, x, y, z]).
Body order in mechanism: [box1, box2, box3].

A small vertical gap (GAP) is added above each contact surface so the
initial gap is positive (not degenerate) and gravity closes it naturally.
"""
const GAP = 1e-3   # 1 mm initial separation at each interface

function initial_state()
    z = zeros(39)
    # box1: bottom at z = GAP (sits just above the ground)
    z[1:3]   = [0.0, 0.0, Z1 + GAP]
    z[7]     = 1.0
    # box2: bottom at z = 2*HX1 + 2*GAP
    z[14:16] = [0.0, 0.0, Z2 + 2*GAP]
    z[20]    = 1.0
    # box3: bottom at z = 2*HX1 + 2*HX2 + 3*GAP
    z[27:29] = [0.0, 0.0, Z3 + 3*GAP]
    z[33]    = 1.0
    return z
end

"""Extract box3 z-position from full maximal state (1-indexed)."""
box3_z(z::Vector) = z[29]   # box3 starts at index 27; z-component is offset +2

# ======================== Forward rollout ========================

function run_one(dt; verbose=false)
    mech  = build_mechanism(timestep=dt)
    T     = max(1, floor(Int, DURATION / dt))
    z     = initial_state()
    u     = zeros(input_dimension(mech))

    times   = Float64[]
    heights = Float64[]
    stable  = true

    for step in 1:T
        z = step!(mech, z, u)

        t = step * dt
        h = box3_z(z)

        if !isfinite(h)
            stable = false
            push!(times,   t)
            push!(heights, NaN)
            break
        end

        if abs(h - Z3) > STABILITY_TOL
            stable = false
        end

        push!(times,   t)
        push!(heights, h)

        if verbose
            @printf("  step %4d  t=%.3fs  box3_z=%.4f  %s\n",
                step, t, h, stable ? "" : "UNSTABLE")
        end
    end

    return Dict(
        "dt"            => dt,
        "T"             => T,
        "time"          => times,
        "top_box_height" => heights,
        "stable"        => stable,
    )
end

# ======================== Binary search ========================

function find_threshold()
    lo      = 0.0005
    hi      = nothing
    probe   = lo * 2.0
    n_evals = 0

    while probe <= BSEARCH_MAX
        n_evals += 1
        @printf("  probe dt=%.4fs ...", probe)
        r = run_one(probe)
        if r["stable"]
            println(" stable")
            lo    = probe
            probe *= 2.0
        else
            println(" UNSTABLE")
            hi = probe
            break
        end
    end

    if hi === nothing
        @printf("  Stable up to %.4fs (BSEARCH_MAX reached)\n", lo)
        return Dict("max_stable_dt" => lo, "n_evals" => n_evals, "hit_max" => true)
    end

    while hi - lo > BSEARCH_TOL
        mid = (lo + hi) / 2.0
        n_evals += 1
        @printf("  bisect dt=%.4fs ...", mid)
        r = run_one(mid)
        if r["stable"]
            println(" stable")
            lo = mid
        else
            println(" UNSTABLE")
            hi = mid
        end
    end

    @printf("  → max stable dt = %.4fs  (n_evals=%d)\n", lo, n_evals)
    return Dict("max_stable_dt" => lo, "n_evals" => n_evals, "hit_max" => false)
end

# ======================== Main ========================

function main()
    experiment = "dt_sweep"
    save_path  = nothing

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--experiment" && i < length(ARGS)
            experiment = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--save" && i < length(ARGS)
            save_path = ARGS[i+1]; i += 2
        else
            i += 1
        end
    end

    base = Dict(
        "simulator"     => "Dojo",
        "problem"       => "stacked_boxes",
        "density"       => DENSITY,
        "hx1"           => HX1,
        "hx2"           => HX2,
        "hx3"           => HX3,
        "z_top_initial" => Z3,
        "stability_tol" => STABILITY_TOL,
    )

    results = nothing

    if experiment == "binary_search"
        println("Dojo — binary_search:")
        threshold = find_threshold()
        results = merge(base, threshold)
    else
        println("Dojo stacked-boxes stability sweep:")
        runs = []
        for dt in DT_VALUES
            T = max(1, floor(Int, DURATION / dt))
            @printf("  dt=%.3fs  (T=%d steps)...", dt, T)
            flush(stdout)
            run = run_one(dt)
            push!(runs, run)
            if run["stable"]
                println(" STABLE")
            else
                @printf(" UNSTABLE at t=%.3fs\n", run["time"][end])
            end
        end
        results = merge(base, Dict("runs" => runs))
    end

    if save_path !== nothing
        mkpath(dirname(abspath(save_path)))
        open(save_path, "w") do io
            JSON.print(io, results, 2)
        end
        println("Saved to $save_path")
    end
end

main()

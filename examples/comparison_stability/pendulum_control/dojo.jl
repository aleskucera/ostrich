"""
Control stability benchmark — Dojo (Julia differentiable physics).

Scene: single pendulum, pivot fixed at (0,0,2), 1m link, 1 kg.  Z-up.
       Start at q=0 rad (hanging). Target: q=π/3 rad.

PD control applied as explicit joint torque each step:
    τ = kp * (Q_TARGET - q) - kd * qd

Joint setup (Revolute about Y axis):
  parent_vertex = [0.0, 0.0, 2.0]  — pivot location in world frame
  child_vertex  = [0.0, 0.0, 0.5]  — pivot in body frame (top end of 1m rod)
  At q=0: body CoM at (0, 0, 1.5), rod hangs from z=2 to z=1.

Maximal state per body: [x(3), v(3), q(4), ω(3)] = 13 elements.
  Quaternion convention: [w, x, y, z].
  Rotation about Y by angle θ: q = [cos(θ/2), 0, sin(θ/2), 0].
  ⇒ joint angle: θ = 2*atan(z[9], z[7])    (z[9]=sin(θ/2), z[7]=cos(θ/2))
  ⇒ joint rate:  θ̇ = z[12]                 (ω_y component)

Input dimension for Revolute joint: 1 (joint torque about axis).

Three experiments:
  dt_sweep      — fix kp=1000, kd=25; sweep Δt ∈ {0.001, 0.005, 0.01, 0.05, 0.1}
  gain_sweep    — fix Δt=0.05; sweep kp ∈ {10, 50, 100, 200, 500, 1000, 5000}
  binary_search — fix kp=1000, kd=25; binary-search for the largest stable Δt

Usage:
    ~/.juliaup/bin/julia +1.10 --startup-file=no \\
        examples/comparison/control_stability/dojo.jl \\
        --experiment dt_sweep \\
        --save examples/comparison/control_stability/results/dojo_dt.json
"""

using Dojo
using LinearAlgebra
using Printf
using JSON

# ======================== Constants ========================

const DURATION      = 3.0
const LINK_LENGTH   = 1.0
const LINK_HEIGHT   = 0.1     # cross-section (box link)
const LINK_MASS     = 1.0
const LINK_INERTIA  = LINK_MASS * LINK_LENGTH^2 / 3.0  # about pivot end

const Q_INIT        = 0.0
const Q_TARGET      = π / 3
const STABILITY_TOL = π

const DT_SWEEP_KP   = 1000.0
const DT_SWEEP_KD   = 25.0
const DT_VALUES     = [0.001, 0.005, 0.01, 0.05, 0.1]

const GAIN_SWEEP_DT = 0.05
const KP_VALUES     = [10, 50, 100, 200, 500, 1000, 5000]

const BSEARCH_KP    = DT_SWEEP_KP
const BSEARCH_KD    = DT_SWEEP_KD
const BSEARCH_MAX   = 2.0
const BSEARCH_TOL   = 0.002
const BSEARCH_DIVERGE = π

# ======================== Helpers ========================

kd_from_kp(kp) = 2.0 * sqrt(kp * LINK_INERTIA)

# ======================== Build mechanism ========================

"""
Build a Dojo pendulum Mechanism: box link on a Revolute joint about Y axis.
Pivot at world (0,0,2); body CoM at (0,0,1.5) when q=0 (hanging).
"""
function build_pendulum(; timestep=0.01, gravity=-9.81)
    origin = Origin{Float64}()

    # Box link: half-extents (length/2, height/2, height/2)
    link = Box(LINK_LENGTH / 2, LINK_HEIGHT / 2, LINK_HEIGHT / 2, LINK_MASS; name=:link)

    # Revolute joint about Y axis
    # parent_vertex: pivot position in world (Origin) frame
    # child_vertex: pivot position in link's body frame (top end of rod)
    j = JointConstraint(
        Revolute(origin, link, Y_AXIS;
            parent_vertex = [0.0, 0.0, 2.0],
            child_vertex  = [0.0, 0.0, LINK_LENGTH / 2],
            spring        = 0.0,
            damper        = 0.0,
        ),
        name = :pivot,
    )

    return Mechanism(origin, [link], [j]; gravity, timestep)
end

# ======================== State helpers ========================

"""
Initial maximal state: link hanging straight down at q=0.
Body CoM at (0, 0, 1.5), quaternion identity, zero velocities.
"""
function initial_state()
    z = zeros(13)
    z[1:3]   = [0.0, 0.0, 1.5]         # CoM position
    z[4:6]   = zeros(3)                  # linear velocity
    z[7:10]  = [1.0, 0.0, 0.0, 0.0]    # quaternion [w, x, y, z] = identity
    z[11:13] = zeros(3)                  # angular velocity
    return z
end

"""
Extract joint angle from maximal state.
Rotation about Y by θ: q = [cos(θ/2), 0, sin(θ/2), 0].
  z[7]=w=cos(θ/2),  z[9]=y=sin(θ/2)  →  θ = 2*atan(z[9], z[7]).
"""
joint_angle(z::Vector) = 2.0 * atan(z[9], z[7])

"""Extract joint angular velocity (ω_y) from maximal state."""
joint_rate(z::Vector) = z[12]   # ω_y: angular velocity about Y

# ======================== Forward rollout ========================

function run_one(dt, kp, kd; verbose=false)
    mech = build_pendulum(timestep=dt)
    T    = max(1, floor(Int, DURATION / dt))

    z0 = initial_state()
    set_maximal_state!(mech, z0)
    z = copy(z0)

    nu = input_dimension(mech)  # = 1 for Revolute joint

    times   = Float64[]
    angles  = Float64[]
    stable  = true

    for step in 1:T
        # PD torque in joint space
        q_now  = joint_angle(z)
        qd_now = joint_rate(z)
        torque = kp * (Q_TARGET - q_now) - kd * qd_now
        u      = [torque]

        # Advance one timestep (gradients computed but discarded — forward only)
        get_maximal_gradients!(mech, z, u)
        z = get_maximal_state(mech)

        t     = step * dt
        q_new = joint_angle(z)

        if !isfinite(q_new)
            stable = false
            push!(times,  t)
            push!(angles, NaN)
            break
        end

        if abs(q_new) > STABILITY_TOL
            stable = false
        end

        push!(times,  t)
        push!(angles, q_new)

        if verbose
            @printf("  step %4d  t=%.3fs  q=%.4f rad  τ=%.2f  %s\n",
                step, t, q_new, torque, stable ? "" : "UNSTABLE")
        end
    end

    return Dict(
        "dt"          => dt,
        "T"           => T,
        "kp"          => kp,
        "kd"          => kd,
        "time"        => times,
        "joint_angle" => angles,
        "stable"      => stable,
    )
end

# ======================== Threshold helper ========================

function is_stable_threshold(dt, kp, kd)
    r = run_one(dt, kp, kd)
    angs = filter(isfinite, r["joint_angle"])
    isempty(angs) && return false
    return maximum(abs, angs) < BSEARCH_DIVERGE
end

function find_threshold(kp, kd)
    lo      = 0.001
    hi      = nothing
    probe   = lo * 2.0
    n_evals = 0

    while probe <= BSEARCH_MAX
        n_evals += 1
        stable = is_stable_threshold(probe, kp, kd)
        @printf("  probe dt=%.4fs → %s\n", probe, stable ? "STABLE" : "UNSTABLE")
        if !stable
            hi = probe
            break
        end
        lo    = probe
        probe *= 2.0
    end

    if hi === nothing
        return Dict("max_stable_dt" => lo, "n_evals" => n_evals, "hit_max" => true)
    end

    while hi - lo > BSEARCH_TOL
        mid = (lo + hi) / 2.0
        n_evals += 1
        stable = is_stable_threshold(mid, kp, kd)
        @printf("  bisect dt=%.4fs → %s\n", mid, stable ? "STABLE" : "UNSTABLE")
        if stable
            lo = mid
        else
            hi = mid
        end
    end

    return Dict("max_stable_dt" => lo, "n_evals" => n_evals, "hit_max" => false)
end

# ======================== Main ========================

function main()
    experiment = "binary_search"
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

    results = Dict{String,Any}(
        "simulator"  => "Dojo",
        "experiment" => experiment,
        "runs"       => [],
    )

    if experiment == "dt_sweep"
        kp, kd = DT_SWEEP_KP, DT_SWEEP_KD
        @printf("Dojo — dt_sweep (kp=%.0f, kd=%.0f):\n", kp, kd)
        for dt in DT_VALUES
            T = max(1, floor(Int, DURATION / dt))
            @printf("  dt=%.3fs (T=%d) ...", dt, T); flush(stdout)
            run = run_one(dt, kp, kd)
            push!(results["runs"], run)
            if run["stable"]
                println(" STABLE")
            else
                angs = filter(isfinite, run["joint_angle"])
                max_a = isempty(angs) ? 0.0 : maximum(abs, angs)
                @printf(" UNSTABLE (max|θ|=%.2f rad)\n", max_a)
            end
        end

    elseif experiment == "gain_sweep"
        dt = GAIN_SWEEP_DT
        @printf("Dojo — gain_sweep (dt=%.2fs):\n", dt)
        for kp in KP_VALUES
            kd = kd_from_kp(kp)
            @printf("  kp=%5d  kd=%.1f ...", kp, kd); flush(stdout)
            run = run_one(dt, kp, kd)
            push!(results["runs"], run)
            if run["stable"]
                println(" STABLE")
            else
                angs = filter(isfinite, run["joint_angle"])
                max_a = isempty(angs) ? 0.0 : maximum(abs, angs)
                @printf(" UNSTABLE (max|θ|=%.2f rad)\n", max_a)
            end
        end

    else  # binary_search
        kp, kd = BSEARCH_KP, BSEARCH_KD
        @printf("Dojo — binary_search (kp=%.0f, kd=%.0f):\n", kp, kd)
        threshold = find_threshold(kp, kd)
        results["max_stable_dt"] = threshold["max_stable_dt"]
        results["n_evals"]       = threshold["n_evals"]
        results["hit_max"]       = threshold["hit_max"]
        results["kp"]            = kp
        results["kd"]            = kd
        suffix = threshold["hit_max"] ? "hit BSEARCH_MAX" : "$(threshold["n_evals"]) evals"
        @printf("  => max_stable_dt = %.4fs (%s)\n", threshold["max_stable_dt"], suffix)
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

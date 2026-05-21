"""
Ball throw optimization using Dojo (Julia differentiable physics).

Comparable to examples/comparison/ball_throw/ball_throw_axion.py.

Optimizes the initial 3D linear velocity of a ball to match a target trajectory.
Gradients flow through BPTT via get_maximal_gradients!.

Maximal state for a single free sphere: [x(3), v(3), q(4), ω(3)] = 13 elements.
Attjac state (used in Jacobians):       [x(3), v(3), φ(3), ω(3)] = 12 elements.
So adj[4:6] = ∂L/∂v0 after the full backward pass.

Usage:
    ~/.juliaup/bin/julia +1.10 --startup-file=no examples/comparison/ball_throw/dojo.jl \\
        --save examples/comparison/ball_throw/results/dojo.json
"""

using Dojo
using LinearAlgebra
using Printf

# ======================== Constants ========================

const DT          = 3e-2
const T_STEPS     = 50        # 1.5 s total
const BALL_RADIUS = 0.2
const BALL_MASS   = 1.0
const BALL_INIT_Z = 1.0       # ball center height at t=0

const INIT_VEL   = [0.0, 2.0, 1.0]   # initial guess  [vx, vy, vz]
const TARGET_VEL = [0.0, 4.0, 7.0]   # target initial velocity

const LEARNING_RATE = 0.02
const MAX_GRAD      = 100.0

# ======================== Build mechanism ========================

function build_ball(; timestep=DT, gravity=-9.81)
    origin = Origin{Float64}()
    ball   = Sphere(BALL_RADIUS, BALL_MASS; name=:ball)
    j      = JointConstraint(Floating(origin, ball; spring=0.0, damper=0.0), name=:base_joint)

    # Sphere contact: single contact point at body center, radius = BALL_RADIUS.
    # Contact activates when ball center z ≤ BALL_RADIUS (ball touches z=0 ground).
    c = contact_constraint(ball, [0.0, 0.0, 1.0];
            friction_coefficient = 0.7,
            contact_origin       = [0.0, 0.0, 0.0],
            contact_radius       = BALL_RADIUS,
            contact_type         = :nonlinear,
            name                 = :ball_ground)

    return Mechanism(origin, [ball], [j], [c]; gravity, timestep)
end

# ======================== State utilities ========================

# Maximal state layout: [x(3), v(3), q(4), ω(3)] = 13 elements
function initial_maximal_state(vel::Vector{Float64})
    z = zeros(13)
    z[1:3]   = [0.0, 0.0, BALL_INIT_Z]  # position
    z[4:6]   = vel                        # linear velocity
    z[7:10]  = [1.0, 0.0, 0.0, 0.0]     # quaternion (identity)
    z[11:13] = zeros(3)                  # angular velocity
    return z
end

ball_xyz(z::Vector) = z[1:3]  # ball position in maximal state

# ======================== Rollout ========================

"""
Run forward rollout, storing states and per-step Jacobians ∂z_{t+1}/∂z_t (attjac form).
No external inputs are applied (ball is in free flight under gravity + contact).
"""
function rollout_with_gradients(mech::Mechanism, z0::Vector{Float64})
    nu = input_dimension(mech)  # 6 for Floating joint (base force/torque), all zero
    u  = zeros(nu)

    zs  = Vector{Vector{Float64}}(undef, T_STEPS + 1)
    Jzs = Vector{Matrix{Float64}}(undef, T_STEPS)

    zs[1] = z0
    set_maximal_state!(mech, z0)

    z = copy(z0)
    for t in 1:T_STEPS
        Jz, _ = get_maximal_gradients!(mech, z, u)
        z       = get_maximal_state(mech)
        zs[t+1] = z
        Jzs[t]  = Jz
    end

    return zs, Jzs
end

# ======================== Loss and BPTT gradient ========================

"""
Compute trajectory loss and gradient w.r.t. initial linear velocity via BPTT.

Loss = Σ_t ||xyz_t - xyz_target_t||²

After propagating the adjoint backward through all Jzs, adj holds ∂L/∂z0 in
attjac space. The attjac ordering for a free sphere is [x(3), v(3), φ(3), ω(3)],
so adj[4:6] = ∂L/∂v0.
"""
function loss_and_grad(zs, Jzs, target_xyz)
    L   = 0.0
    adj = zeros(12)  # attjac: 12 = 13 - 1 (quaternion 4D → rotation 3D)

    for t in T_STEPS:-1:1
        delta     = ball_xyz(zs[t+1]) - target_xyz[t+1]
        L        += dot(delta, delta)
        adj[1:3] .+= 2.0 .* delta
        adj       = Jzs[t]' * adj
    end

    # adj is now ∂L/∂z0 in attjac space; linear velocity is at indices 4:6
    grad_v0 = adj[4:6]
    return L, grad_v0
end

# ======================== Main ========================

function main()
    save_path = nothing
    for i in 1:length(ARGS)-1
        if ARGS[i] == "--save"
            save_path = ARGS[i+1]
        end
    end

    mech = build_ball()

    println("Ball throw mechanism:")
    println("  Bodies:    ", length(mech.bodies))
    println("  Joints:    ", length(mech.joints))
    println("  Contacts:  ", length(mech.contacts))
    println("  Input dim: ", input_dimension(mech), "  (6 base forces/torques, all zero)")
    println("  T=$T_STEPS steps, dt=$DT s")

    # --- Target episode ---
    println("\nSimulating target episode...")
    z0_target   = initial_maximal_state(TARGET_VEL)
    target_zs, _ = rollout_with_gradients(mech, z0_target)
    target_xyz   = [ball_xyz(target_zs[t]) for t in 1:T_STEPS+1]
    p = target_xyz[end]
    @printf("  Final XYZ: (%.3f, %.3f, %.3f)\n", p[1], p[2], p[3])

    # --- Optimization ---
    vel = copy(INIT_VEL)
    println("\nOptimizing: lr=$LEARNING_RATE (gradient descent, max_grad=$MAX_GRAD)")

    iters_log = Int[]
    loss_log  = Float64[]
    time_log  = Float64[]

    for i in 1:30
        t0 = time()
        z0 = initial_maximal_state(vel)
        zs, Jzs = rollout_with_gradients(mech, z0)

        L, grad_v0 = loss_and_grad(zs, Jzs, target_xyz)
        grad_clamped = clamp.(grad_v0, -MAX_GRAD, MAX_GRAD)
        vel = vel .- LEARNING_RATE .* grad_clamped

        t_iter = (time() - t0) * 1000
        @printf("Iter %3d: loss=%.4f | vel=(%.3f, %.3f, %.3f) | grad=(%.3f, %.3f, %.3f) | t=%.0fms\n",
            i, L, vel[1], vel[2], vel[3], grad_v0[1], grad_v0[2], grad_v0[3], t_iter)
        push!(iters_log, i - 1)
        push!(loss_log,  L)
        push!(time_log,  t_iter)

        L < 1e-4 && (println("Converged!"); break)
    end

    if save_path !== nothing
        mkpath(dirname(abspath(save_path)))
        open(save_path, "w") do io
            println(io, "{")
            println(io, "  \"simulator\": \"Dojo\",")
            println(io, "  \"problem\": \"ball_throw\",")
            println(io, "  \"dt\": $DT,")
            println(io, "  \"T\": $T_STEPS,")
            println(io, "  \"iterations\": $(iters_log),")
            println(io, "  \"loss\": $(loss_log),")
            println(io, "  \"time_ms\": $(time_log)")
            println(io, "}")
        end
        println("Saved to $save_path")
        return
    end

    # --- Visualize optimized trajectory ---
    println("\nVisualizing optimized trajectory (requires Julia +1.10)...")
    nu     = input_dimension(mech)
    ctrl_fn = (m, k) -> set_input!(m, zeros(nu))
    z0_opt = initial_maximal_state(vel)
    set_maximal_state!(mech, z0_opt)
    storage = simulate!(mech, T_STEPS * DT, ctrl_fn; record=true, verbose=false)
    vis = visualize(mech, storage)
    println("Open http://127.0.0.1:8700 in your browser.")
    println("Press Ctrl+C to stop the server.")
    while true; sleep(1); end
end

main()

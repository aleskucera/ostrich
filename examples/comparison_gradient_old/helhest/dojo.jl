"""
Helhest trajectory optimization using Dojo (Julia differentiable physics).

Comparable to examples/comparison/helhest_mjx.py and helhest_axion.py.

Optimizes K spline control points linearly interpolated to per-timestep wheel
velocities. Gradients flow through Dojo's implicit differentiation (BPTT via
get_maximal_gradients!).

Actuation model: each Revolute wheel joint has damper=KV, so applying input
u = KV * target_vel gives effective torque τ_net = KV*(target_vel - ω_wheel),
which is a velocity servo matching MJX's <velocity kv="100"/> actuator.

Usage (requires Julia 1.10 for MeshCat visualization):
    ~/.juliaup/bin/julia +1.10 --startup-file=no examples/comparison/helhest/dojo.jl \\
        --save examples/comparison/helhest/results/dojo.json
"""

using Dojo
using LinearAlgebra
using Printf

# ======================== Constants ========================

const DT       = 0.03           # timestep (s); matches debug script for stable contact
const DURATION = 3.0
const T_STEPS  = Int(DURATION / DT)   # 300 steps
const K        = 30             # spline control points
const KV       = 100.0          # velocity servo gain (= Revolute joint damper coeff)

const WHEEL_RADIUS   = 0.36
const WHEEL_WIDTH    = 0.11
const WHEEL_MASS     = 5.5
const CHASSIS_INIT_Z = WHEEL_RADIUS  # initial chassis height: wheel center exactly on ground

# Input vector layout: u = [base_force(3), base_torque(3), τ_left, τ_right, τ_rear]
# Floating joint → 6 inputs, each Revolute wheel joint → 1 input
const WHEEL_DOF_OFFSET = 7      # 1-indexed: wheel torque inputs start here
const NUM_WHEEL_DOFS   = 3

const TARGET_CTRL = [1.0, 6.0, 0.0]   # target wheel velocities [left, right, rear]
const INIT_CTRL   = [2.0, 5.0, 0.0]   # initial spline guess

const TRAJECTORY_WEIGHT     = 10.0
const REGULARIZATION_WEIGHT = 1e-7

# ======================== Build mechanism ========================

function build_helhest(; timestep=DT, gravity=-9.81,
        friction_front=0.7, friction_rear=0.35)
    origin = Origin{Float64}()

    # Chassis: lump all fixed components (battery, motors, wheel holders) into one box
    chassis_mass = 85.0 + 2.0 + 7.0 + 7.0 + 7.0 + 3.0 + 3.0  # 114 kg total
    chassis = Box(0.26, 0.6, 0.18, chassis_mass;
        color=RGBA(0.5, 0.5, 0.5), name=:chassis)

    # Wheels as Capsule(r, h): smoother contact geometry than Cylinder.
    # Capsule axis is body-Z by default; rotate 90° about X → axis along body-Y.
    # The two hemispherical caps sit at [0, ±h/2, 0] in body frame after rotation.
    wheel_orientation = Quaternion(cos(π/4), sin(π/4), 0.0, 0.0)   # RotX(π/2)
    make_wheel = (nm) -> Capsule(WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS;
        orientation_offset=wheel_orientation,
        color=RGBA(0.15, 0.15, 0.15), name=nm)

    left_wheel  = make_wheel(:left_wheel)
    right_wheel = make_wheel(:right_wheel)
    rear_wheel  = make_wheel(:rear_wheel)
    bodies = [chassis, left_wheel, right_wheel, rear_wheel]

    # Floating base joint (6-DOF free body for chassis)
    j_base = JointConstraint(Floating(origin, chassis;
        spring=0.0, damper=0.0), name=:base_joint)

    # Revolute wheel joints: axis=Y in chassis frame, damper=KV for velocity servo
    left_pos  = [0.0,    0.36, 0.0]
    right_pos = [0.0,   -0.36, 0.0]
    rear_pos  = [-0.697, 0.0,  0.0]

    j_left  = JointConstraint(Revolute(chassis, left_wheel,  Y_AXIS;
        parent_vertex=left_pos,  child_vertex=zeros(3),
        spring=0.0, damper=KV), name=:left_wheel_j)
    j_right = JointConstraint(Revolute(chassis, right_wheel, Y_AXIS;
        parent_vertex=right_pos, child_vertex=zeros(3),
        spring=0.0, damper=KV), name=:right_wheel_j)
    j_rear  = JointConstraint(Revolute(chassis, rear_wheel,  Y_AXIS;
        parent_vertex=rear_pos,  child_vertex=zeros(3),
        spring=0.0, damper=KV), name=:rear_wheel_j)
    joints = [j_base, j_left, j_right, j_rear]

    # Capsule contacts: 2 sphere-halfspace contacts per wheel at the cap centers.
    # contact_origin = [0, ±W/2, 0] (cap endpoints in body frame after RotX(π/2)),
    # contact_radius = WHEEL_RADIUS → signed_dist = z_wheel - R (zero when on ground).
    normal = [0.0, 0.0, 1.0]
    function wheel_contacts(wheel, friction, names)
        origins = [[0.0,  WHEEL_WIDTH/2, 0.0],
                   [0.0, -WHEEL_WIDTH/2, 0.0]]
        contact_constraint(wheel, fill(normal, 2);
            friction_coefficients = fill(friction, 2),
            contact_origins       = origins,
            contact_radii         = fill(WHEEL_RADIUS, 2),
            contact_type          = :nonlinear,
            names                 = names)
    end

    c_left  = wheel_contacts(left_wheel,  friction_front, [:left_c1,  :left_c2])
    c_right = wheel_contacts(right_wheel, friction_front, [:right_c1, :right_c2])
    c_rear  = wheel_contacts(rear_wheel,  friction_rear,  [:rear_c1,  :rear_c2])

    return Mechanism(origin, bodies, joints, [c_left; c_right; c_rear]; gravity, timestep)
end

# ======================== State utilities ========================

# Maximal state layout per body: [x(3), v(3), q(4), ω(3)] = 13 elements
# Body order: chassis(1), left_wheel(2), right_wheel(3), rear_wheel(4)

function initial_maximal_state()
    positions = [
        [0.0,    0.0,  CHASSIS_INIT_Z],   # chassis
        [0.0,    0.36, CHASSIS_INIT_Z],   # left_wheel
        [0.0,   -0.36, CHASSIS_INIT_Z],   # right_wheel
        [-0.697, 0.0,  CHASSIS_INIT_Z],   # rear_wheel
    ]
    z = zeros(13 * length(positions))
    for (i, pos) in enumerate(positions)
        idx = 13 * (i - 1)
        z[idx+1:idx+3]   = pos           # position
        z[idx+4:idx+6]   = zeros(3)      # linear velocity
        z[idx+7:idx+10]  = [1, 0, 0, 0]  # quaternion (identity)
        z[idx+11:idx+13] = zeros(3)      # angular velocity
    end
    return z
end

chassis_xy(z::Vector) = z[1:2]  # chassis (body 1) XY position in maximal state

# Input vector: zero base wrench, wheel torques = KV * target velocities
function build_input(ctrl::Vector)
    u = zeros(6 + NUM_WHEEL_DOFS)
    u[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET+NUM_WHEEL_DOFS-1] = KV .* ctrl
    return u
end

# ======================== Spline utilities ========================

function make_interp_matrix(T::Int, K::Int)
    W = zeros(T, K)
    for t in 1:T
        k_float = (t - 1) * (K - 1) / max(T - 1, 1)
        k_low   = floor(Int, k_float) + 1
        k_high  = min(k_low + 1, K)
        alpha   = k_float - (k_low - 1)
        W[t, k_low]  += 1.0 - alpha
        W[t, k_high] += alpha
    end
    return W, vec(sum(W, dims=1))
end

# ======================== Adam optimizer ========================

mutable struct SplineAdam
    lr::Float64; beta1::Float64; beta2::Float64; eps::Float64; clip_grad::Float64
    m::Matrix{Float64}; v::Matrix{Float64}; t::Int
end

SplineAdam(K::Int, n::Int; lr=40.0, beta1=0.5, beta2=0.999, eps=1e-8, clip_grad=200.0) =
    SplineAdam(lr, beta1, beta2, eps, clip_grad, zeros(K, n), zeros(K, n), 0)

function adam_step!(opt::SplineAdam, params::Matrix, grad::Matrix)
    opt.t += 1
    g = clamp.(grad, -opt.clip_grad, opt.clip_grad)
    opt.m = opt.beta1 .* opt.m .+ (1 - opt.beta1) .* g
    opt.v = opt.beta2 .* opt.v .+ (1 - opt.beta2) .* g.^2
    m̂ = opt.m ./ (1 - opt.beta1^opt.t)
    v̂ = opt.v ./ (1 - opt.beta2^opt.t)
    return params .- opt.lr .* m̂ ./ (sqrt.(v̂) .+ opt.eps)
end

# ======================== Differentiable rollout ========================

"""
Run forward rollout, storing states and per-step Jacobians (∂z_next/∂z, ∂z_next/∂u).

Jacobians use Dojo's attjac form (12*Nb dimensional) via get_maximal_gradients!.
The full maximal state z is 13*Nb dimensional.
"""
function rollout_with_gradients(mech::Mechanism, z0::Vector, ctrl_traj::Matrix)
    z = copy(z0)
    Nb = length(mech.bodies)

    zs  = Vector{Vector{Float64}}(undef, T_STEPS + 1)
    Jzs = Vector{Matrix{Float64}}(undef, T_STEPS)   # (12Nb × 12Nb) attjac Jacobians
    Jus = Vector{Matrix{Float64}}(undef, T_STEPS)   # (12Nb × nu)

    zs[1] = z
    set_maximal_state!(mech, z)

    for t in 1:T_STEPS
        u = build_input(ctrl_traj[t, :])
        Jz, Ju = get_maximal_gradients!(mech, z, u)
        z = get_maximal_state(mech)
        zs[t+1] = z
        Jzs[t] = Jz
        Jus[t] = Ju
    end

    return zs, Jzs, Jus
end

# ======================== Loss and BPTT gradient ========================

"""
Compute trajectory loss and gradient w.r.t. spline parameters via BPTT.

Loss = (TRAJECTORY_WEIGHT / T) * Σ_t ||xy_t - xy_target_t||²
     + REGULARIZATION_WEIGHT  * Σ_t ||ctrl_t||²

The adjoint lives in attjac space (12*Nb). Chassis XY position occupies
the same indices (1:2) in both full (13*Nb) and attjac (12*Nb) state vectors.
"""
function loss_and_grad(zs, Jzs, Jus, target_xy, ctrl_traj, W, col_sums)
    Nb    = length(zs[1]) ÷ 13
    nu    = 6 + NUM_WHEEL_DOFS
    L     = 0.0
    adj   = zeros(12 * Nb)
    grad_u_wheels = zeros(T_STEPS, NUM_WHEEL_DOFS)   # ∂L/∂u_wheels[t]

    # Backward pass: t = T_STEPS down to 1
    for t in T_STEPS:-1:1
        # Accumulate loss gradient at state z[t+1]
        delta = chassis_xy(zs[t+1]) - target_xy[t+1]
        L += TRAJECTORY_WEIGHT / T_STEPS * dot(delta, delta)
        adj[1:2] .+= (2 * TRAJECTORY_WEIGHT / T_STEPS) .* delta

        # ∂L/∂u_wheels[t] = Ju[t]' * adj  (wheel-torque slice)
        grad_u_wheels[t, :] = (Jus[t]' * adj)[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET+NUM_WHEEL_DOFS-1]

        # Propagate adjoint backward through dynamics
        adj = Jzs[t]' * adj
    end

    # Convert BPTT gradient from u-space to ctrl-space:
    # u_wheels = KV * ctrl_traj  →  ∂L/∂ctrl_t = KV * ∂L/∂u_t
    grad_ctrl = grad_u_wheels .* KV   # (T, 3)

    # Add regularization gradient directly in ctrl-space
    for t in 1:T_STEPS
        L += REGULARIZATION_WEIGHT * dot(ctrl_traj[t, :], ctrl_traj[t, :])
        grad_ctrl[t, :] .+= 2 * REGULARIZATION_WEIGHT .* ctrl_traj[t, :]
    end

    # Contract gradient from ctrl_traj space to spline params θ:
    # ctrl_traj = W @ θ  →  ∂L/∂θ = W' @ (∂L/∂ctrl_traj) ./ col_sums
    grad_theta = (W' * grad_ctrl) ./ col_sums

    return L, grad_theta
end

# ======================== Main ========================

function main()
    save_path = nothing
    target_only = false
    for i in 1:length(ARGS)
        if ARGS[i] == "--save" && i < length(ARGS)
            save_path = ARGS[i+1]
        elseif ARGS[i] == "--target-only"
            target_only = true
        end
    end

    mech = build_helhest()
    z0   = initial_maximal_state()
    nu   = input_dimension(mech)

    println("Helhest mechanism:")
    println("  Bodies:    ", length(mech.bodies))
    println("  Joints:    ", length(mech.joints))
    println("  Contacts:  ", length(mech.contacts))
    println("  Input dim: ", nu, "  (6 base + 3 wheel torques)")
    println("  T=$T_STEPS steps, dt=$DT s, K=$K control points")

    # --- Target episode ---
    println("\nSimulating target episode...")
    set_maximal_state!(mech, z0)
    target_ctrl_traj = repeat(TARGET_CTRL', T_STEPS)  # (T, 3)
    target_zs, _, _  = rollout_with_gradients(mech, z0, target_ctrl_traj)
    target_xy = [chassis_xy(target_zs[t]) for t in 1:T_STEPS+1]
    println("  Target final xy: ($(round(target_xy[end][1], digits=3)), $(round(target_xy[end][2], digits=3)))")

    if target_only
        if save_path === nothing
            save_path = "helhest_dojo.json"
        end
        traj_json = "[\n" * join(["    [$(xy[1]), $(xy[2])]" for xy in target_xy], ",\n") * "\n  ]"
        mkpath(dirname(abspath(save_path)))
        open(save_path, "w") do io
            println(io, "{")
            println(io, "  \"simulator\": \"Dojo\",")
            println(io, "  \"problem\": \"helhest\",")
            println(io, "  \"dt\": $DT,")
            println(io, "  \"T\": $T_STEPS,")
            println(io, "  \"target_trajectory\": $traj_json")
            println(io, "}")
        end
        println("Saved to $save_path")
        return
    end

    # --- Spline setup ---
    W, col_sums = make_interp_matrix(T_STEPS, K)
    theta = repeat(INIT_CTRL', K)   # (K, 3) initial control points
    opt   = SplineAdam(K, NUM_WHEEL_DOFS, lr=0.3)


    # --- Optimization loop ---
    println("\nOptimizing: lr=0.15 (Adam, beta1=0.5, clip=100)")
    iters_log = Int[]
    loss_log  = Float64[]
    time_log  = Float64[]

    for i in 1:50
        t0 = time()
        ctrl_traj = W * theta    # (T, 3)

        set_maximal_state!(mech, z0)
        zs, Jzs, Jus = rollout_with_gradients(mech, z0, ctrl_traj)

        L, grad_theta = loss_and_grad(zs, Jzs, Jus, target_xy, ctrl_traj, W, col_sums)
        theta = adam_step!(opt, theta, grad_theta)

        t_iter = (time() - t0) * 1000
        p0, pm, pN = theta[1, :], theta[K÷2, :], theta[end, :]
        @printf("Iter %3d: loss=%.4f | cp[1]=(%.2f,%.2f) cp[%d]=(%.2f,%.2f) cp[end]=(%.2f,%.2f) | t=%.0fms\n",
            i, L, p0[1], p0[2], K÷2, pm[1], pm[2], pN[1], pN[2], t_iter)
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
            println(io, "  \"problem\": \"helhest\",")
            println(io, "  \"dt\": $DT,")
            println(io, "  \"T\": $T_STEPS,")
            println(io, "  \"K\": $K,")
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
    ctrl_traj = W * theta
    ctrl_fn = (m, k) -> set_input!(m, build_input(ctrl_traj[min(k, T_STEPS), :]))
    set_maximal_state!(mech, z0)
    storage = simulate!(mech, DURATION, ctrl_fn; record=true, verbose=false)
    vis = visualize(mech, storage)
    println("Open http://127.0.0.1:8700 in your browser.")
    println("Press Ctrl+C to stop the server.")
    while true; sleep(1); end
end

main()

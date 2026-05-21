"""
Hyperparameter sweep for Dojo helhest -- minimize trajectory error vs ground truth.

Usage:
    ~/.juliaup/bin/julia +1.10 --startup-file=no examples/comparison_gradient/helhest/sweep_dojo.jl \
        --ground-truth results/helhest_chrono.json \
        --save results/sweep_dojo.json
"""

using Dojo
using LinearAlgebra
using JSON
using Printf

const DURATION = 3.0
const TARGET_CTRL = [1.0, 6.0, 0.0]

const WHEEL_RADIUS   = 0.36
const WHEEL_WIDTH    = 0.11
const WHEEL_MASS     = 5.5
const CHASSIS_INIT_Z = WHEEL_RADIUS

const WHEEL_DOF_OFFSET = 7
const NUM_WHEEL_DOFS   = 3

# ======================== Build mechanism ========================

function build_helhest(; timestep=0.01, gravity=-9.81,
        kv=100.0, friction_front=0.7, friction_rear=0.35)
    origin = Origin{Float64}()

    chassis_mass = 85.0 + 2.0 + 7.0 + 7.0 + 7.0 + 3.0 + 3.0
    chassis = Box(0.26, 0.6, 0.18, chassis_mass;
        color=RGBA(0.5, 0.5, 0.5), name=:chassis)

    wheel_orientation = Quaternion(cos(π/4), sin(π/4), 0.0, 0.0)
    make_wheel = (nm) -> Capsule(WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS;
        orientation_offset=wheel_orientation,
        color=RGBA(0.15, 0.15, 0.15), name=nm)

    left_wheel  = make_wheel(:left_wheel)
    right_wheel = make_wheel(:right_wheel)
    rear_wheel  = make_wheel(:rear_wheel)
    bodies = [chassis, left_wheel, right_wheel, rear_wheel]

    j_base = JointConstraint(Floating(origin, chassis;
        spring=0.0, damper=0.0), name=:base_joint)

    left_pos  = [0.0,    0.36, 0.0]
    right_pos = [0.0,   -0.36, 0.0]
    rear_pos  = [-0.697, 0.0,  0.0]

    j_left  = JointConstraint(Revolute(chassis, left_wheel,  Y_AXIS;
        parent_vertex=left_pos,  child_vertex=zeros(3),
        spring=0.0, damper=kv), name=:left_wheel_j)
    j_right = JointConstraint(Revolute(chassis, right_wheel, Y_AXIS;
        parent_vertex=right_pos, child_vertex=zeros(3),
        spring=0.0, damper=kv), name=:right_wheel_j)
    j_rear  = JointConstraint(Revolute(chassis, rear_wheel,  Y_AXIS;
        parent_vertex=rear_pos,  child_vertex=zeros(3),
        spring=0.0, damper=kv), name=:rear_wheel_j)
    joints = [j_base, j_left, j_right, j_rear]

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

function initial_maximal_state()
    positions = [
        [0.0,    0.0,  CHASSIS_INIT_Z],
        [0.0,    0.36, CHASSIS_INIT_Z],
        [0.0,   -0.36, CHASSIS_INIT_Z],
        [-0.697, 0.0,  CHASSIS_INIT_Z],
    ]
    z = zeros(13 * length(positions))
    for (i, pos) in enumerate(positions)
        idx = 13 * (i - 1)
        z[idx+1:idx+3]   = pos
        z[idx+4:idx+6]   = zeros(3)
        z[idx+7:idx+10]  = [1, 0, 0, 0]
        z[idx+11:idx+13] = zeros(3)
    end
    return z
end

chassis_xy(z::Vector) = z[1:2]

function build_input(ctrl::Vector, kv::Float64)
    u = zeros(6 + NUM_WHEEL_DOFS)
    u[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET+NUM_WHEEL_DOFS-1] = kv .* ctrl
    return u
end

# ======================== Simulate ========================

function simulate_config(; dt, kv, friction_front, friction_rear)
    T = Int(DURATION / dt)
    mech = build_helhest(; timestep=dt, kv=kv,
        friction_front=friction_front, friction_rear=friction_rear)
    z0 = initial_maximal_state()
    z = copy(z0)

    traj = Vector{Vector{Float64}}()
    push!(traj, chassis_xy(z))

    for t in 1:T
        u = build_input(TARGET_CTRL, kv)
        z = step!(mech, z, u)
        push!(traj, chassis_xy(z))
    end

    return traj
end

# ======================== Trajectory error ========================

function trajectory_error(traj_sim, traj_gt)
    n_sim = length(traj_sim)
    n_gt  = length(traj_gt)
    n = min(n_sim, n_gt, 500)

    t_sim = range(0, 1, length=n_sim)
    t_gt  = range(0, 1, length=n_gt)
    t_common = range(0, 1, length=n)

    # Linear interpolation helper
    function interp1d(ts, vals, t_query)
        result = zeros(length(t_query))
        for (qi, tq) in enumerate(t_query)
            # Find bracketing interval
            idx = searchsortedlast(ts, tq)
            idx = clamp(idx, 1, length(ts) - 1)
            alpha = (tq - ts[idx]) / (ts[idx+1] - ts[idx])
            alpha = clamp(alpha, 0.0, 1.0)
            result[qi] = (1 - alpha) * vals[idx] + alpha * vals[idx+1]
        end
        return result
    end

    sim_x = interp1d(collect(t_sim), [p[1] for p in traj_sim], collect(t_common))
    sim_y = interp1d(collect(t_sim), [p[2] for p in traj_sim], collect(t_common))
    gt_x  = interp1d(collect(t_gt),  [p[1] for p in traj_gt],  collect(t_common))
    gt_y  = interp1d(collect(t_gt),  [p[2] for p in traj_gt],  collect(t_common))

    return mean(sqrt.((sim_x .- gt_x).^2 .+ (sim_y .- gt_y).^2))
end

using Statistics: mean

# ======================== Sweep ========================

function run_sweep(configs, traj_gt, label="")
    results = []
    n = length(configs)
    for (i, cfg) in enumerate(configs)
        try
            traj = simulate_config(; cfg...)
            err = trajectory_error(traj, traj_gt)
            push!(results, (params=cfg, error=err, final_xy=traj[end]))
            if i % 20 == 0 || i == 1
                @printf("  %s [%d/%d] err=%.4f dt=%.4f kv=%.1f ff=%.2f rf=%.2f\n",
                    label, i, n, err, cfg.dt, cfg.kv, cfg.friction_front, cfg.friction_rear)
            end
        catch e
            push!(results, (params=cfg, error=Inf, final_xy=[0.0, 0.0]))
            if i % 20 == 0 || i == 1
                @printf("  %s [%d/%d] FAILED: %s\n", label, i, n, string(e))
            end
        end
    end
    sort!(results, by=r -> r.error)
    return results
end

function build_coarse_configs()
    configs = []
    for dt in [0.005, 0.01, 0.02, 0.03]
        for kv in [50.0, 100.0, 150.0, 200.0]
            for ff in [0.5, 0.7, 1.0, 1.5]
                for rf in [0.2, 0.35, 0.5]
                    push!(configs, (dt=dt, kv=kv, friction_front=ff, friction_rear=rf))
                end
            end
        end
    end
    return configs
end

function build_fine_configs(best)
    configs = []
    dt_b = best.dt
    kv_b = best.kv
    ff_b = best.friction_front
    rf_b = best.friction_rear

    for dt in [dt_b * 0.5, dt_b, dt_b * 2.0]
        for kv in [kv_b * 0.8, kv_b, kv_b * 1.2]
            for ff_mult in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
                for rf_mult in [0.85, 0.95, 1.0, 1.05, 1.15]
                    push!(configs, (dt=dt, kv=kv,
                        friction_front=ff_b * ff_mult,
                        friction_rear=rf_b * rf_mult))
                end
            end
        end
    end
    return configs
end

# ======================== Main ========================

function main()
    gt_path = nothing
    save_path = nothing
    top_n = 10
    for i in 1:length(ARGS)
        if ARGS[i] == "--ground-truth" && i < length(ARGS)
            gt_path = ARGS[i+1]
        elseif ARGS[i] == "--save" && i < length(ARGS)
            save_path = ARGS[i+1]
        elseif ARGS[i] == "--top" && i < length(ARGS)
            top_n = parse(Int, ARGS[i+1])
        end
    end

    if gt_path === nothing
        error("Usage: --ground-truth <path.json> [--save <path.json>] [--top N]")
    end

    # Load ground truth
    gt = JSON.parsefile(gt_path)
    traj_gt = [Float64[p[1], p[2]] for p in gt["target_trajectory"]]
    println("Ground truth: $(get(gt, "simulator", "?")), dt=$(gt["dt"]), T=$(gt["T"]), " *
            "final=($(round(traj_gt[end][1], digits=3)), $(round(traj_gt[end][2], digits=3)))")

    # Stage 1: coarse sweep
    println("\n=== Stage 1: Coarse sweep ===")
    coarse_configs = build_coarse_configs()
    println("Running $(length(coarse_configs)) configurations...")
    t0 = time()
    coarse_results = run_sweep(coarse_configs, traj_gt, "coarse")
    t_coarse = time() - t0
    @printf("Coarse sweep done in %.1fs\n", t_coarse)

    println("\nTop $top_n coarse results:")
    for (i, r) in enumerate(coarse_results[1:min(top_n, end)])
        p = r.params
        @printf("  %d. err=%.4f | dt=%.4f kv=%.1f ff=%.2f rf=%.2f | final=(%.3f, %.3f)\n",
            i, r.error, p.dt, p.kv, p.friction_front, p.friction_rear,
            r.final_xy[1], r.final_xy[2])
    end

    # Stage 2: fine sweep
    println("\n=== Stage 2: Fine sweep ===")
    best_coarse = coarse_results[1].params
    fine_configs = build_fine_configs(best_coarse)
    println("Running $(length(fine_configs)) configurations...")
    t0 = time()
    fine_results = run_sweep(fine_configs, traj_gt, "fine")
    t_fine = time() - t0
    @printf("Fine sweep done in %.1fs\n", t_fine)

    println("\nTop $top_n fine results:")
    for (i, r) in enumerate(fine_results[1:min(top_n, end)])
        p = r.params
        @printf("  %d. err=%.4f | dt=%.4f kv=%.1f ff=%.2f rf=%.2f | final=(%.3f, %.3f)\n",
            i, r.error, p.dt, p.kv, p.friction_front, p.friction_rear,
            r.final_xy[1], r.final_xy[2])
    end

    # Save results
    if save_path !== nothing
        best = fine_results[1]
        # Filter out Inf errors for JSON compatibility
        valid_coarse = [r for r in coarse_results if isfinite(r.error)]
        valid_fine = [r for r in fine_results if isfinite(r.error)]

        best_error = isfinite(best.error) ? best.error : -1.0
        output = Dict(
            "simulator" => "Dojo",
            "ground_truth" => gt_path,
            "best_params" => Dict(pairs(best.params)),
            "best_error" => best_error,
            "best_final_xy" => best.final_xy,
            "coarse_sweep_size" => length(coarse_configs),
            "fine_sweep_size" => length(fine_configs),
            "num_coarse_succeeded" => length(valid_coarse),
            "num_fine_succeeded" => length(valid_fine),
            "top_10_coarse" => [Dict(
                "error" => r.error,
                "params" => Dict(pairs(r.params)),
                "final_xy" => r.final_xy
            ) for r in valid_coarse[1:min(10, end)]],
            "top_10_fine" => [Dict(
                "error" => r.error,
                "params" => Dict(pairs(r.params)),
                "final_xy" => r.final_xy
            ) for r in valid_fine[1:min(10, end)]],
        )
        mkpath(dirname(abspath(save_path)))
        open(save_path, "w") do io
            JSON.print(io, output, 2)
        end
        println("\nSaved to $save_path")
    end
end

main()

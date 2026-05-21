"""
Hyperparameter sweep for Dojo helhest — minimize trajectory error vs real robot.

Usage:
    ~/.juliaup/bin/julia +1.10 --startup-file=no examples/comparison_accuracy/helhest/sweep_dojo.jl \
        --ground-truth examples/comparison_accuracy/helhest/results/helhest_2026_04_10-14_46_18.json \
        --save examples/comparison_accuracy/helhest/results/sweep_dojo_14_46_18.json
"""

using Dojo
using LinearAlgebra
using JSON
using Printf
using Statistics: mean

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

function interp_wheel_vel(wheel_ts, sim_time)
    """Linearly interpolate wheel velocity timeseries at sim_time."""
    # wheel_ts: vector of (t, left, right, rear) named tuples
    n = length(wheel_ts)
    if sim_time <= wheel_ts[1].t
        return [wheel_ts[1].left, wheel_ts[1].right, wheel_ts[1].rear]
    end
    if sim_time >= wheel_ts[end].t
        return [wheel_ts[end].left, wheel_ts[end].right, wheel_ts[end].rear]
    end
    # Binary search for bracket
    lo, hi = 1, n
    while hi - lo > 1
        mid = div(lo + hi, 2)
        if wheel_ts[mid].t <= sim_time
            lo = mid
        else
            hi = mid
        end
    end
    dt_span = wheel_ts[hi].t - wheel_ts[lo].t
    if dt_span < 1e-10
        return [wheel_ts[lo].left, wheel_ts[lo].right, wheel_ts[lo].rear]
    end
    alpha = (sim_time - wheel_ts[lo].t) / dt_span
    return [
        (1 - alpha) * wheel_ts[lo].left  + alpha * wheel_ts[hi].left,
        (1 - alpha) * wheel_ts[lo].right + alpha * wheel_ts[hi].right,
        (1 - alpha) * wheel_ts[lo].rear  + alpha * wheel_ts[hi].rear,
    ]
end

function build_input(ctrl::Vector, kv::Float64)
    u = zeros(6 + NUM_WHEEL_DOFS)
    u[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET+NUM_WHEEL_DOFS-1] = kv .* ctrl
    return u
end

# ======================== Simulate ========================

function simulate_config(; dt, kv, friction_front, friction_rear, duration, wheel_ts,
                          target_ctrl=nothing)
    T = Int(round(duration / dt))
    mech = build_helhest(; timestep=dt, kv=kv,
        friction_front=friction_front, friction_rear=friction_rear)
    z0 = initial_maximal_state()
    z = copy(z0)

    traj = Vector{Vector{Float64}}()
    push!(traj, chassis_xy(z))

    for t in 1:T
        sim_time = t * dt
        ctrl = interp_wheel_vel(wheel_ts, sim_time)
        u = build_input(ctrl, kv)
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

    function interp1d(ts, vals, t_query)
        result = zeros(length(t_query))
        for (qi, tq) in enumerate(t_query)
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

# ======================== Sweep ========================

function run_sweep(configs, gt_data_list, label="")
    results = []
    n = length(configs)
    for (i, cfg) in enumerate(configs)
        errors = Float64[]
        per_traj = Dict{String, Any}()

        for gt_entry in gt_data_list
            bag_name = gt_entry["bag_name"]
            try
                traj = simulate_config(; cfg..., duration=gt_entry["duration"],
                    wheel_ts=gt_entry["wheel_ts"])
                err = trajectory_error(traj, gt_entry["traj_gt"])
                push!(errors, err)
                per_traj[bag_name] = Dict("error" => err, "trajectory" => traj)
            catch e
                push!(errors, Inf)
                per_traj[bag_name] = Dict("error" => Inf, "exception" => string(e)[1:min(200, end)])
            end
        end

        combined_err = mean(errors)
        push!(results, (params=cfg, error=combined_err, per_trajectory=per_traj))

        if i % 20 == 0 || i == 1
            err_strs = join([@sprintf("%.4f", e) for e in errors], " + ")
            @printf("  %s [%d/%d] err=%.4f (%s) dt=%.4f kv=%.1f mu=%.2f\n",
                label, i, n, combined_err, err_strs, cfg.dt, cfg.kv, cfg.friction_front)
        end
    end
    sort!(results, by=r -> r.error)
    return results
end

function build_coarse_configs()
    configs = []
    for dt in [0.01, 0.02, 0.05]
        for kv in [500.0, 1000.0, 2000.0, 4000.0]
            for mu in [0.1, 0.2, 0.35, 0.5, 0.7]
                push!(configs, (dt=dt, kv=kv, friction_front=mu, friction_rear=mu))
            end
        end
    end
    return configs
end

function build_fine_configs(best)
    configs = []
    dt_b = best.dt
    kv_b = best.kv
    mu_b = best.friction_front

    for dt in [dt_b * 0.5, dt_b, dt_b * 2.0]
        for kv in [kv_b * 0.8, kv_b, kv_b * 1.2]
            for mu_mult in [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
                mu = mu_b * mu_mult
                push!(configs, (dt=dt, kv=kv, friction_front=mu, friction_rear=mu))
            end
        end
    end
    return configs
end

# ======================== Load ground truth ========================

function load_ground_truth(path)
    gt = JSON.parsefile(path)
    duration = Float64(get(gt["trajectory"], "constant_speed_duration_s", gt["trajectory"]["duration_s"]))
    traj_gt = [Float64[p["x"], p["y"]] for p in gt["trajectory"]["points"] if p["t"] <= duration]
    target_ctrl = Float64.(gt["target_ctrl_rad_s"])

    # Parse wheel velocity timeseries
    wheel_ts_raw = gt["wheel_velocities"]["timeseries"]
    wheel_ts = [(t=Float64(p["t"]), left=Float64(p["left"]),
                 right=Float64(p["right"]), rear=Float64(p["rear"]))
                for p in wheel_ts_raw]

    return traj_gt, target_ctrl, duration, gt, wheel_ts
end

# ======================== Main ========================

function main()
    gt_paths = String[]
    save_path = nothing
    top_n = 10
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--ground-truth"
            i += 1
            while i <= length(ARGS) && !startswith(ARGS[i], "--")
                push!(gt_paths, ARGS[i])
                i += 1
            end
        elseif ARGS[i] == "--save" && i < length(ARGS)
            save_path = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--top" && i < length(ARGS)
            top_n = parse(Int, ARGS[i+1])
            i += 2
        else
            i += 1
        end
    end

    if isempty(gt_paths)
        error("Usage: --ground-truth <path1.json> [path2.json ...] [--save <path.json>] [--top N]")
    end

    gt_data_list = []
    for gt_path in gt_paths
        traj_gt, target_ctrl, duration, gt_data, wheel_ts = load_ground_truth(gt_path)
        bag_name = get(gt_data, "bag_name", "?")
        push!(gt_data_list, Dict(
            "path" => gt_path,
            "traj_gt" => traj_gt,
            "target_ctrl" => target_ctrl,
            "duration" => duration,
            "bag_name" => bag_name,
            "wheel_ts" => wheel_ts,
        ))
        println("Ground truth: $bag_name")
        @printf("  duration=%.1fs, %d points, final=(%.3f, %.3f)\n",
            duration, length(traj_gt), traj_gt[end][1], traj_gt[end][2])
    end

    # Stage 1: coarse sweep
    println("\n=== Stage 1: Coarse sweep ===")
    coarse_configs = build_coarse_configs()
    println("Running $(length(coarse_configs)) configs x $(length(gt_data_list)) trajectories...")
    t0 = time()
    coarse_results = run_sweep(coarse_configs, gt_data_list, "coarse")
    t_coarse = time() - t0
    @printf("Coarse sweep done in %.1fs\n", t_coarse)

    println("\nTop $top_n coarse results:")
    for (i, r) in enumerate(coarse_results[1:min(top_n, end)])
        p = r.params
        errs = join([@sprintf("%.4f", v["error"]) for (k, v) in r.per_trajectory], " + ")
        @printf("  %d. err=%.4f (%s) | dt=%.4f kv=%.1f mu=%.2f\n",
            i, r.error, errs, p.dt, p.kv, p.friction_front)
    end

    # Stage 2: fine sweep
    println("\n=== Stage 2: Fine sweep ===")
    best_coarse = coarse_results[1].params
    fine_configs = build_fine_configs(best_coarse)
    println("Running $(length(fine_configs)) configs x $(length(gt_data_list)) trajectories...")
    t0 = time()
    fine_results = run_sweep(fine_configs, gt_data_list, "fine")
    t_fine = time() - t0
    @printf("Fine sweep done in %.1fs\n", t_fine)

    println("\nTop $top_n fine results:")
    for (i, r) in enumerate(fine_results[1:min(top_n, end)])
        p = r.params
        errs = join([@sprintf("%.4f", v["error"]) for (k, v) in r.per_trajectory], " + ")
        @printf("  %d. err=%.4f (%s) | dt=%.4f kv=%.1f mu=%.2f\n",
            i, r.error, errs, p.dt, p.kv, p.friction_front)
    end

    # Save results
    if save_path !== nothing
        best = fine_results[1]
        valid_coarse = [r for r in coarse_results if isfinite(r.error)]
        valid_fine = [r for r in fine_results if isfinite(r.error)]

        best_error = isfinite(best.error) ? best.error : -1.0

        # Convert per_trajectory: strip trajectories for JSON size, keep errors
        best_per_traj = Dict{String, Any}()
        for (k, v) in best.per_trajectory
            best_per_traj[k] = Dict("error" => v["error"])
        end

        output = Dict(
            "simulator" => "Dojo",
            "ground_truth" => gt_paths,
            "ground_truth_source" => "real_robot",
            "best_params" => Dict(pairs(best.params)),
            "best_error" => best_error,
            "best_per_trajectory" => best_per_traj,
            "coarse_sweep_size" => length(coarse_configs),
            "fine_sweep_size" => length(fine_configs),
            "top_10_coarse" => [Dict(
                "error" => r.error,
                "params" => Dict(pairs(r.params)),
            ) for r in valid_coarse[1:min(10, end)]],
            "top_10_fine" => [Dict(
                "error" => r.error,
                "params" => Dict(pairs(r.params)),
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

"""
Debug script: Helhest robot in Dojo, no control, 2 seconds simulation.
Prints body positions every 0.1s to verify contact/ground behaviour.

Run with Julia 1.10 for visualization:
    ~/.juliaup/bin/julia +1.10 --startup-file=no examples/comparison/helhest/dojo_debug.jl
"""

using Dojo
using Printf

const WHEEL_RADIUS = 0.36
const WHEEL_WIDTH  = 0.11
const WHEEL_MASS   = 5.5
const DT           = 0.01   # small dt for stability

function build_helhest_debug(; timestep=DT, gravity=-9.81,
        friction_front=0.7, friction_rear=0.35)
    origin = Origin{Float64}()

    # Chassis (lumped mass)
    chassis_mass = 85.0 + 2.0 + 7.0 + 7.0 + 7.0 + 3.0 + 3.0  # 114 kg
    chassis = Box(0.26, 0.6, 0.18, chassis_mass;
        color=RGBA(0.5, 0.5, 0.5), name=:chassis)

    # Wheels as Capsule(r, h): capsule axis along body-Z, rotated 90° → axis along Y.
    # The two hemispherical caps sit at [0, ±h/2, 0] in body frame after rotation.
    wheel_orientation = Quaternion(cos(π/4), sin(π/4), 0.0, 0.0)  # RotX(π/2)
    make_wheel = (nm) -> Capsule(WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS;
        orientation_offset=wheel_orientation,
        color=RGBA(0.15, 0.15, 0.15), name=nm)

    left_wheel  = make_wheel(:left_wheel)
    right_wheel = make_wheel(:right_wheel)
    rear_wheel  = make_wheel(:rear_wheel)
    bodies = [chassis, left_wheel, right_wheel, rear_wheel]

    # Joints
    j_base = JointConstraint(Floating(origin, chassis;
        spring=0.0, damper=0.0), name=:base_joint)

    left_pos  = [0.0,    0.36, 0.0]
    right_pos = [0.0,   -0.36, 0.0]
    rear_pos  = [-0.697, 0.0,  0.0]

    j_left  = JointConstraint(Revolute(chassis, left_wheel,  Y_AXIS;
        parent_vertex=left_pos,  child_vertex=zeros(3),
        spring=0.0, damper=0.0), name=:left_wheel_j)
    j_right = JointConstraint(Revolute(chassis, right_wheel, Y_AXIS;
        parent_vertex=right_pos, child_vertex=zeros(3),
        spring=0.0, damper=0.0), name=:right_wheel_j)
    j_rear  = JointConstraint(Revolute(chassis, rear_wheel,  Y_AXIS;
        parent_vertex=rear_pos,  child_vertex=zeros(3),
        spring=0.0, damper=0.0), name=:rear_wheel_j)
    joints = [j_base, j_left, j_right, j_rear]

    # Capsule contacts: 2 sphere contacts per wheel at cap centers [0, ±W/2, 0],
    # contact_radius = WHEEL_RADIUS so signed_dist = z_wheel - R (zero when on ground).
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

    contacts = [
        wheel_contacts(left_wheel,  friction_front, [:lc1, :lc2]);
        wheel_contacts(right_wheel, friction_front, [:rc1, :rc2]);
        wheel_contacts(rear_wheel,  friction_rear,  [:rrc1, :rrc2]);
    ]

    return Mechanism(origin, bodies, joints, contacts; gravity, timestep)
end

function initial_state()
    # Wheel center at z = WHEEL_RADIUS → exactly touching ground (signed_dist = 0)
    positions = [
        [0.0,    0.0,  WHEEL_RADIUS],   # chassis
        [0.0,    0.36, WHEEL_RADIUS],   # left_wheel
        [0.0,   -0.36, WHEEL_RADIUS],   # right_wheel
        [-0.697, 0.0,  WHEEL_RADIUS],   # rear_wheel
    ]
    z = zeros(13 * length(positions))
    for (i, pos) in enumerate(positions)
        idx = 13 * (i - 1)
        z[idx+1:idx+3]  = pos
        z[idx+7:idx+10] = [1, 0, 0, 0]  # identity quaternion
    end
    return z
end

function main()
    mech = build_helhest_debug()
    println("Mechanism: $(length(mech.bodies)) bodies, $(length(mech.contacts)) contacts")

    z0 = initial_state()
    set_maximal_state!(mech, z0)

    println("\nSimulating 2s (no control, dt=$(DT)s)...\n")
    println(@sprintf("%-6s  %-24s  %-24s  %-24s  %-24s", "t(s)",
        "chassis xyz", "left_wheel xyz", "right_wheel xyz", "rear_wheel xyz"))
    println("-"^105)

    steps_per_print = max(1, round(Int, 0.1 / DT))
    storage = simulate!(mech, 2.0,
        (m, k) -> nothing;   # no control
        record=true, verbose=false)

    for k in 1:steps_per_print:length(storage.x[1])
        t = (k - 1) * DT
        xs = [storage.x[b][k] for b in 1:4]
        @printf("%-6.2f  (% .3f,% .3f,% .3f)  (% .3f,% .3f,% .3f)  (% .3f,% .3f,% .3f)  (% .3f,% .3f,% .3f)\n",
            t,
            xs[1]..., xs[2]..., xs[3]..., xs[4]...)
    end

    println("\nFinal chassis z = $(round(storage.x[1][end][3], digits=4)) m  (expected ≈ $(WHEEL_RADIUS))")

    println("\nStarting visualizer at http://127.0.0.1:8700")
    vis = visualize(mech, storage)
    println("Press Ctrl+C to stop.")
    while true; sleep(1); end
end

main()

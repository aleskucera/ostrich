#!/bin/bash
# Run all differentiable simulator gradient tests.
# Each test runs in a separate Python process to avoid CUDA state contamination
# between models with different topologies.

set -e
cd "$(dirname "$0")"

TESTS=(
    test_zero_gradient.py
    test_velocity_gradient.py
    test_multi_step_gradient.py
    test_pose_gradient.py
    test_position_loss.py
    test_optimization.py
    test_symmetry.py
    test_contact_boundary.py
    test_cartpole_gradient.py
    test_wheeled_robot.py
)

PASSED=0
FAILED=0
FAILED_NAMES=()

for test in "${TESTS[@]}"; do
    echo "========================================"
    echo "Running: $test"
    echo "========================================"
    if python "$test" 2>&1 | grep -v "^Module\|^Warp [0-9]"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_NAMES+=("$test")
    fi
    echo ""
done

echo "========================================"
echo "Results: $PASSED passed, $FAILED failed"
if [ $FAILED -gt 0 ]; then
    echo "Failed tests:"
    for name in "${FAILED_NAMES[@]}"; do
        echo "  - $name"
    done
    exit 1
fi
echo "All tests passed!"

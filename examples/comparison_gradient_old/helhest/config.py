"""Shared constants for helhest trajectory optimization benchmark."""

DT = 5e-2
DURATION = 3.0
K = 30  # number of spline control points

# Target and initial wheel control values (left, right, rear)
TARGET_CTRL = (1.0, 6.0, 0.0)
INIT_CTRL = (2.0, 5.0, 0.0)

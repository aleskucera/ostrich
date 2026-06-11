from enum import IntEnum


class JointMode(IntEnum):
    """
    Specifies the control mode for a joint's actuation.

    Joint modes determine how a joint is actuated or controlled during simulation.
    """

    NONE = 0
    """No implicit control is applied to the joint, but the joint can be controlled by applying forces."""

    TARGET_POSITION = 1
    """The joint is controlled to reach a target position."""

    TARGET_VELOCITY = 2
    """The joint is controlled to reach a target velocity."""

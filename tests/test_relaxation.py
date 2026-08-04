"""Tests for the structural relaxation rungs (RELAXATION_PLAN.md)."""
import pytest

from ostrich import RelaxationConfig
from ostrich import OstrichEngineConfig
from ostrich.constraints.friction_constraint import FRICTION_BILATERAL
from ostrich.constraints.friction_constraint import FRICTION_BILATERAL_PATCH
from ostrich.constraints.friction_constraint import FRICTION_CONE
from ostrich.constraints.contact_constraint import CONTACT_COMPLEMENTARITY
from ostrich.constraints.contact_constraint import CONTACT_EQUALITY


def test_friction_mode_ints_match_config():
    """The kernel constants and the config's string order must agree.

    `RelaxationConfig.friction_mode` is `_FRICTION_MODES.index(...)`, and the
    kernels branch on the raw int. If the two drift apart, every rung silently
    runs a different formulation than the one named — which would not fail
    loudly anywhere else.
    """
    modes = RelaxationConfig._FRICTION_MODES
    assert modes.index("cone") == FRICTION_CONE
    assert modes.index("bilateral") == FRICTION_BILATERAL
    assert modes.index("bilateral_patch") == FRICTION_BILATERAL_PATCH


def test_contact_mode_ints_match_config():
    modes = RelaxationConfig._CONTACT_MODES
    assert modes.index("complementarity") == CONTACT_COMPLEMENTARITY
    assert modes.index("equality") == CONTACT_EQUALITY


def test_default_relaxation_is_the_full_formulation():
    cfg = RelaxationConfig()
    assert cfg.is_default
    assert cfg.friction_mode == FRICTION_CONE
    assert cfg.contact_mode == CONTACT_COMPLEMENTARITY
    assert cfg.gyro_scale == 1.0


def test_gyro_scale_toggles():
    assert RelaxationConfig(gyro=False).gyro_scale == 0.0


def test_unknown_friction_mode_rejected():
    with pytest.raises(ValueError, match="relaxation.friction"):
        RelaxationConfig(friction="no_slip")

    with pytest.raises(ValueError, match="relaxation.contact"):
        RelaxationConfig(contact="rigid")


@pytest.mark.parametrize(
    "kwargs",
    [{"friction": "bilateral"}, {"friction": "bilateral_patch"},
     {"contact": "equality"}, {"gyro": False}],
)
def test_differentiable_relaxed_rung_is_refused(kwargs):
    """The adjoint freezes stick/slip against mu*lambda_n, which is meaningless
    when lambda_f is unbounded. Wrong gradients here would be plausible-looking
    and silent, so the engine must refuse to build rather than emit them."""
    cfg = OstrichEngineConfig(
        differentiable=True, relaxation=RelaxationConfig(**kwargs)
    )
    with pytest.raises(NotImplementedError, match="adjoint"):
        cfg.create_engine(model=None)


def test_relaxation_coerces_from_dict():
    """Hydra leaves nested overrides as dict-likes; the sub-config must survive."""
    cfg = OstrichEngineConfig(
        relaxation={"friction": "bilateral_patch", "contact": "equality", "gyro": False}
    )
    assert isinstance(cfg.relaxation, RelaxationConfig)
    assert cfg.relaxation.friction_mode == FRICTION_BILATERAL_PATCH
    assert cfg.relaxation.contact_mode == CONTACT_EQUALITY
    assert cfg.relaxation.gyro_scale == 0.0

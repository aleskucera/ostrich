import torch

from axion.neural_solver.utils.pendulum_lambda_layout import (
    PENDULUM_FULL_LAMBDA_DIM,
    PENDULUM_LAMBDA_N_CTRL,
    PENDULUM_LAMBDA_N_J,
    contract_pendulum_canonical_to_engine_lambdas_torch,
    expand_pendulum_engine_lambdas_torch,
)


def test_expand_contract_roundtrip_22_to_24_to_22():
    engine_dim = PENDULUM_FULL_LAMBDA_DIM - PENDULUM_LAMBDA_N_CTRL
    engine = torch.arange(engine_dim, dtype=torch.float32).unsqueeze(0)
    expanded = expand_pendulum_engine_lambdas_torch(engine, PENDULUM_FULL_LAMBDA_DIM)

    assert expanded.shape[-1] == PENDULUM_FULL_LAMBDA_DIM
    assert torch.all(expanded[0, :PENDULUM_LAMBDA_N_J] == engine[0, :PENDULUM_LAMBDA_N_J])
    assert torch.all(expanded[0, PENDULUM_LAMBDA_N_J : PENDULUM_LAMBDA_N_J + PENDULUM_LAMBDA_N_CTRL] == 0)
    assert torch.all(
        expanded[0, PENDULUM_LAMBDA_N_J + PENDULUM_LAMBDA_N_CTRL :]
        == engine[0, PENDULUM_LAMBDA_N_J :]
    )

    contracted = torch.empty_like(engine)
    contract_pendulum_canonical_to_engine_lambdas_torch(contracted, expanded)
    assert torch.allclose(contracted, engine)


def test_contract_drops_control_slots():
    canonical = torch.zeros(1, PENDULUM_FULL_LAMBDA_DIM, dtype=torch.float32)
    canonical[0, :PENDULUM_LAMBDA_N_J] = 1.0
    canonical[0, PENDULUM_LAMBDA_N_J] = 100.0
    canonical[0, PENDULUM_LAMBDA_N_J + 1] = 200.0
    canonical[0, PENDULUM_LAMBDA_N_J + PENDULUM_LAMBDA_N_CTRL :] = 2.0

    engine_dim = PENDULUM_FULL_LAMBDA_DIM - PENDULUM_LAMBDA_N_CTRL
    dest = torch.empty(1, engine_dim, dtype=torch.float32)
    contract_pendulum_canonical_to_engine_lambdas_torch(dest, canonical)

    assert dest[0, :PENDULUM_LAMBDA_N_J].sum() == PENDULUM_LAMBDA_N_J
    assert dest[0, PENDULUM_LAMBDA_N_J:].sum() == (engine_dim - PENDULUM_LAMBDA_N_J) * 2.0
    assert 100.0 not in dest[0].tolist()
    assert 200.0 not in dest[0].tolist()

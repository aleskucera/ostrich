from typing import Optional

import warp as wp
from newton import Contacts
from newton import Control
from newton import Model
from newton import State

from .base_engine import AxionEngineBase
from .engine_config import AxionEngineConfig
from .logging_config import LoggingConfig


class AxionEngine(AxionEngineBase):
    def __init__(
        self,
        model: Model,
        sim_steps: int,
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
        logging_config: Optional[LoggingConfig] = LoggingConfig(),
        differentiable_simulation: bool = False,
    ):
        super().__init__(model, sim_steps, config, logging_config, differentiable_simulation)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        prof = self.profiler
        end_to_end = prof.enabled and prof.mode == "end_to_end"
        # End-to-end phase boundaries (matches END_TO_END_PHASES order):
        # 0: collide-start (simulator) | 1: load_data-start
        # 2: warm_start_copy-start    | 3: nr_solve-start
        # 4: backtracking-start (in _solve) | 5: output_copy-start
        # 6: output_copy-end
        # The simulator brackets the collide phase (boundaries 0 -> 1);
        # the engine brackets the rest.
        if end_to_end:
            prof.record_boundary(1)
        self.load_data(state_in, control, contacts, dt)
        if end_to_end:
            prof.record_boundary(2)

        # Initial guess: use current state (zero-order warm start)
        wp.copy(dest=self.data.body_pose, src=state_in.body_q)
        wp.copy(dest=self.data.body_vel, src=state_in.body_qd)

        self.data._constr_force.zero_()
        # _constr_force_prev_iter is zeroed inside load_data, before
        # warm_starter.apply, so warm-started values aren't clobbered.
        if self.config.warm_start.seed_iterate:
            # Seed NR's initial λ from the warm-starter's output. Only
            # contact-normal slots are populated; joint / friction slots
            # remain zero (cold-start). See warm_start_iterate_seeding_issue.md.
            wp.copy(dest=self.data._constr_force, src=self.data._constr_force_prev_iter)
        if end_to_end:
            prof.record_boundary(3)

        # _solve emits boundary 4 (NR-end / backtracking-start) internally.
        self._solve()
        if end_to_end:
            prof.record_boundary(5)

        wp.copy(dest=state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=state_out.body_qd, src=self.data.body_vel)
        if end_to_end:
            prof.record_boundary(6)

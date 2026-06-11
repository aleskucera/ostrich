import warp as wp
from ostrich.core.engine_data import EngineData
from newton import Model
from newton import State


class EpisodeTrajectory:
    def __init__(self, model: Model, num_steps: int, device: wp.Device) -> None:
        self.model = model
        self.num_steps = num_steps
        self.device = device

        with wp.ScopedDevice(self.device):
            self.body_q = wp.zeros(
                shape=(self.num_steps + 1, self.model.body_count),
                dtype=wp.transform,
                requires_grad=True,
            )
            self.body_qd = wp.zeros(
                shape=(self.num_steps + 1, self.model.body_count),
                dtype=wp.spatial_vector,
                requires_grad=True,
            )

    def reset(self):
        self.body_q.zero_()
        self.body_q.grad.zero_()
        self.body_qd.zero_()
        self.body_qd.grad.zero_()

    def save_state(self, state: State, idx: int):
        wp.copy(self.body_q[idx], state.body_q)
        wp.copy(self.body_qd[idx], state.body_qd)


class NewtonTargetTrajectory:
    def __init__(self, model: Model, num_steps: int, device: wp.Device) -> None:
        self.model = model
        self.num_steps = num_steps
        self.device = device

        with wp.ScopedDevice(self.device):
            self.body_q = wp.zeros(
                shape=(self.num_steps + 1, self.model.body_count),
                dtype=wp.transform,
            )
            self.body_qd = wp.zeros(
                shape=(self.num_steps + 1, self.model.body_count),
                dtype=wp.spatial_vector,
            )

    def reset(self):
        self.body_q.zero_()
        self.body_qd.zero_()

    def save_state(self, state: State, idx: int):
        wp.copy(self.body_q[idx], state.body_q)
        wp.copy(self.body_qd[idx], state.body_qd)

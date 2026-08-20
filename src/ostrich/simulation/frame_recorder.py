import numpy as np


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Slerp between two [B, 4] arrays of (x, y, z, w) quaternions."""
    q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0.0, -q1, q1)
    dot = np.clip(np.sum(q0 * q1, axis=-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    # Below this the arc is short enough that slerp and lerp agree to well
    # under float32 precision, and dividing by sin(theta) would blow up.
    degenerate = sin_theta < 1e-6
    safe = np.where(degenerate, 1.0, sin_theta)
    w0 = np.where(degenerate, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / safe)
    w1 = np.where(degenerate, alpha, np.sin(alpha * theta) / safe)

    q = w0 * q0 + w1 * q1
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)


class FrameRecorder:
    """Captures body poses straight onto a fixed-fps grid as the sim runs.

    Physics advances at whatever ``dt`` it needs; the recorder emits one
    frame per grid slot, interpolating between the two steps that bracket
    each frame time (linear on position, slerp on rotation). It works in
    both directions -- a ``dt`` finer than the frame interval decimates,
    a coarser one interpolates -- so the captured trajectory comes out
    smooth and real-time whatever the timestep, without a resampling pass
    afterwards.

    Poses are ``[num_bodies, 7]`` arrays laid out as
    ``(px, py, pz, qx, qy, qz, qw)``.
    """

    def __init__(self, fps: float, duration: float, num_bodies: int):
        self.fps = float(fps)
        self.num_bodies = int(num_bodies)
        self.num_frames = int(round(duration * fps))
        # Endpoint-inclusive grid: num_frames samples spanning [0, duration],
        # which played back at `fps` runs for `duration` seconds.
        self.times = np.linspace(0.0, duration, self.num_frames, dtype=np.float64)
        self.frames = np.zeros((self.num_frames, self.num_bodies, 7), dtype=np.float32)

        self._next = 0
        self._prev_pose: np.ndarray | None = None
        self._prev_time = 0.0

    def start(self, pose: np.ndarray):
        """Records the t=0 pose and arms the recorder."""
        self._prev_pose = np.asarray(pose, dtype=np.float32)[: self.num_bodies].copy()
        self._prev_time = 0.0
        self._emit(self._prev_pose, self._prev_pose, 0.0)

    def record(self, pose: np.ndarray, time: float):
        """Feeds the pose left by a physics step that reached sim time ``time``.

        Emits every grid frame that the step just stepped over, so a large
        ``dt`` yields several interpolated frames and a small one may yield
        none.
        """
        pose = np.asarray(pose, dtype=np.float32)[: self.num_bodies]
        span = time - self._prev_time
        while self._next < self.num_frames and self.times[self._next] <= time + 1e-9:
            alpha = 0.0 if span <= 0.0 else (self.times[self._next] - self._prev_time) / span
            self._emit(self._prev_pose, pose, min(1.0, max(0.0, alpha)))
        self._prev_pose = pose.copy()
        self._prev_time = time

    def finish(self) -> np.ndarray:
        """Holds the last pose over any unfilled frames and returns the grid.

        Frames go unfilled when the run stops early -- a diverged solve, or
        a duration that does not divide evenly -- matching how the previous
        ``np.interp`` resampling clamped past the end of the raw trajectory.
        """
        while self._next < self.num_frames:
            self._emit(self._prev_pose, self._prev_pose, 1.0)
        return self.frames

    def _emit(self, pose_0: np.ndarray, pose_1: np.ndarray, alpha: float):
        frame = self.frames[self._next]
        frame[:, :3] = pose_0[:, :3] + alpha * (pose_1[:, :3] - pose_0[:, :3])
        frame[:, 3:] = _slerp(
            pose_0[:, 3:].astype(np.float64), pose_1[:, 3:].astype(np.float64), alpha
        )
        self._next += 1

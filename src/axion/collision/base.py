"""Abstract base class for contact reducers and a no-op default."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axion.core.contacts import AxionContacts


class ContactReducer(ABC):
    """In-place contact reducer.

    A reducer compacts the active contact list inside an ``AxionContacts``
    instance so that, for each (body0, body1) pair, at most a configured
    number of contacts survive. Survivors are placed at the front of the
    per-world contact arrays and ``contact_count[w]`` is decreased to the
    survivor count, so downstream constraint kernels (which gate on
    ``c_idx >= contact_count[w]``) see the reduced set without any
    further changes.

    Implementations must be CUDA-graph capture safe: kernel launch
    dimensions and memory addresses cannot depend on host-side decisions
    that change between captures.
    """

    @abstractmethod
    def apply(self, contacts: "AxionContacts") -> None:
        """Reduce contacts in place. Called once per simulation step,
        after ``AxionContacts.load_contact_data`` and before the engine
        builds the linear system for that step."""
        raise NotImplementedError


class NoOpReducer(ContactReducer):
    """Reducer that does nothing — preserves pre-module engine behavior."""

    def apply(self, contacts: "AxionContacts") -> None:  # noqa: D401
        return

"""Value objects and callable contracts shared by repository validators."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol


@dataclass(frozen=True)
class VerifierConfig:
    active_task_dirs: tuple[str, ...]
    required_sections: Mapping[str, tuple[str, ...]]
    tier_a_notebooks: tuple[str, ...]

    def __post_init__(self) -> None:
        snapshot = {
            key: tuple(value) for key, value in self.required_sections.items()
        }
        object.__setattr__(self, "required_sections", MappingProxyType(snapshot))


@dataclass
class Finding:
    id: str
    check: str
    severity: str
    location: str
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    findings: list[Finding] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


class RunCommand(Protocol):
    def __call__(
        self, cmd: list[str], cwd: Path, timeout: int | None = None
    ) -> tuple[int, str, str]: ...

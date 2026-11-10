"""Ready-made knowledge bases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    title: str
    noun: str  # what one class is called in the interface
    features: tuple[str, ...]
    rows: tuple[tuple[str, tuple[int, ...]], ...]


INSTRUMENTS = Preset(
    title="Musical instruments",
    noun="instrument",
    features=("String", "Bowed", "Plucked", "Wind", "Keyboard", "Percussion", "Electric", "Wooden"),
    rows=(
        ("Violin", (1, 1, 0, 0, 0, 0, 0, 1)),
        ("Guitar", (1, 0, 1, 0, 0, 0, 0, 1)),
        ("Electric guitar", (1, 0, 1, 0, 0, 0, 1, 0)),
        ("Flute", (0, 0, 0, 1, 0, 0, 0, 0)),
        ("Drum", (0, 0, 0, 0, 0, 1, 0, 0)),
        ("Xylophone", (0, 0, 0, 0, 0, 1, 0, 1)),
        ("Piano", (1, 0, 0, 0, 1, 1, 0, 1)),
        ("Synthesizer", (0, 0, 0, 0, 1, 0, 1, 0)),
    ),
)

# The worked example from the lecture.
FLYING_OBJECTS = Preset(
    title="Lecture example: bird, plane, glider",
    noun="object",
    features=("Wings", "Tail", "Beak", "Engine", "Plumage", "Landing gear"),
    rows=(
        ("Bird", (1, 1, 1, 0, 1, 0)),
        ("Plane", (1, 1, 0, 1, 0, 1)),
        ("Glider", (1, 1, 0, 0, 0, 0)),
    ),
)

PRESETS: dict[str, Preset] = {preset.title: preset for preset in (INSTRUMENTS, FLYING_OBJECTS)}

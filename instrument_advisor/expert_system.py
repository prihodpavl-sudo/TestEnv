"""Expert system on the imitation principle.

The model is linear: one equation (a column of the weight matrix) per class,
``y_j = a_1j*x_1 + ... + a_nj*x_n``. Training shows the system examples and
applies "carrot and stick": the example vector is added to the column of the
right class and subtracted from the others. In examination mode the example is
multiplied by the matrix; each result is the "voice" of a demon and the
loudest demon names the class.

Nothing here knows about the user interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Voices closer than this are treated as equally loud.
TIE_TOLERANCE = 1e-9


class Rule(str, Enum):
    """When and whom the teacher rewards and punishes."""

    MISTAKES = "mistakes"  # repeat only the examples the system gets wrong
    EVERY_EXAMPLE = "every"  # reinforce every example in every epoch


@dataclass(frozen=True)
class TrainingSettings:
    rule: Rule = Rule.MISTAKES
    step: float = 1.0
    max_epochs: int = 50
    normalize: bool = False  # pull weights back into the [-1, 1] hypercube after each lesson
    stop_when_error_free: bool = True


@dataclass(frozen=True, eq=False)
class KnowledgeBase:
    """Training examples: named features, named classes, one feature vector per example."""

    features: tuple[str, ...]
    classes: tuple[str, ...]
    examples: np.ndarray  # shape (n_examples, n_features), values in [0, 1]
    labels: tuple[int, ...]  # class index of each example

    @classmethod
    def from_rows(cls, features: Sequence[str], rows: Iterable[tuple[str, Sequence[float]]]) -> KnowledgeBase:
        """Build from (class name, feature vector) rows; classes keep first-seen order."""
        classes: list[str] = []
        labels: list[int] = []
        vectors: list[list[float]] = []
        for name, vector in rows:
            if len(vector) != len(features):
                raise ValueError(f"{name!r} has {len(vector)} values but there are {len(features)} features")
            if name not in classes:
                classes.append(name)
            labels.append(classes.index(name))
            vectors.append([float(v) for v in vector])
        examples = np.array(vectors, dtype=float).reshape(len(vectors), len(features))
        return cls(tuple(features), tuple(classes), examples, tuple(labels))

    def prototypes(self) -> np.ndarray:
        """Mean feature vector of each class, shape (n_classes, n_features)."""
        labels = np.array(self.labels, dtype=int)
        return np.array([self.examples[labels == c].mean(axis=0) for c in range(len(self.classes))])

    def problems(self) -> list[str]:
        """Reasons the data cannot be learned cleanly, in plain language."""
        found: list[str] = []
        if len(self.classes) < 2:
            found.append("At least two different classes are needed to tell them apart.")
        if not self.features:
            found.append("At least one feature is needed.")
        for vector, label in zip(self.examples, self.labels):
            if not vector.any():
                found.append(
                    f"{self.classes[label]} has no features set, so no demon can ever speak up for it."
                )
        seen: dict[tuple[float, ...], int] = {}
        clashes: list[tuple[int, int]] = []
        for vector, label in zip(self.examples, self.labels):
            other = seen.setdefault(tuple(vector), label)
            if other != label and (other, label) not in clashes:
                clashes.append((other, label))
        for other, label in clashes:
            found.append(
                f"{self.classes[other]} and {self.classes[label]} have identical features, "
                "so they cannot be told apart."
            )
        return found


@dataclass(frozen=True, eq=False)
class Verdict:
    """Result of examining one object."""

    demons: np.ndarray  # voice of each class
    winner: int | None  # None when several demons are equally loud
    leaders: tuple[int, ...]  # the loudest class or classes
    margin: float  # lead of the loudest demon over the next one, 0 on a tie

    @property
    def ranking(self) -> tuple[int, ...]:
        """Class indices from loudest to quietest."""
        return tuple(int(i) for i in np.argsort(-self.demons, kind="stable"))


@dataclass(frozen=True, eq=False)
class Lesson:
    """One example shown to the system during an epoch."""

    example: int
    label: int
    verdict: Verdict  # what the system said before being corrected
    recognised: bool
    taught: bool  # whether the weights were changed


@dataclass(frozen=True, eq=False)
class Epoch:
    number: int
    lessons: tuple[Lesson, ...]
    weights: np.ndarray  # matrix at the end of the epoch
    exam: tuple[Verdict, ...]  # every example examined on that matrix
    exam_errors: int

    @property
    def mistakes(self) -> int:
        """Examples the system got wrong while they were being shown."""
        return sum(not lesson.recognised for lesson in self.lessons)


@dataclass(frozen=True, eq=False)
class TrainingResult:
    knowledge: KnowledgeBase
    settings: TrainingSettings
    epochs: tuple[Epoch, ...]

    @property
    def weights(self) -> np.ndarray:
        if self.epochs:
            return self.epochs[-1].weights
        return np.zeros((len(self.knowledge.features), len(self.knowledge.classes)))

    @property
    def converged(self) -> bool:
        """True when the final examination recognises every example."""
        return bool(self.epochs) and self.epochs[-1].exam_errors == 0

    @property
    def error_curve(self) -> list[int]:
        return [epoch.exam_errors for epoch in self.epochs]

    @property
    def mistake_curve(self) -> list[int]:
        return [epoch.mistakes for epoch in self.epochs]


def examine(weights: np.ndarray, vector: Sequence[float]) -> Verdict:
    """Examination mode: ask every demon and pick the loudest."""
    demons = np.asarray(vector, dtype=float) @ weights
    if demons.size == 0:
        return Verdict(demons, None, (), 0.0)
    top = demons.max()
    leaders = tuple(int(i) for i in np.flatnonzero(demons >= top - TIE_TOLERANCE))
    if len(leaders) > 1:
        return Verdict(demons, None, leaders, 0.0)
    rest = np.delete(demons, leaders[0])
    margin = float(top - rest.max()) if rest.size else float("inf")
    return Verdict(demons, leaders[0], leaders, margin)


def _teach(weights: np.ndarray, vector: np.ndarray, label: int, settings: TrainingSettings) -> np.ndarray:
    """Carrot and stick: reward the right class with the example, punish all the others."""
    signs = -np.ones(weights.shape[1])
    signs[label] = 1.0
    weights = weights + settings.step * np.outer(vector, signs)
    if settings.normalize:
        weights = np.clip(weights, -1.0, 1.0)
    return weights


def train(knowledge: KnowledgeBase, settings: TrainingSettings = TrainingSettings()) -> TrainingResult:
    """Training mode: start from a zero matrix and run epochs over the examples.

    After each epoch the system is switched to examination mode and tested on
    everything it was taught; training stops early once it makes no errors
    (unless ``stop_when_error_free`` is off).
    """
    weights = np.zeros((len(knowledge.features), len(knowledge.classes)))
    epochs: list[Epoch] = []
    for number in range(1, settings.max_epochs + 1):
        lessons: list[Lesson] = []
        for index, (vector, label) in enumerate(zip(knowledge.examples, knowledge.labels)):
            verdict = examine(weights, vector)
            recognised = verdict.winner == label
            taught = settings.rule is Rule.EVERY_EXAMPLE or not recognised
            if taught:
                weights = _teach(weights, vector, label, settings)
            lessons.append(Lesson(index, label, verdict, recognised, taught))
        exam = tuple(examine(weights, vector) for vector in knowledge.examples)
        errors = sum(v.winner != label for v, label in zip(exam, knowledge.labels))
        epochs.append(Epoch(number, tuple(lessons), weights.copy(), exam, errors))
        if settings.stop_when_error_free and errors == 0:
            break
    return TrainingResult(knowledge, settings, tuple(epochs))


def _number(value: float) -> str:
    return f"{value:.4g}"


def equations(weights: np.ndarray) -> list[str]:
    """The model as a system of equations, e.g. ``y1 = -x1 + 2·x3``."""
    lines = []
    for j in range(weights.shape[1]):
        terms = []
        for i, a in enumerate(weights[:, j]):
            if abs(a) < TIE_TOLERANCE:
                continue
            sign = "-" if a < 0 else "+"
            size = "" if abs(abs(a) - 1) < TIE_TOLERANCE else f"{_number(abs(a))}·"
            terms.append((sign, f"{size}x{i + 1}"))
        if not terms:
            lines.append(f"y{j + 1} = 0")
            continue
        first_sign, first = terms[0]
        body = ("-" if first_sign == "-" else "") + first
        body += "".join(f" {sign} {term}" for sign, term in terms[1:])
        lines.append(f"y{j + 1} = {body}")
    return lines


def feature_influence(weights: np.ndarray) -> np.ndarray:
    """How strongly each feature separates the classes: the spread of its weights.

    A feature whose weights are the same for every class changes all voices
    equally and so never changes the answer.
    """
    if weights.shape[1] == 0:
        return np.zeros(weights.shape[0])
    return weights.max(axis=1) - weights.min(axis=1)


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    return np.abs(points[:, None, :] - points[None, :, :]).sum(axis=2)


def world_distances(knowledge: KnowledgeBase) -> np.ndarray:
    """Distances between classes as the data describes them (Hamming for 0/1 features)."""
    return _pairwise_distances(knowledge.prototypes())


def mind_distances(weights: np.ndarray) -> np.ndarray:
    """Distances between classes as the trained model sees them (between weight columns)."""
    return _pairwise_distances(weights.T)


def layout_2d(distances: np.ndarray) -> np.ndarray:
    """Place points on a plane so their distances are preserved as well as possible (classical MDS)."""
    n = distances.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distances**2) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    coords = vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((n, 2 - coords.shape[1]))])
    return coords

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

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

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


def _term(size: float, variable: str) -> str:
    return variable if abs(size - 1) < TIE_TOLERANCE else f"{_number(size)}·{variable}"


def equations(weights: np.ndarray, variables: Sequence[str] | None = None) -> list[str]:
    """The model as a system of equations, e.g. ``y1 = -x1 + 2·x3``.

    ``variables`` names the rows of ``weights``; by default they are x1, x2, ...
    """
    names = list(variables) if variables is not None else [f"x{i + 1}" for i in range(weights.shape[0])]
    lines = []
    for j in range(weights.shape[1]):
        terms = [(a, name) for a, name in zip(weights[:, j], names) if abs(a) >= TIE_TOLERANCE]
        if not terms:
            lines.append(f"y{j + 1} = 0")
            continue
        (a, name), rest = terms[0], terms[1:]
        body = ("-" if a < 0 else "") + _term(abs(a), name)
        body += "".join(f" {'-' if a < 0 else '+'} {_term(abs(a), name)}" for a, name in rest)
        lines.append(f"y{j + 1} = {body}")
    return lines


@dataclass(frozen=True, eq=False)
class Border:
    """A dividing line ``a·X + b·Y = 0`` on a feature plane, clipped to where it divides two regions.

    Group ``upper`` is the loudest where ``a·X + b·Y > 0``, group ``lower`` where it is ``< 0``.
    Either ``a > 0``, or ``a == 0`` and ``b > 0``.
    """

    upper: int
    lower: int
    a: float
    b: float
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(frozen=True, eq=False)
class FeaturePlane:
    """Class regions on the plane of two features, with every other feature held at 0.

    There each class's voice is ``a·X + b·Y``, where a and b are its weights for
    the two features. Classes with equal weights for both always speak equally
    loudly on this plane, so they form one group and share a region.
    """

    fx: int
    fy: int
    groups: tuple[tuple[int, ...], ...]
    coefficients: np.ndarray  # (a, b) of each group, shape (n_groups, 2)
    borders: tuple[Border, ...]  # ordered anticlockwise around the origin
    low: float  # the plane is examined on the square [low, high]²
    high: float

    @property
    def regions(self) -> tuple[int, ...]:
        """Groups that are the loudest somewhere on the plane."""
        if len(self.groups) == 1:
            return (0,)
        # Every region is a wedge at the origin, inside the square, so it has a border.
        return tuple(sorted({g for border in self.borders for g in (border.upper, border.lower)}))

    def winners(self, points: np.ndarray) -> np.ndarray:
        """Loudest group at each (X, Y) point, or -1 where groups are equally loud."""
        voices = np.asarray(points, dtype=float) @ self.coefficients.T
        loudest = voices >= voices.max(axis=1, keepdims=True) - TIE_TOLERANCE
        return np.where(loudest.sum(axis=1) > 1, -1, voices.argmax(axis=1))


def feature_plane(
    weights: np.ndarray, fx: int, fy: int, low: float = -0.5, high: float = 1.5
) -> FeaturePlane:
    """Regions and dividing lines on the plane of features ``fx`` and ``fy``.

    The voices have no constant term, so every dividing line passes through the
    origin and the regions are wedges around it. A line is kept only where it
    actually separates two regions inside the square [low, high]², which must
    contain the origin.
    """
    if not low < 0 < high:
        raise ValueError("The square must contain the origin.")
    pairs = weights[[fx, fy], :].T
    groups: list[list[int]] = []
    for c, pair in enumerate(pairs):
        same = next((g for g in groups if np.abs(pairs[g[0]] - pair).max() < TIE_TOLERANCE), None)
        if same is None:
            groups.append([c])
        else:
            same.append(c)
    coefficients = np.array([pairs[g[0]] for g in groups]).reshape(len(groups), 2)
    borders = [
        border
        for g in range(len(groups))
        for h in range(g + 1, len(groups))
        if (border := _border(coefficients, g, h, low, high)) is not None
    ]
    borders.sort(key=_angle)
    return FeaturePlane(fx, fy, tuple(tuple(g) for g in groups), coefficients, tuple(borders), low, high)


def _angle(border: Border) -> float:
    x, y = np.add(border.start, border.end) / 2
    return math.atan2(y, x) % (2 * math.pi)


def _border(coefficients: np.ndarray, g: int, h: int, low: float, high: float) -> Border | None:
    """The part of the line where groups g and h are equally loud and no other group is louder."""
    a, b = (float(v) if abs(v) >= TIE_TOLERANCE else 0.0 for v in coefficients[g] - coefficients[h])
    if a < 0 or (a == 0 and b < 0):
        g, h, a, b = h, g, -a, -b
    normal = np.array([a, b]) / math.hypot(a, b)  # points into the region of g
    along = np.array([-normal[1], normal[0]])
    # Points of the line are t·along; keep the t that stay inside the square ...
    t_min, t_max = -math.inf, math.inf
    for component in along:
        if abs(component) >= TIE_TOLERANCE:
            first, second = sorted((low / component, high / component))
            t_min, t_max = max(t_min, first), min(t_max, second)
    # ... where no other group k is louder than g and h.
    for k in range(len(coefficients)):
        if k in (g, h):
            continue
        slope = float((coefficients[g] - coefficients[k]) @ along)
        if slope >= TIE_TOLERANCE:
            t_min = max(t_min, 0.0)
        elif slope <= -TIE_TOLERANCE:
            t_max = min(t_max, 0.0)
        elif (coefficients[g] - coefficients[k]) @ normal < TIE_TOLERANCE or (
            coefficients[k] - coefficients[h]
        ) @ normal < TIE_TOLERANCE:
            # k is as loud as g and h all along the line, so the line only divides g from h
            # if each of them beats k on its own side.
            return None
    if t_max - t_min < TIE_TOLERANCE:
        return None
    start, end = t_min * along, t_max * along
    return Border(g, h, a, b, (float(start[0]), float(start[1])), (float(end[0]), float(end[1])))


def _whole_ratio(*values: float) -> list[float]:
    """Scale values by a positive factor to the smallest whole numbers in the same ratio."""
    fractions = [Fraction(v).limit_denominator(1000) for v in values]
    scale = math.lcm(*(f.denominator for f in fractions))
    whole = [int(f * scale) for f in fractions]
    divisor = math.gcd(*whole) or 1
    return [n / divisor for n in whole]


def line_equation(border: Border, x: str, y: str, relation: str = "=") -> str:
    """The border as an equation in the plane's two features, e.g. ``x6 = 3·x8``.

    With ``relation`` ">" it describes the side of ``border.upper``, with "<" the side of ``border.lower``.
    """
    a, b = _whole_ratio(border.a, border.b)
    if b == 0:
        return f"{x} {relation} 0"
    if a == 0:
        return f"{y} {relation} 0"
    if b < 0:
        return f"{_term(a, x)} {relation} {_term(-b, y)}"
    return f"{_term(a, x)} + {_term(b, y)} {relation} 0"


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

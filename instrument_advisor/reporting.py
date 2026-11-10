"""Plain-language reports on how training went. No interface code here."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from expert_system import KnowledgeBase, Rule, TrainingResult, TrainingSettings, Verdict, train

RULE_LABELS = {Rule.MISTAKES: "Learn from mistakes", Rule.EVERY_EXAMPLE: "Reinforce everything"}
WITHOUT_NORMALISATION = "Without normalisation"
WITH_NORMALISATION = "With normalisation"


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def listing(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def answer_text(verdict: Verdict, classes: Sequence[str]) -> str:
    """What the system said about one object."""
    if verdict.winner is not None:
        return classes[verdict.winner]
    if len(verdict.leaders) == len(classes):
        return "Undecided (all demons equally loud)"
    return "Undecided: " + " / ".join(classes[i] for i in verdict.leaders)


def describe_failure(verdict: Verdict, label: int, classes: Sequence[str]) -> str:
    if verdict.winner is None:
        tied = listing([classes[i] for i in verdict.leaders])
        return f"{classes[label]}: the demons for {tied} are equally loud"
    return f"{classes[label]} is taken for {classes[verdict.winner]}"


def training_summary(result: TrainingResult, noun: str) -> tuple[bool, str, str]:
    """(success, headline, detail) describing how the training went."""
    settings = result.settings
    run = len(result.epochs)
    total = len(result.knowledge.labels)
    last = result.epochs[-1]
    ran = f"all {run} epochs were run" if run > 1 else "the single epoch was run"
    if result.converged:
        headline = (
            f"Training complete — {ran} and the system now recognises "
            f"all {total} {noun} examples without a single error."
        )
        if settings.stop_when_error_free and run < settings.max_epochs:
            detail = (
                f"It stopped after epoch {run} because the examination showed no errors, "
                f"so the remaining epochs (limit {settings.max_epochs}) were not needed."
            )
        else:
            detail = f"The full programme of {plural(settings.max_epochs, 'epoch')} was carried out."
        return True, headline, detail

    classes = result.knowledge.classes
    failures = [
        describe_failure(verdict, label, classes)
        for verdict, label in zip(last.exam, result.knowledge.labels)
        if verdict.winner != label
    ]
    headline = (
        f"Training finished — {ran}, but {last.exam_errors} of {total} {noun} examples "
        f"{'is' if last.exam_errors == 1 else 'are'} still not recognised."
    )
    detail = "; ".join(failures) + "."
    if result.knowledge.problems():
        detail += " See the data problems below."
    elif settings.rule is Rule.EVERY_EXAMPLE:
        detail += f" Try the “{RULE_LABELS[Rule.MISTAKES]}” rule in the sidebar."
    elif settings.normalize:
        detail += " Try a smaller step or switch normalisation off in the sidebar."
    else:
        detail += " Try a higher epoch limit in the sidebar."
    return False, headline, detail


def step_outcomes(
    knowledge: KnowledgeBase, settings: TrainingSettings, steps: Sequence[float]
) -> dict[str, list[int | None]]:
    """Train from scratch with each step, with and without normalisation.

    Each entry is the number of epochs needed to become error-free, or None
    when the epoch limit was reached first.
    """
    outcomes: dict[str, list[int | None]] = {}
    for label, normalize in ((WITHOUT_NORMALISATION, False), (WITH_NORMALISATION, True)):
        runs = [
            train(knowledge, replace(settings, step=step, normalize=normalize, stop_when_error_free=True))
            for step in steps
        ]
        outcomes[label] = [len(run.epochs) if run.converged else None for run in runs]
    return outcomes


def step_conclusions(steps: Sequence[float], outcomes: dict[str, list[int | None]], limit: int) -> list[str]:
    """One sentence per group, derived from the measured outcomes."""
    lines = []
    for label, needed in outcomes.items():
        reached = {step: n for step, n in zip(steps, needed) if n is not None}
        failed = [f"{step:g}" for step, n in zip(steps, needed) if n is None]
        if not reached:
            lines.append(f"**{label}:** no step made the system error-free within {plural(limit, 'epoch')}.")
        elif not failed and len(set(reached.values())) == 1:
            line = (
                f"**{label}:** the step made no difference; "
                f"every run needed {plural(next(iter(reached.values())), 'epoch')}."
            )
            if label == WITHOUT_NORMALISATION:
                line += (
                    " Starting from a zero matrix, the step only multiplies all weights by the same "
                    "number, which does not change which demon is loudest."
                )
            lines.append(line)
        else:
            best = min(reached.values())
            fastest = listing([f"{step:g}" for step, n in reached.items() if n == best])
            line = f"**{label}:** fastest with step {fastest} ({plural(best, 'epoch')})"
            if failed:
                line += (
                    f"; with step {listing(failed)} the system was still making errors "
                    f"after {plural(limit, 'epoch')}"
                )
            lines.append(line + ".")
    return lines

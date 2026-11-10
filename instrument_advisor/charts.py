"""Plotly figures for the expert system. Each function takes results and returns a figure."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from expert_system import (
    TIE_TOLERANCE,
    FeaturePlane,
    KnowledgeBase,
    TrainingResult,
    Verdict,
    feature_influence,
    layout_2d,
    line_equation,
    mind_distances,
    world_distances,
)

PALETTE = qualitative.Safe  # colour-blind-safe categorical colours
ACCENT = "#2E86AB"
SECOND = "#F18F01"
WARN = "#E0A100"
MUTED = "#B4BCC6"
INK = "#1F2933"
SHARED = "#7B8794"  # region of classes that are equally loud on a feature plane


def class_colour(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _tidy(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def learning_curve(result: TrainingResult) -> go.Figure:
    """Errors by epoch: during the lessons and in the examination that follows each epoch."""
    numbers = [epoch.number for epoch in result.epochs]
    total = len(result.knowledge.labels)
    few = len(numbers) <= 20
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=numbers,
            y=result.mistake_curve,
            name="Mistakes during the lessons",
            mode="lines+markers",
            line=dict(color=MUTED, width=2, dash="dot"),
            marker=dict(size=7),
            hovertemplate="Epoch %{x}: %{y} mistakes during the lessons<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=numbers,
            y=result.error_curve,
            name="Errors in the examination after the epoch",
            mode="lines+markers+text" if few else "lines+markers",
            text=[str(v) for v in result.error_curve],
            textposition="top center",
            line=dict(color=ACCENT, width=3),
            marker=dict(size=10),
            hovertemplate="Epoch %{x}: %{y} errors in the examination<extra></extra>",
        )
    )
    fig.update_xaxes(title="Epoch", dtick=1 if few else None, range=[0.5, max(numbers) + 0.5])
    fig.update_yaxes(
        title="Examples not recognised", range=[-0.4, total + 0.9], dtick=1 if total <= 12 else None
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return _tidy(fig, 380)


def weight_matrix(weights: np.ndarray, features: Sequence[str], classes: Sequence[str]) -> go.Figure:
    """The model itself: one column of coefficients per class."""
    limit = max(float(np.abs(weights).max()), 1.0)
    fig = go.Figure(
        go.Heatmap(
            z=weights,
            x=[f"y{j + 1} · {name}" for j, name in enumerate(classes)],
            y=[f"x{i + 1} · {name}" for i, name in enumerate(features)],
            colorscale="RdBu",
            zmin=-limit,
            zmax=limit,
            texttemplate="%{z:.3~g}",
            hovertemplate="%{y} → %{x}<br>weight %{z:.4~g}<extra></extra>",
            colorbar=dict(title="Weight"),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top", tickangle=-30)
    return _tidy(fig, 90 + 42 * max(len(features), 4))


def demon_voices(verdict: Verdict, classes: Sequence[str]) -> go.Figure:
    """How loudly each class answers for the examined object, loudest on top.

    Voices are often all negative, so bars are drawn from just below the
    quietest one: the longest bar is always the loudest demon.
    """
    order = verdict.ranking[::-1]
    lead_colour = ACCENT if verdict.winner is not None else WARN
    low, high = float(verdict.demons.min()), float(verdict.demons.max())
    base = low - max(high - low, 1.0) * 0.12
    fig = go.Figure(
        go.Bar(
            x=[verdict.demons[i] - base for i in order],
            base=base,
            y=[classes[i] for i in order],
            orientation="h",
            marker_color=[lead_colour if i in verdict.leaders else MUTED for i in order],
            text=[f"{verdict.demons[i]:.3g}" for i in order],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Voice of the demon (longer bar = louder)", zeroline=False)
    return _tidy(fig, 70 + 34 * max(len(classes), 3))


def class_regions(plane: FeaturePlane, knowledge: KnowledgeBase) -> go.Figure:
    """The plane of two features coloured by the loudest class, with the numbered dividing lines.

    Every other feature is held at 0. Classes that are equally loud on this
    plane share a grey region.
    """
    classes = knowledge.classes
    fx, fy = plane.fx, plane.fy
    axis = np.linspace(plane.low, plane.high, 161)
    xx, yy = np.meshgrid(axis, axis)
    winners = plane.winners(np.column_stack([xx.ravel(), yy.ravel()]))
    names = [" / ".join(classes[c] for c in group) for group in plane.groups]

    n = len(plane.groups)
    scale = []
    for g, group in enumerate(plane.groups):
        colour = class_colour(group[0]) if len(group) == 1 else SHARED
        scale += [(g / n, colour), ((g + 1) / n, colour)]
    fig = go.Figure(
        go.Heatmap(
            x=axis,
            y=axis,
            z=np.where(winners < 0, np.nan, winners).reshape(xx.shape),
            text=np.array([names[w] if w >= 0 else "" for w in winners]).reshape(xx.shape),
            colorscale=scale,
            zmin=-0.5,
            zmax=n - 0.5,
            opacity=0.45,
            showscale=False,
            hoverongaps=False,
            hovertemplate="Region of %{text}<extra></extra>",
        )
    )
    if any(len(group) > 1 for group in plane.groups):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Shared region",
                marker=dict(size=12, symbol="square", color=SHARED, opacity=0.6),
            )
        )

    if plane.borders:
        x_name, y_name = f"x{fx + 1}", f"x{fy + 1}"
        xs: list[float | None] = []
        ys: list[float | None] = []
        for border in plane.borders:
            xs += [border.start[0], border.end[0], None]
            ys += [border.start[1], border.end[1], None]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Dividing line",
                line=dict(color=INK, width=2.5),
                hoverinfo="skip",
            )
        )
        middles = np.array([np.add(border.start, border.end) / 2 for border in plane.borders])
        fig.add_trace(
            go.Scatter(
                x=middles[:, 0],
                y=middles[:, 1],
                mode="markers+text",
                text=[str(number) for number in range(1, len(plane.borders) + 1)],
                textfont=dict(color="white", size=12),
                marker=dict(size=22, color=INK, line=dict(color="white", width=1.5)),
                hovertext=[
                    f"Line {number}: {line_equation(border, x_name, y_name)}"
                    for number, border in enumerate(plane.borders, start=1)
                ],
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            )
        )

    # Classes that share a corner are fanned out around it so each stays visible.
    prototypes = knowledge.prototypes()
    corners: dict[tuple[float, float], list[int]] = {}
    for c in range(len(classes)):
        corners.setdefault((float(prototypes[c, fx]), float(prototypes[c, fy])), []).append(c)
    for c in range(len(classes)):
        x, y = float(prototypes[c, fx]), float(prototypes[c, fy])
        members = corners[(x, y)]
        angle = 2 * np.pi * members.index(c) / len(members)
        radius = 0.0 if len(members) == 1 else 0.11
        fig.add_trace(
            go.Scatter(
                x=[x + radius * np.cos(angle)],
                y=[y + radius * np.sin(angle)],
                mode="markers",
                name=classes[c],
                marker=dict(size=15, color=class_colour(c), line=dict(color="white", width=2)),
                hovertemplate=f"{classes[c]}<br>{knowledge.features[fx]}: {x:g}"
                f"<br>{knowledge.features[fy]}: {y:g}<extra></extra>",
            )
        )
    ticks = dict(
        tickvals=[0, 1],
        ticktext=["No (0)", "Yes (1)"],
        range=[plane.low, plane.high],
        zeroline=False,
        constrain="domain",
    )
    fig.update_xaxes(title=f"x{fx + 1} · {knowledge.features[fx]}", **ticks)
    fig.update_yaxes(title=f"x{fy + 1} · {knowledge.features[fy]}", scaleanchor="x", **ticks)
    return _tidy(fig, 480)


def _align(reference: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate or mirror ``points`` to lie as close to ``reference`` as possible."""
    u, _, vt = np.linalg.svd(points.T @ reference)
    return points @ (u @ vt)


def _unit(points: np.ndarray) -> np.ndarray:
    spread = np.sqrt((points**2).sum(axis=1).mean())
    return points / spread if spread > 0 else points


def class_maps(knowledge: KnowledgeBase, weights: np.ndarray) -> go.Figure:
    """Two maps of the classes: by their real features and by what the model learned."""
    world = _unit(layout_2d(world_distances(knowledge)))
    mind = _align(world, _unit(layout_2d(mind_distances(weights))))
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("The world as the data describes it", "The world as the system learned it"),
        horizontal_spacing=0.06,
    )
    count = len(knowledge.classes)
    for column, points in enumerate((world, mind), start=1):
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers+text",
                text=list(knowledge.classes),
                # Alternating sides keeps neighbouring labels from covering each other.
                textposition=["top center" if c % 2 == 0 else "bottom center" for c in range(count)],
                marker=dict(
                    size=14,
                    color=[class_colour(c) for c in range(count)],
                    line=dict(color="white", width=1.5),
                ),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
                cliponaxis=False,
            ),
            row=1,
            col=column,
        )
    reach = max(float(np.abs(world).max()), float(np.abs(mind).max()), 1.0) * 1.25
    axes = dict(showticklabels=False, zeroline=False, range=[-reach, reach], constrain="domain")
    fig.update_xaxes(**axes)
    fig.update_yaxes(**axes)
    fig.update_yaxes(scaleanchor="x", row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", row=1, col=2)
    return _tidy(fig, 520)


def influence(weights: np.ndarray, features: Sequence[str]) -> go.Figure:
    """Which features move the answer most, strongest on top."""
    values = feature_influence(weights)
    order = np.argsort(values, kind="stable")
    fig = go.Figure(
        go.Bar(
            x=values[order],
            y=[f"x{i + 1} · {features[i]}" for i in order],
            orientation="h",
            marker_color=[ACCENT if values[i] > TIE_TOLERANCE else MUTED for i in order],
            text=[f"{values[i]:.3g}" if values[i] > TIE_TOLERANCE else "no effect" for i in order],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: spread of weights %{x:.4g}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Spread of the feature's weights across classes")
    return _tidy(fig, 70 + 34 * max(len(features), 3))


def step_study(steps: Sequence[float], outcomes: dict[str, list[int | None]], limit: int) -> go.Figure:
    """Epochs needed to become error-free for each step; hatched bars never got there."""
    fig = go.Figure()
    for colour, (label, needed) in zip((ACCENT, SECOND), outcomes.items()):
        fig.add_trace(
            go.Bar(
                x=[f"{step:g}" for step in steps],
                y=[limit if n is None else n for n in needed],
                name=label,
                marker=dict(color=colour, pattern_shape=["/" if n is None else "" for n in needed]),
                text=["not reached" if n is None else str(n) for n in needed],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="Step %{x}: %{text}<extra>" + label + "</extra>",
            )
        )
    fig.update_xaxes(title="Step h", type="category")
    fig.update_yaxes(title="Epochs until error-free", range=[0, limit * 1.15])
    fig.update_layout(barmode="group", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return _tidy(fig, 400)

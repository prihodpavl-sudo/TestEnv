"""Behaviour of the expert system, checked against the lecture's worked example and the lab draft's output."""

import numpy as np

from expert_system import (
    KnowledgeBase,
    Rule,
    TrainingSettings,
    equations,
    examine,
    feature_influence,
    feature_plane,
    layout_2d,
    line_equation,
    mind_distances,
    train,
    world_distances,
)
from presets import FLYING_OBJECTS, INSTRUMENTS


def knowledge(preset):
    return KnowledgeBase.from_rows(preset.features, preset.rows)


# Matrix from the lecture after iterations 1-3 (bird, plane, glider shown once each).
LECTURE_AFTER_THREE_LESSONS = np.array(
    [[-1, -1, -1], [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, -1, -1], [-1, 1, -1]], dtype=float
)
# Matrix from the lecture after iteration 4 (glider repeated) and normalisation.
LECTURE_FINAL = np.array(
    [[-1, -1, 0], [-1, -1, 0], [1, -1, -1], [-1, 1, -1], [1, -1, -1], [-1, 1, -1]], dtype=float
)


def test_first_epoch_reproduces_lecture_matrix_and_leaves_glider_unrecognised():
    result = train(knowledge(FLYING_OBJECTS), TrainingSettings(max_epochs=1))

    np.testing.assert_array_equal(result.weights, LECTURE_AFTER_THREE_LESSONS)
    bird, plane, glider = result.epochs[0].exam
    np.testing.assert_array_equal(bird.demons, [0, -4, -4])
    np.testing.assert_array_equal(plane.demons, [-4, 0, -4])
    np.testing.assert_array_equal(glider.demons, [-2, -2, -2])
    assert (bird.winner, plane.winner, glider.winner) == (0, 1, None)
    assert not result.converged


def test_second_epoch_repeats_only_the_glider_as_in_lecture_iteration_four():
    result = train(knowledge(FLYING_OBJECTS), TrainingSettings(max_epochs=2))

    assert [lesson.taught for lesson in result.epochs[1].lessons] == [False, False, True]
    # The lecture's matrix after iteration 4, before it is normalised.
    np.testing.assert_array_equal(
        result.weights,
        [[-2, -2, 0], [-2, -2, 0], [1, -1, -1], [-1, 1, -1], [1, -1, -1], [-1, 1, -1]],
    )


def test_relearning_can_make_the_system_forget_then_recover():
    # Without normalisation, repeating the glider makes the bird tie with it (lecture, corollary 3).
    result = train(knowledge(FLYING_OBJECTS))

    assert result.error_curve == [1, 2, 0]
    assert result.converged


def test_normalised_training_reaches_the_lecture_final_matrix():
    result = train(knowledge(FLYING_OBJECTS), TrainingSettings(normalize=True))

    np.testing.assert_array_equal(result.weights, LECTURE_FINAL)
    assert result.converged
    bird, plane, glider = result.epochs[-1].exam
    np.testing.assert_array_equal(bird.demons, [0, -4, -2])
    np.testing.assert_array_equal(plane.demons, [-4, 0, -2])
    np.testing.assert_array_equal(glider.demons, [-2, -2, 0])


def test_unknown_rocket_is_recognised_as_a_plane():
    rocket = [0, 1, 0, 1, 0, 0]

    verdict = examine(LECTURE_FINAL, rocket)

    np.testing.assert_array_equal(verdict.demons, [-2, 0, -1])
    assert verdict.winner == 1
    assert verdict.margin == 1
    assert verdict.ranking == (1, 2, 0)


def test_equations_match_the_lecture():
    assert equations(LECTURE_FINAL) == [
        "y1 = -x1 - x2 + x3 - x4 + x5 - x6",
        "y2 = -x1 - x2 - x3 + x4 - x5 + x6",
        "y3 = -x3 - x4 - x5 - x6",
    ]


def test_equations_on_the_beak_engine_plane_match_the_lecture():
    # Lecture: Y1 = ... + x3 - x4 + ..., Y2 = ... - x3 + x4 + ..., Y3 = ... - x3 - x4 + ...
    assert equations(LECTURE_FINAL[[2, 3], :], ("x3", "x4")) == [
        "y1 = x3 - x4",
        "y2 = -x3 + x4",
        "y3 = -x3 - x4",
    ]


def describe(plane, x, y):
    """(equation, side of upper, side of lower, sorted ends) of each dividing line, in order."""
    return [
        (
            line_equation(b, x, y),
            plane.groups[b.upper],
            line_equation(b, x, y, ">"),
            plane.groups[b.lower],
            line_equation(b, x, y, "<"),
            sorted([b.start, b.end]),
        )
        for b in plane.borders
    ]


def test_dividing_lines_on_the_beak_engine_plane_match_the_lecture():
    # Lecture: bird (x3 > x4) & (x3 > 0), plane (x4 > x3) & (x4 > 0), glider (x3 < 0) & (x4 < 0),
    # with borders x3 = x4, x4 = 0, x3 = 0.
    plane = feature_plane(LECTURE_FINAL, 2, 3)

    lines = describe(plane, "x3", "x4")

    assert [line[:5] for line in lines] == [
        ("x3 = x4", (0,), "x3 > x4", (1,), "x3 < x4"),
        ("x4 = 0", (1,), "x4 > 0", (2,), "x4 < 0"),
        ("x3 = 0", (0,), "x3 > 0", (2,), "x3 < 0"),
    ]
    ends = [line[5] for line in lines]
    np.testing.assert_allclose(ends[0], [(0, 0), (1.5, 1.5)], atol=1e-12)
    np.testing.assert_allclose(ends[1], [(-0.5, 0), (0, 0)], atol=1e-12)
    np.testing.assert_allclose(ends[2], [(0, -0.5), (0, 0)], atol=1e-12)
    assert plane.regions == (0, 1, 2)


def test_classes_with_equal_weights_share_a_region_and_a_class_can_have_none():
    # Columns A, B, C, D, E on features x1, x2, x3. C and D differ only in x3.
    weights = np.array(
        [
            [-1, -3, -9, -9, -5],
            [-9, -3, -5, -5, -5],
            [0, 0, 5, -5, 0],
        ],
        dtype=float,
    )

    plane = feature_plane(weights, 0, 1)

    assert plane.groups == ((0,), (1,), (2, 3), (4,))
    assert plane.regions == (0, 1, 2)  # E is never the loudest here
    assert [line[:5] for line in describe(plane, "x1", "x2")] == [
        ("x1 = 3·x2", (0,), "x1 > 3·x2", (1,), "x1 < 3·x2"),
        ("3·x1 + x2 = 0", (1,), "3·x1 + x2 > 0", (2, 3), "3·x1 + x2 < 0"),
        ("2·x1 = x2", (0,), "2·x1 > x2", (2, 3), "2·x1 < x2"),
    ]
    winners = plane.winners(np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.1], [0.0, 0.0]]))
    np.testing.assert_array_equal(winners, [0, 1, 2, -1])


def test_a_class_between_two_others_does_not_get_a_line():
    # Voices x1, 0 and -x1: the middle class ties with both along x1 = 0 but never wins.
    weights = np.array([[1, 0, -1], [0, 0, 0]], dtype=float)

    plane = feature_plane(weights, 0, 1)

    assert [line[:5] for line in describe(plane, "x1", "x2")] == [
        ("x1 = 0", (0,), "x1 > 0", (2,), "x1 < 0"),
    ]
    assert plane.regions == (0, 2)


def test_lines_are_written_in_whole_numbers_whatever_the_step():
    lines = {
        step: [line_equation(b, "x3", "x4") for b in feature_plane(LECTURE_FINAL * step, 2, 3).borders]
        for step in (0.05, 0.25, 5.0)
    }

    assert lines[0.05] == lines[0.25] == lines[5.0] == ["x3 = x4", "x4 = 0", "x3 = 0"]


def test_world_and_mind_distances_match_the_lecture():
    np.testing.assert_array_equal(
        world_distances(knowledge(FLYING_OBJECTS)), [[0, 4, 2], [4, 0, 2], [2, 2, 0]]
    )
    np.testing.assert_array_equal(mind_distances(LECTURE_FINAL), [[0, 8, 6], [8, 0, 6], [6, 6, 0]])


def test_layout_preserves_distances_that_fit_on_a_plane():
    distances = np.array([[0.0, 3, 4], [3, 0, 5], [4, 5, 0]])

    points = layout_2d(distances)

    rebuilt = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.testing.assert_allclose(rebuilt, distances, atol=1e-9)


def test_features_shared_by_every_class_have_no_influence():
    influence = feature_influence(LECTURE_AFTER_THREE_LESSONS)

    # Wings and tail are present in all three classes.
    np.testing.assert_array_equal(influence, [0, 0, 2, 2, 2, 2])


# Final weight matrix printed by the lab draft (4 epochs, every example reinforced).
DRAFT_WEIGHTS = np.array(
    [
        [-8, -8, -8, -16, -16, -16, -8, -16],
        [4, -4, -4, -4, -4, -4, -4, -4],
        [-8, 0, 0, -8, -8, -8, -8, -8],
        [-4, -4, -4, 4, -4, -4, -4, -4],
        [-8, -8, -8, -8, -8, -8, 0, 0],
        [-12, -12, -12, -12, -4, -4, -4, -12],
        [-8, -8, 0, -8, -8, -8, -8, 0],
        [-8, -8, -16, -16, -16, -8, -8, -16],
    ],
    dtype=float,
)
DRAFT_SETTINGS = TrainingSettings(rule=Rule.EVERY_EXAMPLE, max_epochs=4, stop_when_error_free=False)


def test_draft_rule_reproduces_the_draft_weight_matrix():
    result = train(knowledge(INSTRUMENTS), DRAFT_SETTINGS)

    assert len(result.epochs) == 4
    np.testing.assert_array_equal(result.weights, DRAFT_WEIGHTS)
    assert equations(result.weights)[0] == "y1 = -8·x1 + 4·x2 - 8·x3 - 4·x4 - 8·x5 - 12·x6 - 8·x7 - 8·x8"


def test_draft_rule_cannot_separate_drum_from_xylophone_and_piano():
    result = train(knowledge(INSTRUMENTS), DRAFT_SETTINGS)

    drum = result.epochs[-1].exam[4]
    assert drum.winner is None
    assert drum.leaders == (4, 5, 6)
    assert not result.converged


def test_learning_from_mistakes_recognises_every_instrument():
    kb = knowledge(INSTRUMENTS)

    result = train(kb)

    assert result.converged
    assert [v.winner for v in result.epochs[-1].exam] == list(kb.labels)
    assert all(v.margin > 0 for v in result.epochs[-1].exam)


def test_training_stops_at_the_epoch_limit_when_classes_are_identical():
    kb = KnowledgeBase.from_rows(["a", "b"], [("One", [1, 0]), ("Two", [1, 0]), ("Three", [0, 1])])

    result = train(kb, TrainingSettings(max_epochs=7))

    assert len(result.epochs) == 7
    assert not result.converged
    assert any("identical" in problem for problem in kb.problems())


def test_identical_classes_are_reported_once():
    kb = KnowledgeBase.from_rows(["a"], [("One", [1]), ("Two", [1]), ("Two", [1]), ("Two", [1])])

    assert kb.problems() == ["One and Two have identical features, so they cannot be told apart."]


def test_problems_flags_examples_without_features_and_too_few_classes():
    kb = KnowledgeBase.from_rows(["a", "b"], [("Silent", [0, 0])])

    problems = kb.problems()

    assert any("two different classes" in p for p in problems)
    assert any("Silent has no features" in p for p in problems)
    assert knowledge(INSTRUMENTS).problems() == []


def test_step_size_only_matters_once_weights_are_normalised():
    kb = knowledge(INSTRUMENTS)

    plain = {step: train(kb, TrainingSettings(step=step)).error_curve for step in (0.1, 1.0, 5.0)}
    assert plain[0.1] == plain[1.0] == plain[5.0]

    clipped = {
        step: train(kb, TrainingSettings(step=step, normalize=True, max_epochs=40)).error_curve
        for step in (0.1, 1.0)
    }
    assert clipped[0.1] != clipped[1.0]


def test_uncertain_feature_values_still_give_an_answer():
    kb = knowledge(FLYING_OBJECTS)
    weights = train(kb).weights
    # Beak certain, plumage unknown (0.5): still a bird, but less confidently than a full bird.
    unsure = examine(weights, [1, 1, 1, 0, 0.5, 0])
    sure = examine(weights, [1, 1, 1, 0, 1, 0])

    assert unsure.winner == sure.winner == 0
    assert unsure.margin < sure.margin


def test_several_examples_of_one_class_share_a_demon():
    kb = KnowledgeBase.from_rows(
        ["String", "Plucked", "Wind"],
        [("Guitar", [1, 1, 0]), ("Flute", [0, 0, 1]), ("Guitar", [1, 0, 0])],
    )

    result = train(kb)

    assert kb.classes == ("Guitar", "Flute")
    assert kb.labels == (0, 1, 0)
    assert result.weights.shape == (3, 2)
    assert result.converged

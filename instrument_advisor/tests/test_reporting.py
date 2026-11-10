"""Wording of the training reports."""

from expert_system import KnowledgeBase, Rule, TrainingSettings, train
from presets import FLYING_OBJECTS, INSTRUMENTS
from reporting import (
    WITH_NORMALISATION,
    WITHOUT_NORMALISATION,
    step_conclusions,
    step_outcomes,
    training_summary,
)


def knowledge(preset):
    return KnowledgeBase.from_rows(preset.features, preset.rows)


def test_successful_training_says_how_many_epochs_were_run_and_why_it_stopped():
    result = train(knowledge(INSTRUMENTS))

    success, headline, detail = training_summary(result, "instrument")

    assert success
    assert "all 4 epochs were run" in headline
    assert "all 8 instrument examples without a single error" in headline
    assert "stopped after epoch 4" in detail and "limit 50" in detail


def test_fixed_programme_is_reported_as_carried_out_in_full():
    settings = TrainingSettings(max_epochs=6, stop_when_error_free=False)

    success, headline, detail = training_summary(train(knowledge(INSTRUMENTS), settings), "instrument")

    assert success
    assert "all 6 epochs were run" in headline
    assert "full programme of 6 epochs" in detail


def test_failed_training_names_what_is_confused_and_suggests_the_other_rule():
    settings = TrainingSettings(rule=Rule.EVERY_EXAMPLE, max_epochs=4)

    success, headline, detail = training_summary(train(knowledge(INSTRUMENTS), settings), "instrument")

    assert not success
    assert "all 4 epochs were run, but 2 of 8 instrument examples are still not recognised" in headline
    assert "Drum: the demons for Drum, Xylophone and Piano are equally loud" in detail
    assert "Xylophone: the demons for Xylophone and Piano are equally loud" in detail
    assert "Learn from mistakes" in detail


def test_single_failed_epoch_is_not_called_a_pass():
    success, headline, _ = training_summary(
        train(knowledge(FLYING_OBJECTS), TrainingSettings(max_epochs=1)), "object"
    )

    assert not success
    assert "the single epoch was run, but 1 of 3 object examples is still not recognised" in headline


def test_step_study_reports_that_the_step_is_irrelevant_without_normalisation():
    steps = (0.1, 1.0, 5.0)
    outcomes = step_outcomes(knowledge(INSTRUMENTS), TrainingSettings(), steps)

    free, clipped = step_conclusions(steps, outcomes, 50)

    assert outcomes[WITHOUT_NORMALISATION] == [4, 4, 4]
    assert "the step made no difference; every run needed 4 epochs" in free
    assert "multiplies all weights by the same number" in free
    assert outcomes[WITH_NORMALISATION][0] is not None
    assert outcomes[WITH_NORMALISATION][1:] == [None, None]
    assert "fastest with step 0.1" in clipped
    assert "with step 1 and 5 the system was still making errors after 50 epochs" in clipped


def test_scaling_explanation_is_not_given_for_normalised_runs():
    outcomes = {WITHOUT_NORMALISATION: [3, 3], WITH_NORMALISATION: [2, 2]}

    free, clipped = step_conclusions((0.5, 1.0), outcomes, 50)

    assert "multiplies all weights" in free
    assert "multiplies all weights" not in clipped
    assert "every run needed 2 epochs" in clipped

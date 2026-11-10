"""The interface as a whole, driven headlessly through Streamlit's test harness."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from expert_system import Rule

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def answer(at: AppTest, feature: str, value: str) -> None:
    at.button_group(key=f"answer::{feature}").set_value(value)


def page_text(at: AppTest) -> str:
    blocks = list(at.markdown) + list(at.caption) + list(at.success) + list(at.warning) + list(at.info)
    return "\n".join(str(block.value) for block in blocks)


def test_opens_trained_and_reports_that_all_epochs_were_run():
    at = run_app()

    assert len(at.success) == 1
    assert "Training complete — all 4 epochs were run" in at.success[0].value
    assert "all 8 instrument examples without a single error" in at.success[0].value
    assert not at.warning


def test_describing_a_guitar_names_the_guitar():
    at = run_app()

    answer(at, "String", "Yes")
    answer(at, "Plucked", "Yes")
    answer(at, "Wooden", "Yes")
    at.run()

    assert not at.exception
    assert "## 🎵 Guitar" in page_text(at)
    assert "exactly the features the system was taught for Guitar" in page_text(at)


def test_unseen_combination_is_presented_as_a_forecast():
    at = run_app()

    # Nothing else is bowed, so a bowed string instrument should still be a violin without "wooden".
    answer(at, "String", "Yes")
    answer(at, "Bowed", "Yes")
    at.run()

    assert "## 🎵 Violin" in page_text(at)
    assert "never shown this combination" in page_text(at)


def test_teaching_a_new_instrument_retrains_and_recognises_it():
    at = run_app()
    answer(at, "Wind", "Yes")
    answer(at, "Wooden", "Yes")
    at.run()

    at.text_input(key="teach_name").set_value("Recorder")
    at.button(key="teach_submit").click()
    at.run()

    assert not at.exception
    assert "“Recorder” was added to the knowledge base" in page_text(at)
    assert "all 9 instrument examples without a single error" in at.success[0].value
    assert "## 🎵 Recorder" in page_text(at)


def test_an_undecided_answer_can_be_taught():
    at = run_app()
    at.sidebar.radio[0].set_value(Rule.EVERY_EXAMPLE)
    answer(at, "Percussion", "Yes")
    at.run()
    assert "Undecided" in page_text(at)

    at.text_input(key="teach_name").set_value("Bongo")
    at.button(key="teach_submit").click()
    at.run()

    assert not at.exception
    assert "“Bongo” was added to the knowledge base" in page_text(at)


def test_draft_rule_reports_the_instruments_it_cannot_separate():
    at = run_app()

    at.sidebar.radio[0].set_value(Rule.EVERY_EXAMPLE)
    at.run()

    assert not at.exception
    assert not at.success
    assert "2 of 8 instrument examples are still not recognised" in at.warning[0].value
    assert "Drum: the demons for Drum, Xylophone and Piano are equally loud" in at.warning[0].value


def test_lecture_example_loads_and_trains():
    at = run_app()

    at.sidebar.selectbox[0].set_value("Lecture example: bird, plane, glider")
    at.run()

    assert not at.exception
    assert "all 3 object examples without a single error" in at.success[0].value


def test_model_tab_writes_the_lecture_dividing_lines():
    at = run_app()
    at.sidebar.selectbox[0].set_value("Lecture example: bird, plane, glider")
    at.sidebar.toggle[0].set_value(True)  # normalise, as the lecture does
    at.run()
    {box.label: box for box in at.selectbox}["Horizontal axis"].set_value(2)
    at.run()
    {box.label: box for box in at.selectbox}["Vertical axis"].set_value(3)
    at.run()

    assert not at.exception
    text = page_text(at)
    # Lecture: bird (x3 > x4) & (x3 > 0), plane (x4 > x3) & (x4 > 0), glider (x3 < 0) & (x4 < 0).
    assert "1. `x3 = x4` — Bird where `x3 > x4`, Plane where `x3 < x4`" in text
    assert "2. `x4 = 0` — Plane where `x4 > 0`, Glider where `x4 < 0`" in text
    assert "3. `x3 = 0` — Bird where `x3 > 0`, Glider where `x3 < 0`" in text
    assert "y1 = y2: `x3 - x4 = -x3 + x4`, which simplifies to `x3 = x4`" in text

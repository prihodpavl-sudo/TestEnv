"""Instrument Advisor: a self-learning expert system with a Streamlit interface.

Run with:  python app.py   (or: streamlit run app.py)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

import charts
from expert_system import (
    FeaturePlane,
    KnowledgeBase,
    Rule,
    TrainingResult,
    TrainingSettings,
    equations,
    examine,
    feature_influence,
    feature_plane,
    line_equation,
    train,
)
from presets import INSTRUMENTS, PRESETS, Preset
from reporting import (
    RULE_LABELS,
    answer_text,
    listing,
    plural,
    step_conclusions,
    step_outcomes,
    training_summary,
)

NAME = "__name__"  # table column that holds the class name
ANSWERS = {"No": 0.0, "Not sure": 0.5, "Yes": 1.0}
STEPS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
RULE_CAPTIONS = [
    "Repeat only what the system gets wrong, as in the lecture.",
    "Reward and punish on every example in every epoch, as in the lab draft.",
]


def show(figure, key: str) -> None:
    st.plotly_chart(figure, key=key, config={"displayModeBar": False})


# ---------- knowledge base table ----------


def table_from_preset(preset: Preset) -> pd.DataFrame:
    data: dict[str, list] = {NAME: [name for name, _ in preset.rows]}
    for i, feature in enumerate(preset.features):
        data[feature] = [bool(vector[i]) for _, vector in preset.rows]
    return pd.DataFrame(data)


def feature_names(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column != NAME]


def clean(table: pd.DataFrame) -> pd.DataFrame:
    """Drop unnamed rows and turn half-filled cells from the editor into plain values.

    Names that differ only in letter case are treated as the same class.
    """
    table = table.copy()
    spelling: dict[str, str] = {}
    names = [str(v).strip() if pd.notna(v) else "" for v in table[NAME]]
    table[NAME] = [spelling.setdefault(name.lower(), name) for name in names]
    table = table[table[NAME] != ""]
    for feature in feature_names(table):
        table[feature] = [bool(v) if pd.notna(v) else False for v in table[feature]]
    return table.reset_index(drop=True)


def knowledge_from(table: pd.DataFrame) -> KnowledgeBase:
    features = feature_names(table)
    rows = [(row[NAME], [float(row[f]) for f in features]) for _, row in table.iterrows()]
    return KnowledgeBase.from_rows(features, rows)


def load_preset(title: str) -> None:
    st.session_state.table = table_from_preset(PRESETS[title])
    st.session_state.table_version += 1
    clear_answers()


def replace_table(table: pd.DataFrame, notice: str) -> None:
    """Store a programmatic change; the editor is rebuilt and the next run retrains."""
    st.session_state.table = table.reset_index(drop=True)
    st.session_state.table_version += 1
    st.session_state.notice = notice


def clear_answers() -> None:
    for key in st.session_state:
        if str(key).startswith("answer::"):
            st.session_state[key] = "No"


def init_state() -> None:
    if "table" not in st.session_state:
        st.session_state.table_version = 0
        st.session_state.table = table_from_preset(INSTRUMENTS)


# ---------- sidebar ----------


def sidebar() -> tuple[TrainingSettings, Preset]:
    with st.sidebar:
        st.header("Settings")
        title = st.selectbox(
            "Knowledge base",
            list(PRESETS),
            key="preset",
            on_change=lambda: load_preset(st.session_state.preset),
            help="Switching replaces the table with the chosen example data.",
        )
        st.subheader("Training")
        rule = st.radio("Teaching rule", list(Rule), format_func=RULE_LABELS.get, captions=RULE_CAPTIONS)
        step = st.select_slider(
            "Step h",
            options=STEPS,
            value=1.0,
            help="The example is multiplied by h before it is added to or subtracted from the matrix.",
        )
        normalize = st.toggle(
            "Normalise the matrix",
            value=False,
            help="After every lesson, pull weights back to the nearest border of the −1…1 hypercube.",
        )
        max_epochs = st.slider("Epoch limit", 1, 200, 50)
        stop = st.toggle("Stop once the examination is error-free", value=True)
        st.caption("Any change here or in the knowledge base retrains the system from scratch.")
    settings = TrainingSettings(
        rule=rule or Rule.MISTAKES,
        step=float(step),
        max_epochs=int(max_epochs),
        normalize=normalize,
        stop_when_error_free=stop,
    )
    return settings, PRESETS[title]


# ---------- tabs ----------


def knowledge_editor(preset: Preset) -> pd.DataFrame:
    source: pd.DataFrame = st.session_state.table
    features = feature_names(source)
    noun = preset.noun
    st.subheader("What the system is taught")
    st.caption(
        f"Tick the features of each {noun}. Type in the empty bottom row to add a new one, "
        "or select rows and press Delete to remove them. The system retrains after every change."
    )
    config = {NAME: st.column_config.TextColumn(noun.capitalize(), required=True, width="medium")}
    for feature in features:
        config[feature] = st.column_config.CheckboxColumn(feature, default=False)
    edited = st.data_editor(
        source,
        key=f"editor_{st.session_state.table_version}",
        num_rows="dynamic",
        hide_index=True,
        column_config=config,
    )
    table = clean(edited)

    with st.expander("Add or remove a feature"):
        add, remove = st.columns(2)
        with add, st.form("add_feature", clear_on_submit=True):
            new = st.text_input("New feature", placeholder="e.g. Brass").strip()
            if st.form_submit_button("Add feature"):
                if not new:
                    st.error("Give the feature a name.")
                elif new.lower() in {f.lower() for f in features} or new == NAME:
                    st.error(f"“{new}” is already a feature.")
                else:
                    table[new] = False
                    replace_table(table, f"Feature “{new}” added. Tick it for the {noun}s that have it.")
                    st.rerun()
        with remove, st.form("remove_feature"):
            gone = st.selectbox("Feature to remove", features)
            if st.form_submit_button("Remove feature"):
                if len(features) <= 1:
                    st.error("At least one feature has to stay.")
                else:
                    replace_table(table.drop(columns=[gone]), f"Feature “{gone}” removed.")
                    st.session_state.pop(f"answer::{gone}", None)
                    st.rerun()
    st.button("Restore the original data", on_click=load_preset, args=(preset.title,))
    return table


def advisor(result: TrainingResult, table: pd.DataFrame, noun: str) -> None:
    knowledge = result.knowledge
    classes = knowledge.classes
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader(f"Describe the {noun}")
        st.caption("Say which features it has. “Not sure” counts as half a yes.")
        answers = []
        grid = st.columns(2)
        for i, feature in enumerate(knowledge.features):
            with grid[i % 2]:
                choice = st.segmented_control(feature, list(ANSWERS), default="No", key=f"answer::{feature}")
                answers.append(ANSWERS[choice or "No"])
        st.button("Clear answers", on_click=clear_answers)
    vector = np.array(answers)

    with right:
        st.subheader("The system's answer")
        if not vector.any():
            st.info(f"Choose at least one feature on the left and the best matching {noun} appears here.")
            return
        verdict = examine(result.weights, vector)
        # Classes the system was taught with exactly these features.
        known = [
            classes[label]
            for example, label in zip(knowledge.examples, knowledge.labels)
            if np.array_equal(example, vector)
        ]
        with st.container(border=True):
            if verdict.winner is None:
                tied = listing([classes[i] for i in verdict.leaders])
                st.markdown("#### 🤔 Undecided")
                st.warning(
                    f"The demons for {tied} are equally loud, so the system cannot choose. "
                    "Change or add a feature, or teach it the right answer below."
                )
            else:
                best = classes[verdict.winner]
                st.caption("Best match")
                st.markdown(f"## 🎵 {best}")
                runner_up = classes[verdict.ranking[1]]
                st.markdown(f"Its demon is the loudest, **{verdict.margin:.3g}** ahead of {runner_up}.")
                if best in known:
                    st.caption(f"These are exactly the features the system was taught for {best}.")
                elif not known:
                    st.caption(
                        "The system was never shown this combination: this is its forecast "
                        "from what it has learned."
                    )
                exam = result.epochs[-1].exam
                if result.converged and verdict.margin < min(v.margin for v in exam):
                    st.warning(
                        "Close call: this lead is smaller than for anything the system was taught, "
                        "so the object lies near the border between classes and may be confused."
                    )
            show(charts.demon_voices(verdict, classes), "voices")

        with st.form("teach", clear_on_submit=True):
            st.markdown("**Teach the system**")
            st.caption(
                f"Wrong answer, or a new {noun}? Name it: it is added to the knowledge base "
                "and the system retrains itself."
            )
            name = st.text_input(
                f"What is this {noun}?", placeholder="e.g. Ukulele", key="teach_name"
            ).strip()
            if st.form_submit_button("Add and retrain", type="primary", key="teach_submit"):
                existing = {c.lower(): c for c in classes}
                name = existing.get(name.lower(), name)
                if not name:
                    st.error("Type a name first.")
                elif 0.5 in vector:
                    st.error(
                        "Change every “Not sure” to Yes or No first: "
                        "the knowledge base stores definite features."
                    )
                elif name in known:
                    st.info(f"The system already knows {name} with exactly these features.")
                else:
                    row = {NAME: name, **{f: bool(v) for f, v in zip(knowledge.features, vector)}}
                    replace_table(
                        pd.concat([table, pd.DataFrame([row])], ignore_index=True),
                        f"“{name}” was added to the knowledge base.",
                    )
                    st.rerun()


def training_report(result: TrainingResult, noun: str) -> None:
    knowledge = result.knowledge
    classes = knowledge.classes
    last = result.epochs[-1]
    total = len(knowledge.labels)
    run = len(result.epochs)
    updates = [sum(lesson.taught for lesson in epoch.lessons) for epoch in result.epochs]
    leads = [v.margin for v, label in zip(last.exam, knowledge.labels) if v.winner == label]

    metrics = st.columns(4)
    metrics[0].metric("Epochs run", run, border=True)
    metrics[1].metric("Recognised in the exam", f"{total - last.exam_errors} of {total}", border=True)
    metrics[2].metric("Matrix updates", sum(updates), border=True)
    metrics[3].metric("Smallest winning lead", f"{min(leads):.3g}" if leads else "—", border=True)
    st.progress(1.0, text=f"Finished: {plural(run, 'epoch')} run (limit {result.settings.max_epochs})")

    st.subheader("Learning curve")
    show(charts.learning_curve(result), "curve")
    st.caption(
        "After every epoch the system is switched to examination mode and tested on everything it was taught."
    )

    st.subheader("How each epoch went")
    st.dataframe(
        pd.DataFrame(
            {
                "Epoch": [e.number for e in result.epochs],
                "Mistakes during the lessons": [e.mistakes for e in result.epochs],
                "Matrix updates": updates,
                "Errors in the examination": [e.exam_errors for e in result.epochs],
                "Outcome": [
                    "✅ Error-free" if e.exam_errors == 0 else "⏳ Keep learning" for e in result.epochs
                ],
            }
        ),
        hide_index=True,
    )

    st.subheader("Final examination")
    st.dataframe(
        pd.DataFrame(
            {
                noun.capitalize(): [classes[label] for label in knowledge.labels],
                "Recognised as": [answer_text(v, classes) for v in last.exam],
                "Winning lead": [f"{v.margin:.3g}" if v.winner is not None else "—" for v in last.exam],
                "Result": [
                    "✅ Correct" if v.winner == label else "❌ Error"
                    for v, label in zip(last.exam, knowledge.labels)
                ],
            }
        ),
        hide_index=True,
    )

    st.subheader("Lesson by lesson")
    number = st.slider("Epoch to look at", 1, run, run) if run > 1 else 1
    epoch = result.epochs[number - 1]
    st.dataframe(
        pd.DataFrame(
            {
                "Shown": [classes[lesson.label] for lesson in epoch.lessons],
                "The system answered": [answer_text(lesson.verdict, classes) for lesson in epoch.lessons],
                "Result": ["✅ Correct" if lesson.recognised else "❌ Mistake" for lesson in epoch.lessons],
                "Teacher's reaction": [
                    f"Rewarded {classes[lesson.label]}, punished the others" if lesson.taught else "No change"
                    for lesson in epoch.lessons
                ],
            }
        ),
        hide_index=True,
    )


def dividing_lines(plane: FeaturePlane, weights: np.ndarray, knowledge: KnowledgeBase) -> None:
    """Equations of the lines that divide the regions on the plane, and where they come from."""
    classes, features = knowledge.classes, knowledge.features
    x, y = f"x{plane.fx + 1}", f"x{plane.fy + 1}"

    def members(group: int) -> str:
        return " / ".join(classes[c] for c in plane.groups[group])

    st.markdown(
        f"**Equations of the dividing lines** ({x} = {features[plane.fx]}, {y} = {features[plane.fy]})"
    )
    if plane.borders:
        st.markdown(
            "\n".join(
                f"{number}. `{line_equation(border, x, y)}` — {members(border.upper)} where "
                f"`{line_equation(border, x, y, '>')}`, {members(border.lower)} where "
                f"`{line_equation(border, x, y, '<')}`"
                for number, border in enumerate(plane.borders, start=1)
            )
        )
    else:
        st.info(
            f"There are no dividing lines: every class has the same weights for {x} and {y}, "
            "so these two features alone cannot tell them apart."
        )
    for group in plane.groups:
        if 1 < len(group) < len(classes):
            st.caption(
                f"{listing([classes[c] for c in group])} have the same weights for {x} and {y}, "
                "so they share a region here; other features tell them apart."
            )
    silent = [classes[c] for g, group in enumerate(plane.groups) if g not in plane.regions for c in group]
    if silent:
        one = len(silent) == 1
        st.caption(
            f"{listing(silent)} {'is' if one else 'are'} never the loudest on this plane, "
            f"so {'it has' if one else 'they have'} no region here."
        )
    with st.expander("Where these equations come from"):
        voices = equations(weights[[plane.fx, plane.fy], :], (x, y))
        st.markdown(f"With every other feature at 0, each class's equation keeps only its {x} and {y} terms:")
        width = max(len(line) for line in voices)
        st.code(
            "\n".join(f"{line:<{width}}   ({name})" for line, name in zip(voices, classes)), language=None
        )
        if plane.borders:
            first = plane.borders[0]
            j, k = plane.groups[first.upper][0], plane.groups[first.lower][0]
            left, right = voices[j].split(" = ", 1)[1], voices[k].split(" = ", 1)[1]
            st.markdown(
                "A dividing line is where two voices are equal. Line 1, for example, is where "
                f"y{j + 1} = y{k + 1}: `{left} = {right}`, which simplifies to "
                f"`{line_equation(first, x, y)}`. There is no constant term, so every line passes "
                "through the origin (0, 0)."
            )


def model_view(result: TrainingResult) -> None:
    knowledge = result.knowledge
    features, classes = knowledge.features, knowledge.classes
    run = len(result.epochs)
    number = st.slider("Show the model as it was after epoch", 1, run, run) if run > 1 else 1
    weights = result.epochs[number - 1].weights

    def axis_label(i: int) -> str:
        return f"x{i + 1} · {features[i]}"

    st.subheader("Weight matrix")
    show(charts.weight_matrix(weights, features, classes), "matrix")
    st.caption("Blue cells speak for a class, red cells against it.")

    st.subheader("System of equations")
    st.code("\n".join(equations(weights)), language=None, wrap_lines=True)
    st.caption("Features: " + ", ".join(f"x{i + 1} = {name}" for i, name in enumerate(features)))
    st.caption("Classes: " + ", ".join(f"y{j + 1} = {name}" for j, name in enumerate(classes)))

    st.subheader("How the classes are separated")
    if len(features) < 2:
        st.info("Two features are needed to draw this picture.")
    else:
        strongest = [int(i) for i in np.argsort(-feature_influence(weights), kind="stable")[:2]]
        pick_x, pick_y = st.columns(2)
        fx = pick_x.selectbox(
            "Horizontal axis", range(len(features)), index=strongest[0], format_func=axis_label
        )
        others = [i for i in range(len(features)) if i != fx]
        default_y = strongest[1] if strongest[1] != fx else others[0]
        fy = pick_y.selectbox("Vertical axis", others, index=others.index(default_y), format_func=axis_label)
        plane = feature_plane(weights, fx, fy)
        show(charts.class_regions(plane, knowledge), "regions")
        st.caption(
            "A slice through feature space with every other feature set to No. Each colour is the region "
            "where one class has the loudest demon; grey marks classes that are equally loud here. "
            "The numbered black lines divide the regions, and their equations are listed below. "
            "Dots show where the taught classes sit on these two features."
        )
        dividing_lines(plane, weights, knowledge)

    st.subheader("Which features matter")
    show(charts.influence(weights, features), "influence")
    st.caption("A feature whose weights are the same for every class cannot change the answer.")

    st.subheader("The world and the system's picture of it")
    show(charts.class_maps(knowledge, weights), "maps")
    st.caption(
        "Left: classes placed by how many features they differ in. Right: the same classes placed by how "
        "far apart their equations are. Both are flattened to two dimensions, so distances are approximate."
    )


def step_study(knowledge: KnowledgeBase, settings: TrainingSettings) -> None:
    st.subheader("Training from scratch with different steps")
    st.caption(
        f"Each bar is a separate training run with the “{RULE_LABELS[settings.rule]}” rule "
        f"and a limit of {plural(settings.max_epochs, 'epoch')}."
    )
    outcomes = step_outcomes(knowledge, settings, STEPS)
    show(charts.step_study(STEPS, outcomes, settings.max_epochs), "steps")
    st.markdown("**Conclusion**")
    for line in step_conclusions(STEPS, outcomes, settings.max_epochs):
        st.markdown("- " + line)


# ---------- page ----------


def main() -> None:
    st.set_page_config(page_title="Instrument Advisor", page_icon="🎵", layout="wide")
    init_state()
    settings, preset = sidebar()
    noun = preset.noun

    st.title("🎵 Instrument Advisor")
    st.caption(
        "A self-learning expert system: it is shown examples, rewarded for right answers and punished "
        f"for wrong ones, and then names the {noun} that best fits the features you describe."
    )
    banner = st.container()
    tab_advisor, tab_knowledge, tab_training, tab_model, tab_steps = st.tabs(
        ["🎯 Advisor", "📚 Knowledge base", "🎓 Training report", "📐 Model and graphs", "🔬 Step study"]
    )

    # The editor runs first because every other tab depends on its content.
    with tab_knowledge:
        table = knowledge_editor(preset)
    knowledge = knowledge_from(table)
    notice = st.session_state.pop("notice", None)

    if len(knowledge.classes) < 2 or not knowledge.features:
        with banner:
            st.error(
                f"There is nothing to learn yet: add at least two different {noun}s "
                "in the Knowledge base tab."
            )
        return

    result = train(knowledge, settings)
    success, headline, detail = training_summary(result, noun)
    signature = (
        knowledge.features,
        knowledge.classes,
        knowledge.labels,
        knowledge.examples.tobytes(),
        settings,
    )
    retrained = st.session_state.get("signature") not in (None, signature)
    st.session_state.signature = signature
    if retrained:
        st.toast(f"Retrained automatically: {plural(len(result.epochs), 'epoch')} run.", icon="🔄")

    with banner:
        if notice:
            st.info(f"{notice} The system retrained itself with the new data.", icon="🔄")
        (st.success if success else st.warning)(f"**{headline}**  \n{detail}", icon="✅" if success else "⚠️")
        for problem in knowledge.problems():
            st.warning(problem, icon="🧩")

    with tab_advisor:
        advisor(result, table, noun)
    with tab_training:
        training_report(result, noun)
    with tab_model:
        model_view(result)
    with tab_steps:
        step_study(knowledge, settings)


if __name__ == "__main__":
    if get_script_run_ctx(suppress_warning=True) is None:
        # Started as "python app.py": hand over to Streamlit, which then runs this file itself.
        from streamlit.web import cli

        sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
        sys.exit(cli.main())
    main()

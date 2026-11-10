# TestEnv

## River aquarium

An animated canvas aquarium of Central European river fauna: northern pike, wels catfish, common carp, brown trout, European perch, a school of roach and a noble crayfish, among reeds, waterweed, yellow water-lily pads and alder roots.

Open `index.html` in a browser, or serve the folder with `python3 -m http.server`. No build step or dependencies.

- Hover a fish to see its common and Latin name.
- Click the water to drop food.
- Switch lighting between Day, Dusk and Night (the catfish gets livelier after dark).

## Instrument Advisor

A self-learning expert system on the imitation principle, in `instrument_advisor/`. It is taught musical instruments by their features, then names the instrument that best fits the features you describe.

```sh
cd instrument_advisor
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tested with Python 3.12.

How it works:

- The model is a weight matrix with one row per feature and one column per instrument, starting at zero.
- **Training:** each example is added to the column of the right instrument and subtracted from the others ("carrot and stick"). After every epoch the system is examined on everything it was taught, and training stops once it makes no errors.
- **Examination:** the feature vector is multiplied by the matrix. Each result is the voice of a "demon"; the loudest names the instrument. Equal voices mean the object is not recognised.

What the interface does:

- **Advisor:** answer Yes / No / Not sure for each feature and see the best match, its lead and every demon's voice. If the answer is wrong or the instrument is new, name it and the system adds it and retrains.
- **Knowledge base:** edit the table of instruments and features, add rows, add or remove features. Every change retrains the system automatically.
- **Training report:** whether training finished, how many epochs were run, the learning curve, each epoch's results, the final examination and each lesson.
- **Model and graphs:** the weight matrix after any epoch, the system of equations, class regions on two chosen features, which features matter, and maps of the classes as the data describes them and as the system learned them.
- **Step study:** training from scratch with different step sizes, with and without normalisation.

The sidebar switches between the instrument data and the lecture's bird / plane / glider example, and sets the teaching rule, step, normalisation and epoch limit.

Files: `expert_system.py` (the model, no interface code), `reporting.py` (wording of the reports), `charts.py` (Plotly figures), `app.py` (Streamlit interface), `presets.py` (built-in data).

Run the tests with `pip install -r requirements-dev.txt && pytest` from `instrument_advisor/`.

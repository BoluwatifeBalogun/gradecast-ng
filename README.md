# GradeCast NG

Implementation of **"Development of a Data Driven Predictive System for Student
Academic Performance Using an Ensemble Classifier"** (Alli Sofiat Abidemi and
Adeboye Samuel, HND Computer Science, Yaba College of Technology).

A Flask web application that predicts a student's likely class of result
(First Class, Upper Second, Lower Second, Pass or Fail) from 19 academic,
behavioural and socio-economic attributes. A Random Forest and a Gradient
Boosting classifier are combined by a Voting Classifier, trained on the Kaggle
"Student Performance Factors" dataset balanced with SMOTE.

## Run it

Needs Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                      # http://localhost:5000
```

A trained model is included, so the app predicts straight away. If you install
different library versions and the saved model fails to load, retrain it:

```bash
python train.py            # 1 to 2 minutes, depending on CPU cores
python train.py --cv 5     # also runs 5-fold cross-validation, 4 to 10 minutes
```

### Demonstration accounts

| Role | Email | Password |
|---|---|---|
| System Administrator | admin@gradecast.ng | Admin@2026 |
| Academic Officer | officer@gradecast.ng | Officer@2026 |
| Student | student@gradecast.ng | Student@2026 |

Change these before putting the system online, and set `SHOW_DEMO_ACCOUNTS=0`
to take them off the sign-in page. On first run the officer
account is given 48 sample records (real dataset rows under invented names,
marked `SAMPLE`) so the dashboard is not empty. Remove them from
Admin > Datasets, or start with `SEED_SAMPLE=0`.

## Photographs on the front page

The landing page has four photo slots. Put your pictures in
`webapp/static/img/` with these names and they appear automatically:

| File | Where it shows |
|---|---|
| `hero.jpg` | Large picture in the photo section, and the sign-in side panel |
| `campus-1.jpg`, `campus-2.jpg`, `campus-3.jpg` | The three smaller pictures |

`.jpeg`, `.png` and `.webp` also work. Landscape, about 1600 pixels wide, under
500 KB each. The layout adapts to one, two, three or four pictures. Edit the
alt text, captions and credits in `webapp/photos.py`.

Out of the box only `hero` has a default: a free Unsplash photograph of
graduands in Lagos by Blessfield John, loaded from the internet. Run
`python tools/fetch_photos.py` once while online to save it locally so it
shows without a connection. A picture that cannot load is removed from the
page instead of showing as broken. Photographs from your own institution are
the better choice. Only use pictures you have permission to use.

## How it maps to the report

| Report | Where it lives |
|---|---|
| 3.3 Proposed system, 3.5.1 three-layer architecture | Presentation: `webapp/templates`, `webapp/static`. Processing: `ml/schema.py` (validation), `ml/pipeline.py` (encode, scale). Model: `ml/service.py`, `models/ensemble.joblib` |
| 3.4 Dataset and attributes | `data/StudentPerformanceFactors.csv`, `ml/schema.py` |
| 3.4.4 Preprocessing | `pipeline.clean`, `encode`, `balance`, `scale` |
| Ensemble (RF + GB + Voting) | `pipeline.build_ensemble` |
| 3.5.2 Use cases: Student / Academic Officer | `/predict`, `/predictions/<id>`, `/history`, `/dashboard` |
| 3.5.2 Use cases: System Administrator | `/admin/datasets`, `/admin/training`, `/admin/users` |
| 3.6 and 3.8 Evaluation | `/insights`, `models/metrics.json` |
| 1.5 Relational database | SQLite, `webapp/db.py` |
| 2.8 KDD framework | The five stages in `ml/pipeline.py`, shown on `/insights` |

## Training pipeline

1. **Selection.** Load the CSV (6,607 records, 19 attributes plus `Exam_Score`).
2. **Preprocessing.** Fix types, fill missing values (median or mode), drop
   duplicates, cap one score recorded as 101.
3. **Transformation.**
   - Ordinal-encode the 13 categorical attributes.
   - Band `Exam_Score` into five classes (see below).
   - Stratified 80/20 split.
   - SMOTE on the **training split only**, every class raised to 4,100
     records: 20,500 in total, 15,215 of them synthetic.
   - Standardise the six numeric attributes.
4. **Data mining.** Random Forest (250 trees) and Gradient Boosting (300
   stages) joined by soft voting, weights 1 to 2.
5. **Evaluation.** Accuracy, precision, recall, F1 and confusion matrix on the
   1,322 held-out original records.

The balanced training table is written to `data/processed/training_balanced.csv`
with a `Record_Type` column (Original or Synthetic) so the 20,000+ records can
be shown. It holds the encoded values the model trains on: categories appear
as 0, 1, 2 in the order listed in `ml/schema.py`, and synthetic rows carry
decimals because SMOTE interpolates between real students.

Used as an early-warning screen on the same 1,322 held-out students, the
"At risk" flag (predicted Fail or Pass) caught 243 of the 290 students who
really ended in Fail or Pass (84%), including all 63 who failed, and 91% of
flagged students were truly in those classes. The 47 it missed were all
predicted Lower Second.

### Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Single decision tree | 56.2% | 55.6% | 56.8% | 56.0% |
| Random Forest | 74.1% | 75.9% | 70.7% | 72.6% |
| Gradient Boosting | 80.3% | 81.9% | 77.7% | 79.5% |
| **Voting ensemble** | **79.8%** | **82.2%** | **77.3%** | **79.4%** |

Five-fold cross-validation accuracy: 79.5% (plus or minus 0.5%). 99.8% of test
predictions are the true class or the class next to it.

## Decisions worth knowing before the defence

**Class bands.** `Exam_Score` only spans 55 to 101 and clusters around 67. The
usual 40/45/50/60/70 marks would put nearly every student in one class and
leave Fail empty, so scores are banded relative to the cohort: Fail up to 61,
Pass 62 to 64, Lower Second 65 to 67, Upper Second 68 to 70, First Class 71 and
above. Change `DEFAULT_CUTOFFS` in `ml/schema.py` to move them.

**SMOTE only in training.** Figure 3.4 of the report draws "Load and preprocess
dataset" and "Apply SMOTE" inside the prediction flow. SMOTE balances training
data; it has no role when one student is being scored. Here a prediction runs
validate, encode, scale, vote. SMOTE is also applied after the train/test
split, otherwise synthetic copies of test students leak into training and
inflate accuracy. Missing values are filled, and the scaler fitted, from the
training split for the same reason. Synthetic records keep their decimal
values. The first version of this build used SMOTE-NC and rounded synthetic
values to whole numbers, and scored 77.1%, about 2.7 points lower.

**Ensemble versus its parts.** The ensemble beats a single decision tree by
about 24 points and Random Forest by about 6. Gradient Boosting alone is level
with it (0.5 point ahead on accuracy, the ensemble slightly ahead on
precision). Do not claim the ensemble beats every component. The honest claim
is that it clearly beats a single classifier and is as good as its best member.

**Units.** In the dataset `Hours_Studied` is hours per week (1 to 44) and
`Tutoring_Sessions` is per month. Section 3.4.1 of the report says per day and
per week.

**Nigerian context.** The training data is the public Kaggle dataset, not
Nigerian student records. State, geopolitical zone and institution type are
tags on each prediction for reporting. They are not model inputs. An
administrator can upload an institution's own CSV with the same columns and
retrain.

**Suggestions.** "What would help most" is a what-if sensitivity analysis of
the trained model. Each suggestion is one realistic step (attendance up 15
points, six more study hours a week, one level up on a Low/Medium/High scale).
The most helpful step is found, applied, and the rest are tested again on top
of it. It shows what the model responds to. It is not a causal claim.

**Outside the training range.** The dataset has no student below 60%
attendance or below 50 in previous score. The form accepts lower values, but a
tree model treats 45% attendance exactly like 60%. The form says what range the
model was trained on and the result page flags any value outside it.

## Tests

```bash
python -m unittest tests.test_system -v
```

Thirty tests on a temporary database: the model matches Chapter Three (RF plus
GB under a Voting Classifier, 19 inputs, five classes, 20,000+ balanced
records, no test student in the training file), reported metrics reproduce,
the web form gives the same class as the raw model, validation, roles,
privacy between users, CSRF, uploads and training.

## Hosting

`Procfile` and `render.yaml` are included.

```bash
gunicorn -w 1 --threads 4 --timeout 180 run:app
```

Keep it to **one worker** (threads are fine): training progress is tracked in
memory. Set `SECRET_KEY` in the environment, and `COOKIE_SECURE=1` when serving over HTTPS. The SQLite file lives in
`instance/`; on hosts with an ephemeral disk set `DATABASE_PATH` to a
persistent location or records reset on each deploy. Serving peaked at about 385 MB of memory when
measured, and retraining needs more. That is tight on a 512 MB instance, so
retrain locally and deploy the saved model there, or use a 1 GB instance.

## Project layout

```
run.py                entry point
train.py              command-line training
ml/                   schema, pipeline, prediction service, advice text
webapp/               Flask app: auth, views, admin, templates, static
data/                 dataset, uploads, balanced training export
models/               ensemble.joblib, metrics.json
tests/                functional and model tests
```

Front-end assets are bundled (Chart.js, Lucide icons, Bricolage Grotesque and
Instrument Sans), so the app works without an internet connection.

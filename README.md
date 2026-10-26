# GradeCast NG

Implementation of **"Development of a Data Driven Predictive System for Student
Academic Performance Using an Ensemble Classifier"** (Alli Sofiat Abidemi and
Adeboye Samuel, HND Computer Science, Yaba College of Technology).

A Flask web application that predicts a student's likely class of result
from 19 academic,
behavioural and socio-economic attributes. A Random Forest and a Gradient
Boosting classifier are combined by a Voting Classifier, trained on the Kaggle
"Student Performance Factors" dataset balanced with SMOTE.

Classes are named the way the student's institution names them:

| Level | University | Polytechnic (ND, HND) | College of Education (NCE) |
|---|---|---|---|
| 5 | First Class | Distinction | Distinction |
| 4 | Second Class Upper | Upper Credit | Credit |
| 3 | Second Class Lower | Lower Credit | Merit |
| 2 | Third Class | Pass | Pass |
| 1 | Fail | Fail | Fail |

The model predicts the level; the institution type chosen on the form decides
the name. Edit the names in `webapp/grading.py`.

Each prediction comes with numbered recommendations, and can be downloaded on
its own as a PDF report (from the result page, or the icon on each row of
History). The CSV export of all predictions includes the recommendations too.

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
"At risk" flag (predicted at the two lowest levels) caught 245 of the 290 students who
really ended at those levels (84%), including 63 of the 63 who failed, and
91% of flagged students were truly at those levels. The 45 it missed were
all predicted at the middle level (Second Class Lower or Lower Credit).

### Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Single decision tree | 56.0% | 56.1% | 56.1% | 56.0% |
| Random Forest | 73.9% | 76.2% | 70.6% | 72.8% |
| Gradient Boosting | 79.4% | 80.1% | 76.8% | 78.3% |
| **Voting ensemble** | **79.8%** | **81.4%** | **76.8%** | **78.7%** |

Five-fold cross-validation accuracy: 79.2% (plus or minus 1.1%). 99.8% of test
predictions are the true class or the class next to it.

## Decisions worth knowing before the defence

**Class bands.** `Exam_Score` only spans 55 to 101 and clusters around 67. The
usual 40/45/50/60/70 marks would put nearly every student in one class and
leave Fail empty, so scores are banded relative to the cohort into five
levels: up to 61, 62 to 64, 65 to 67, 68 to 70, and 71 and above. Change
`DEFAULT_CUTOFFS` in `ml/schema.py` to move them. The levels are named per
institution type (table above), so a level is the student's standing among
peers, not a CGPA. A university Pass degree, where it still exists, has no
level of its own: the model has five levels and a university needs six names.

**SMOTE only in training.** Figure 3.4 of the report draws "Load and preprocess
dataset" and "Apply SMOTE" inside the prediction flow. SMOTE balances training
data; it has no role when one student is being scored. Here a prediction runs
validate, encode, scale, vote. SMOTE is also applied after the train/test
split, otherwise synthetic copies of test students leak into training and
inflate accuracy. Missing values are filled, and the scaler fitted, from the
training split for the same reason. Synthetic records keep their decimal
values. The first version of this build used SMOTE-NC and rounded synthetic
values to whole numbers, and scored 77.1%, about 2.7 points lower.

**Same figures on any computer.** SMOTE's default neighbour search goes
through the maths library, and the number of CPU threads changed the
synthetic students and so the scores (a laptop and a server could differ by
up to a point). The build uses a KD-tree search, which gives identical
results whatever the hardware: `python train.py` reproduces the table above.

**Ensemble versus its parts.** The ensemble beats a single decision tree by
about 24 points and Random Forest by about 6. Against Gradient Boosting, its
strongest member, it is ahead by 0.4 points on accuracy, 1.3 on precision and
0.5 on F1, and level on recall (76.8% against 76.8%).
The honest claim is that it clearly beats a single classifier and edges its best
member; the margin over Gradient Boosting is small.

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
python -m unittest tests.test_system tests.test_scales_and_reports -v
```

Forty-four tests on a temporary database. They check that the model matches
Chapter Three (RF plus GB under a Voting Classifier, 19 inputs, five levels,
20,000+ balanced records, no test student in the training file), that the
reported metrics reproduce, that the web form gives the same class as the raw
model, the grading scales, per-student PDF reports, recommendations in the CSV
(with formula-like text made safe for Excel), "edit and predict again", the
per-student timeline, upgrading an older database, validation, roles,
privacy between users, CSRF, uploads and training.

With `DATABASE_URL` pointing at an empty PostgreSQL database, the same suites
run on PostgreSQL, and `tests/test_postgres.py` adds four checks of its own.

## Hosting

`Procfile` and `render.yaml` are included (`render.yaml` selects Render's free
plan).

```bash
gunicorn -w 1 --threads 4 --timeout 180 run:app
```

Keep it to **one worker** (threads are fine): training progress is tracked in
memory. Set `SECRET_KEY` in the environment, and `COOKIE_SECURE=1` when
serving over HTTPS.

### Keeping accounts on a free host (PostgreSQL)

Free hosts such as Render's free plan wipe local files whenever the service
sleeps or restarts. With the default SQLite file that means every account and
prediction created on the site disappears. To keep them:

1. Create a free PostgreSQL database with any provider and copy its
   connection string. It looks like
   `postgresql://user:password@host/dbname?sslmode=require`.
2. In the Render dashboard open the service, go to Environment, and add
   `DATABASE_URL` with that string as the value.
3. Save. Render redeploys, the app creates its tables and demo accounts in
   the new database once, and from then on records survive restarts.

With `DATABASE_URL` unset the app uses SQLite in `instance/`, which is right
for a laptop. On a paid host with a persistent disk you can stay on SQLite
and set `DATABASE_PATH` to a file on the disk.

Still wiped on a free host, with or without PostgreSQL: uploaded dataset
files and models retrained on the site. Retrain on your own computer with
`python train.py`, commit `models/ensemble.joblib` and `models/metrics.json`,
and push.

Serving peaked at about 385 MB of memory when measured, and retraining needs
more, so do not retrain on a small instance.

## Project layout

```
run.py                entry point
train.py              command-line training
ml/                   schema, pipeline, prediction service, advice text
webapp/grading.py     class names for universities, polytechnics, NCE
webapp/report.py      one-student PDF report
webapp/               Flask app: auth, views, admin, templates, static
data/                 dataset, uploads, balanced training export
models/               ensemble.joblib, metrics.json
tests/                functional and model tests
```

Front-end assets are bundled (Chart.js, Lucide icons, Bricolage Grotesque and
Instrument Sans), so the app works without an internet connection.

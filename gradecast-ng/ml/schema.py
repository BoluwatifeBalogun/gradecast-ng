"""Single source of truth for the dataset schema.

Everything that needs to know about the 19 input attributes (the training
pipeline, the prediction service, form validation and the web forms) reads
from this module, so the model and the interface can never drift apart.

Dataset: "Student Performance Factors" (Kaggle, lainguyn123), Section 3.4 of
the report.
"""

TARGET = "Exam_Score"

# Output classes, ordered from weakest to strongest (Section 3.4.2).
CLASSES = ["Fail", "Pass", "Lower Second", "Upper Second", "First Class"]

# Exam_Score in this dataset only spans 55 to 101 and is tightly packed around
# 67, so the usual 40/45/50/60/70 marks would put almost every student in one
# class and leave "Fail" empty. The score is therefore banded relative to the
# cohort: each value below is the highest Exam_Score that still belongs to the
# class at the same position in CLASSES (the last class is open ended).
DEFAULT_CUTOFFS = [61, 64, 67, 70]

# Approximate cohort shares used if an uploaded dataset has a different score
# scale and the fixed cut-offs would leave a class nearly empty.
FALLBACK_QUANTILES = [0.05, 0.22, 0.55, 0.84]

LOW_MED_HIGH = ["Low", "Medium", "High"]
YES_NO = ["No", "Yes"]

# kind: "num" or "cat". For categorical attributes the order of `options`
# is the ordinal encoding (index 0, 1, 2 ...), weakest to strongest where an
# order exists. `actionable` marks attributes a student or institution can
# realistically change; only those are used for recommendations.
FEATURES = [
    # ---- Study habits -----------------------------------------------------
    dict(name="Hours_Studied", label="Hours studied per week", kind="num",
         min=0, max=44, step=1, unit="hrs", default=20, group="study",
         help="Self-study time in a typical week, outside lectures.",
         actionable=True, better="high"),
    dict(name="Attendance", label="Class attendance", kind="num",
         min=0, max=100, step=1, unit="%", default=80, group="study",
         help="Share of lectures and practicals attended.",
         actionable=True, better="high"),
    dict(name="Previous_Scores", label="Previous score", kind="num",
         min=0, max=100, step=1, unit="/100", default=75, group="study",
         help="Average from the last level completed, for example the ND, "
              "WASSCE or post-UTME result scaled to 100.",
         actionable=False, better="high"),
    dict(name="Tutoring_Sessions", label="Tutoring sessions per month",
         kind="num", min=0, max=8, step=1, unit="", default=1, group="study",
         help="Extra tutorials or study-group sessions attended in a month.",
         actionable=True, better="high"),
    dict(name="Motivation_Level", label="Motivation level", kind="cat",
         options=LOW_MED_HIGH, default="Medium", group="study",
         help="How driven the student is towards academic work.",
         actionable=True, better="high"),
    dict(name="Extracurricular_Activities", label="Extracurricular activities",
         kind="cat", options=YES_NO, default="Yes", group="study",
         help="Takes part in clubs, sports, fellowship or union activities.",
         actionable=True, better="high"),

    # ---- Wellbeing --------------------------------------------------------
    dict(name="Sleep_Hours", label="Sleep per night", kind="num",
         min=3, max=12, step=1, unit="hrs", default=7, group="wellbeing",
         help="Average hours of sleep on a school night.",
         actionable=True, better="mid"),
    dict(name="Physical_Activity", label="Physical activity per week",
         kind="num", min=0, max=6, step=1, unit="hrs", default=3,
         group="wellbeing", help="Hours of exercise or sport in a week.",
         actionable=True, better="high"),
    dict(name="Learning_Disabilities", label="Learning disability",
         kind="cat", options=YES_NO, default="No", group="wellbeing",
         help="A diagnosed or suspected learning difficulty.",
         actionable=False, better="low"),
    dict(name="Gender", label="Gender", kind="cat",
         options=["Female", "Male"], default="Male", group="wellbeing",
         help="Recorded because the dataset includes it. It has almost no "
              "effect on the prediction.",
         actionable=False, better=None),

    # ---- Home and background ----------------------------------------------
    dict(name="Parental_Involvement", label="Parental involvement",
         kind="cat", options=LOW_MED_HIGH, default="Medium", group="home",
         help="How closely parents or guardians follow the student's studies.",
         actionable=True, better="high"),
    dict(name="Parental_Education_Level", label="Parents' education",
         kind="cat", options=["High School", "College", "Postgraduate"],
         default="High School", group="home",
         help="Highest level reached by a parent or guardian. High School "
              "covers SSCE and below, College covers OND, NCE, HND and BSc.",
         actionable=False, better="high"),
    dict(name="Family_Income", label="Family income", kind="cat",
         options=LOW_MED_HIGH, default="Medium", group="home",
         help="Household income relative to other students.",
         actionable=False, better="high"),
    dict(name="Internet_Access", label="Internet access", kind="cat",
         options=YES_NO, default="Yes", group="home",
         help="Reliable data or Wi-Fi for study.",
         actionable=True, better="high"),
    dict(name="Distance_from_Home", label="Distance from home", kind="cat",
         options=["Near", "Moderate", "Far"], default="Near", group="home",
         help="How far the student lives from campus.",
         actionable=True, better="low"),

    # ---- School environment -----------------------------------------------
    dict(name="Access_to_Resources", label="Access to learning resources",
         kind="cat", options=LOW_MED_HIGH, default="Medium", group="school",
         help="Textbooks, library, laboratory and computer access.",
         actionable=True, better="high"),
    dict(name="Teacher_Quality", label="Teaching quality", kind="cat",
         options=LOW_MED_HIGH, default="Medium", group="school",
         help="How effective the student finds the teaching.",
         actionable=True, better="high"),
    dict(name="School_Type", label="Previous school type", kind="cat",
         options=["Public", "Private"], default="Public", group="school",
         help="Type of school last attended.",
         actionable=False, better=None),
    dict(name="Peer_Influence", label="Peer influence", kind="cat",
         options=["Negative", "Neutral", "Positive"], default="Neutral",
         group="school", help="Effect of friends on study habits.",
         actionable=True, better="high"),
]

GROUPS = [
    ("study", "Study habits", "How the student works week to week."),
    ("wellbeing", "Wellbeing", "Rest, activity and personal factors."),
    ("home", "Home and background", "Support and conditions outside school."),
    ("school", "School environment", "What the institution and peers provide."),
]

FEATURE_NAMES = [f["name"] for f in FEATURES]
NUMERIC = [f["name"] for f in FEATURES if f["kind"] == "num"]
CATEGORICAL = [f["name"] for f in FEATURES if f["kind"] == "cat"]
BY_NAME = {f["name"]: f for f in FEATURES}
REQUIRED_COLUMNS = FEATURE_NAMES + [TARGET]


def defaults():
    """A neutral student profile (dataset medians and modes)."""
    return {f["name"]: f["default"] for f in FEATURES}


def validate(raw):
    """Validate one student's attributes.

    Returns (clean, errors). `clean` holds typed values for every valid
    field; `errors` maps a field name to a message the form can show next
    to that field.
    """
    clean, errors = {}, {}
    for f in FEATURES:
        name, value = f["name"], raw.get(f["name"])
        if value is None or str(value).strip() == "":
            errors[name] = f"Enter {f['label'].lower()}."
            continue
        if f["kind"] == "num":
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors[name] = f"{f['label']} must be a number."
                continue
            if number != number or number < f["min"] or number > f["max"]:
                errors[name] = (f"{f['label']} must be between "
                                f"{f['min']} and {f['max']}.")
                continue
            clean[name] = int(number) if float(number).is_integer() else number
        else:
            text = str(value).strip()
            match = next((o for o in f["options"]
                          if o.lower() == text.lower()), None)
            if match is None:
                errors[name] = (f"Choose one of: {', '.join(f['options'])}.")
                continue
            clean[name] = match
    return clean, errors

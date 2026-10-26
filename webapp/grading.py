"""Nigerian grading scales.

The model predicts one of five ordered bands, weakest (0) to strongest (4).
What each band is called depends on the type of institution: a university
student is told "Second Class Upper", a polytechnic student "Upper Credit".
Only the names change; the prediction itself is the same.

To rename a class, edit the `labels` list of its scale (weakest first).
"""

SCALES = {
    "university": {
        "name": "University",
        "award": "degree",
        "labels": ["Fail", "Third Class", "Second Class Lower",
                   "Second Class Upper", "First Class"],
    },
    "polytechnic": {
        "name": "Polytechnic",
        "award": "ND or HND",
        "labels": ["Fail", "Pass", "Lower Credit", "Upper Credit",
                   "Distinction"],
    },
    "coe": {
        "name": "College of Education",
        "award": "NCE",
        "labels": ["Fail", "Pass", "Merit", "Credit", "Distinction"],
    },
}

ORDER = ["university", "polytechnic", "coe"]
DEFAULT = "university"

# Institution type (as stored on each record) -> grading scale.
INSTITUTION_SCALE = {
    "Federal University": "university",
    "State University": "university",
    "Private University": "university",
    "Federal Polytechnic": "polytechnic",
    "State Polytechnic": "polytechnic",
    "Private Polytechnic": "polytechnic",
    "Monotechnic or Specialised College": "polytechnic",
    "College of Education": "coe",
}

# Institution types grouped for the form's drop-down.
GROUPS = [
    ("Universities", ["Federal University", "State University",
                      "Private University"]),
    ("Polytechnics and monotechnics", [
        "Federal Polytechnic", "State Polytechnic", "Private Polytechnic",
        "Monotechnic or Specialised College"]),
    ("Colleges of Education", ["College of Education"]),
]


def scale_for(institution_type):
    """Scale key for an institution type. Records saved before institution
    type was required fall back to the university scale."""
    return INSTITUTION_SCALE.get(institution_type or "", DEFAULT)


def valid_scale(key):
    return key if key in SCALES else None


def labels(scale_key):
    return SCALES.get(scale_key, SCALES[DEFAULT])["labels"]


def label(index, institution_type=None, scale_key=None):
    """Display name of band `index` for an institution type (or scale)."""
    key = scale_key or scale_for(institution_type)
    return labels(key)[int(index)]


def any_scale_label(index):
    """One band named on every scale, for filters: 'First Class / Distinction'."""
    names = []
    for key in ORDER:
        name = SCALES[key]["labels"][int(index)]
        if name not in names:
            names.append(name)
    return " / ".join(names)

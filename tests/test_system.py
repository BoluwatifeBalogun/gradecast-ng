"""Functional and model tests (Section 3.8 of the report).

    python -m unittest tests.test_system -v

Runs against a temporary database, so it never touches real records.
"""
import io
import json
import os
import re
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import pipeline as P, schema as S, service
from webapp import admin as admin_module, create_app

TOKEN = re.compile(r'name="csrf_token" value="([^"]+)"')

WEAK = {**S.defaults(), "Hours_Studied": 7, "Attendance": 62,
        "Previous_Scores": 55, "Tutoring_Sessions": 0,
        "Motivation_Level": "Low", "Access_to_Resources": "Low",
        "Parental_Involvement": "Low", "Internet_Access": "No",
        "Peer_Influence": "Negative"}
STRONG = {**S.defaults(), "Hours_Studied": 32, "Attendance": 97,
          "Previous_Scores": 92, "Tutoring_Sessions": 4,
          "Motivation_Level": "High", "Access_to_Resources": "High",
          "Parental_Involvement": "High", "Peer_Influence": "Positive"}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.app = create_app({
            "DATABASE": os.path.join(cls.tmp.name, "test.db"),
            "UPLOAD_DIR": cls.tmp.name, "TESTING": True})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def client(self, email=None, password=None):
        c = self.app.test_client()
        if email:
            r = c.post("/login", data={"csrf_token": self.token(c, "/login"),
                                       "email": email, "password": password})
            self.assertEqual(r.status_code, 302)
        return c

    def token(self, c, path="/predict"):
        return TOKEN.search(c.get(path).text).group(1)

    def admin(self):
        return self.client("admin@gradecast.ng", "Admin@2026")

    def officer(self):
        return self.client("officer@gradecast.ng", "Officer@2026")

    def student(self):
        return self.client("student@gradecast.ng", "Student@2026")

    def submit(self, c, features, name="Test Student", **record):
        data = {"csrf_token": self.token(c), "student_name": name,
                **record, **features}
        return c.post("/predict", data=data)


class ModelTests(unittest.TestCase):
    """The trained artefact matches what Chapter Three describes."""

    @classmethod
    def setUpClass(cls):
        service.load(force=True)
        cls.bundle = service._bundle
        cls.metrics = service.metrics()
        df, _, _ = P.clean(P.load_dataset(P.DEFAULT_DATASET))
        y = P.band(df[S.TARGET], cls.bundle["cutoffs"])
        X = P.encode(df)
        cls.split = train_test_split(X, y, test_size=0.2, stratify=y,
                                     random_state=42)

    def test_ensemble_is_rf_plus_gb_voting(self):
        model = self.bundle["model"]
        self.assertEqual(type(model).__name__, "VotingClassifier")
        kinds = {type(e).__name__ for e in model.named_estimators_.values()}
        self.assertEqual(kinds, {"RandomForestClassifier",
                                 "GradientBoostingClassifier"})

    def test_nineteen_inputs_and_five_classes(self):
        self.assertEqual(len(self.bundle["features"]), 19)
        self.assertNotIn("Exam_Score", self.bundle["features"])
        self.assertEqual(self.bundle["classes"], S.CLASSES)
        self.assertEqual(len(S.NUMERIC), 6)
        self.assertEqual(len(S.CATEGORICAL), 13)

    def test_reported_metrics_reproduce(self):
        _, X_te, _, y_te = self.split
        pred = self.bundle["model"].predict(P.scale(X_te, self.bundle["scaler"]))
        self.assertAlmostEqual(float((pred == y_te).mean()),
                               self.metrics["results"]["ensemble"]["accuracy"],
                               places=6)

    def test_smote_balances_past_20000_records(self):
        after = self.metrics["class_distribution"]["train_after"]
        self.assertEqual(len(set(after)), 1)
        self.assertGreater(sum(after), 20000)

    def test_no_test_student_in_training_file(self):
        X_tr, X_te, _, _ = self.split
        bal = pd.read_csv(P.BALANCED_PATH)
        self.assertEqual(len(bal), sum(
            self.metrics["class_distribution"]["train_after"]))
        original = bal[bal.Record_Type == "Original"][S.FEATURE_NAMES]
        self.assertTrue(np.allclose(original.to_numpy(), X_tr))
        train_keys = {tuple(r) for r in np.round(
            bal[S.FEATURE_NAMES].to_numpy(), 3)}
        leaked = sum(tuple(r) in train_keys for r in np.round(X_te, 3))
        self.assertEqual(leaked, 0)

    def test_soft_vote_is_weighted_average(self):
        _, X_te, _, _ = self.split
        model = self.bundle["model"]
        Xs = P.scale(X_te[:100], self.bundle["scaler"])
        rf = model.named_estimators_["rf"].predict_proba(Xs)
        gb = model.named_estimators_["gb"].predict_proba(Xs)
        w = model.weights
        self.assertTrue(np.allclose(model.predict_proba(Xs),
                                    (w[0] * rf + w[1] * gb) / sum(w)))

    def test_ensemble_beats_single_tree(self):
        r = self.metrics["results"]
        self.assertGreater(r["ensemble"]["accuracy"],
                           r["single_decision_tree"]["accuracy"] + 0.10)

    def test_weak_and_strong_profiles(self):
        self.assertEqual(service.predict(WEAK)["risk"], "At risk")
        self.assertEqual(service.predict(STRONG)["risk"], "On track")

    def test_more_attendance_never_hurts_much(self):
        base = S.defaults()
        ladder = [service.predict({**base, "Attendance": a})["ladder"]
                  for a in range(60, 101, 5)]
        for lo, hi in zip(ladder, ladder[1:]):
            self.assertGreater(hi, lo - 0.15)
        self.assertGreater(ladder[-1], ladder[0] + 1)

    def test_out_of_range_is_reported(self):
        notes = service.out_of_range({**S.defaults(), "Attendance": 40})
        self.assertEqual(notes[0]["label"], "Class attendance")
        self.assertEqual(service.out_of_range(S.defaults()), [])


class ValidationTests(unittest.TestCase):
    def test_every_numeric_boundary(self):
        for f in S.FEATURES:
            if f["kind"] != "num":
                continue
            for value, valid in [(f["min"], True), (f["max"], True),
                                 (f["min"] - 1, False), (f["max"] + 1, False),
                                 ("abc", False), ("", False), ("nan", False)]:
                _, errors = S.validate({**S.defaults(), f["name"]: value})
                self.assertEqual(f["name"] not in errors, valid,
                                 f"{f['name']}={value!r}")

    def test_every_categorical_option(self):
        for f in S.FEATURES:
            if f["kind"] != "cat":
                continue
            for option in f["options"]:
                clean, errors = S.validate({**S.defaults(),
                                            f["name"]: option.lower()})
                self.assertFalse(errors)
                self.assertEqual(clean[f["name"]], option)
            _, errors = S.validate({**S.defaults(), f["name"]: "Unknown"})
            self.assertIn(f["name"], errors)

    def test_missing_fields_all_reported(self):
        _, errors = S.validate({})
        self.assertEqual(len(errors), 19)


class WebTests(Base):
    def test_public_pages(self):
        c = self.client()
        self.assertEqual(c.get("/").status_code, 200)
        self.assertEqual(c.get("/login").status_code, 200)
        self.assertEqual(c.get("/register").status_code, 200)
        self.assertEqual(c.get("/dashboard").status_code, 302)
        self.assertEqual(c.get("/missing").status_code, 404)

    def test_live_preview(self):
        r = self.client().post("/api/preview", json={"Attendance": 95,
                                                     "Hours_Studied": 30})
        self.assertEqual(r.status_code, 200)
        self.assertAlmostEqual(sum(r.json["proba"]), 1.0, places=5)
        r = self.client().post("/api/preview", json={"Attendance": 500})
        self.assertEqual(r.status_code, 400)

    def test_register_rules(self):
        c = self.client()
        t = self.token(c, "/register")
        r = c.post("/register", data={"csrf_token": t, "name": "",
                                      "email": "bad", "password": "123",
                                      "role": "admin"})
        for text in ("Enter your full name", "valid email",
                     "at least 8", "Choose student or academic officer"):
            self.assertIn(text, r.text.replace("At least 8", "at least 8"))
        r = c.post("/register", data={"csrf_token": t, "name": "Dup",
                                      "email": "admin@gradecast.ng",
                                      "password": "password1",
                                      "role": "student"})
        self.assertIn("already exists", r.text)

    def test_login_rejects_wrong_password_and_open_redirect(self):
        c = self.client()
        r = c.post("/login", data={"csrf_token": self.token(c, "/login"),
                                   "email": "admin@gradecast.ng",
                                   "password": "nope"})
        self.assertIn("do not match", r.text)
        r = c.post("/login?next=//evil.example", data={
            "csrf_token": self.token(c, "/login"),
            "email": "admin@gradecast.ng", "password": "Admin@2026"})
        self.assertEqual(r.headers["Location"], "/dashboard")

    def test_csrf_required(self):
        c = self.student()
        self.assertEqual(c.post("/predict", data={"student_name": "x"}
                                ).status_code, 400)
        self.assertEqual(c.post("/logout").status_code, 400)

    def test_prediction_flow_and_storage(self):
        c = self.officer()
        r = self.submit(c, WEAK, name="Weak Profile", state="Kano",
                        institution_type="Federal Polytechnic",
                        student_ref="ND/25/001")
        self.assertEqual(r.status_code, 302)
        page = c.get(r.headers["Location"]).text
        self.assertIn("At risk", page)
        self.assertIn("Kano (North West)", page)
        self.assertIn("Raise class attendance", page)
        r = self.submit(c, STRONG, name="Strong Profile")
        self.assertIn("On track", c.get(r.headers["Location"]).text)

    def test_invalid_form_is_returned_with_messages(self):
        c = self.officer()
        r = self.submit(c, {**S.defaults(), "Attendance": 140,
                            "Motivation_Level": "Extreme"}, name="")
        self.assertEqual(r.status_code, 400)
        self.assertIn("must be between 0 and 100", r.text)
        self.assertIn("Choose one of: Low, Medium, High", r.text)
        self.assertIn("Enter the student&#39;s name", r.text)
        r = self.submit(c, S.defaults(), state="Atlantis")
        self.assertIn("Choose a state from the list", r.text)

    def test_web_prediction_equals_offline_model(self):
        """No train/serve skew: the form path and the raw model agree."""
        df, _, _ = P.clean(P.load_dataset(P.DEFAULT_DATASET))
        sample = df.sample(25, random_state=11)
        bundle = service._bundle
        offline = bundle["model"].predict(
            P.scale(P.encode(sample), bundle["scaler"]))
        c = self.officer()
        for (_, row), expected in zip(sample.iterrows(), offline):
            feats = {n: row[n] for n in S.FEATURE_NAMES}
            r = self.submit(c, feats, name="Skew Check")
            pid = int(r.headers["Location"].rsplit("/", 1)[1])
            with self.app.app_context():
                from webapp import db
                got = db.get_db().execute(
                    "SELECT label_index FROM predictions WHERE id=?",
                    (pid,)).fetchone()[0]
            self.assertEqual(got, int(expected))

    def test_records_are_private(self):
        officer, student, admin = self.officer(), self.student(), self.admin()
        r = self.submit(officer, WEAK, name="Private Record")
        url = r.headers["Location"]
        self.assertEqual(student.get(url).status_code, 404)
        self.assertEqual(admin.get(url).status_code, 200)
        self.assertEqual(student.post(url + "/delete", data={
            "csrf_token": self.token(student)}).status_code, 404)
        self.assertNotIn("Private Record", student.get("/history").text)
        self.assertNotIn("Private Record",
                         student.get("/history/export.csv").text)

    def test_admin_area_is_admin_only(self):
        for c in (self.student(), self.officer()):
            for path in ("/admin/datasets", "/admin/training", "/admin/users",
                         "/admin/training/status"):
                self.assertEqual(c.get(path).status_code, 403, path)
            self.assertEqual(c.post("/admin/training/start", data={
                "csrf_token": self.token(c)}).status_code, 403)

    def test_hostile_text_is_escaped_and_parameterised(self):
        c = self.officer()
        r = self.submit(c, S.defaults(),
                        name="<script>alert(1)</script>",
                        student_ref="'; DROP TABLE users;--")
        page = c.get(r.headers["Location"]).text
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)
        r = c.get("/history?q=' OR 1=1 --")
        self.assertEqual(r.status_code, 200)
        self.assertIn("No predictions match", r.text)
        self.assertEqual(self.admin().get("/admin/users").status_code, 200)

    def test_history_filters_and_export(self):
        c = self.officer()
        self.submit(c, WEAK, name="Filter Weak Zed")
        r = c.get("/history?q=Filter+Weak+Zed&risk=At+risk")
        self.assertIn("Filter Weak Zed", r.text)
        r = c.get("/history?q=Filter+Weak+Zed&risk=On+track")
        self.assertIn("No predictions match", r.text)
        csv_text = c.get("/history/export.csv?q=Filter+Weak+Zed").text
        rows = pd.read_csv(io.StringIO(csv_text))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["Attendance"], 62)

    def _upload(self, c, frame_or_bytes, name="upload.csv"):
        if isinstance(frame_or_bytes, pd.DataFrame):
            frame_or_bytes = frame_or_bytes.to_csv(index=False).encode()
        return c.post("/admin/datasets/upload", data={
            "csrf_token": self.token(c, "/admin/datasets"),
            "file": (io.BytesIO(frame_or_bytes), name)},
            content_type="multipart/form-data", follow_redirects=True)

    def test_dataset_upload_rules(self):
        c = self.admin()
        df = pd.read_csv(P.DEFAULT_DATASET)
        self.assertIn("Only CSV files", self._upload(c, b"hi", "a.txt").text)
        self.assertIn("missing these columns: Attendance", self._upload(
            c, df.drop(columns=["Attendance"]).head(400)).text)
        self.assertIn("at least 200 rows",
                      self._upload(c, df.head(50)).text)
        self.assertIn("Upload rejected", self._upload(c, b"", "empty.csv").text)
        flat = df.head(600).assign(Exam_Score=70)
        self.assertIn("does not vary enough", self._upload(c, flat).text)

    def test_messy_but_valid_dataset_is_accepted(self):
        """Stray spaces, mixed case, blanks and a wider score scale."""
        df = pd.read_csv(P.DEFAULT_DATASET).sample(900, random_state=5)
        df["Motivation_Level"] = df["Motivation_Level"].str.upper() + "  "
        df["Gender"] = " " + df["Gender"].str.lower()
        df.loc[df.index[:40], "Attendance"] = np.nan
        df.loc[df.index[40:60], "Family_Income"] = "n/a"
        df["Exam_Score"] = ((df["Exam_Score"] - 55) * 2.2).clip(0, 100)
        c = self.admin()
        self.assertIn("Uploaded messy.csv", self._upload(c, df, "messy.csv").text)
        path = os.path.join(self.tmp.name, "messy_direct.csv")
        df.to_csv(path, index=False)
        clean, report, _ = P.clean(P.load_dataset(path))
        self.assertEqual(clean.isna().sum().sum(), 0)
        self.assertGreaterEqual(report["missing_cells"], 60)
        cutoffs, mode = P.choose_cutoffs(clean[S.TARGET])
        self.assertEqual(mode, "quantile")
        counts = np.bincount(P.band(clean[S.TARGET], cutoffs), minlength=5)
        self.assertTrue((counts >= 10).all())

    def test_second_training_run_is_refused(self):
        c = self.admin()
        admin_module._job["running"] = True
        try:
            r = c.post("/admin/training/start", data={
                "csrf_token": self.token(c, "/admin/training")})
            self.assertEqual(r.status_code, 409)
        finally:
            admin_module._job["running"] = False

    def test_deleting_a_user_removes_their_predictions(self):
        c = self.client()
        c.post("/register", data={"csrf_token": self.token(c, "/register"),
                                  "name": "Temp User",
                                  "email": "temp@example.com",
                                  "password": "password1", "role": "student"})
        self.submit(c, S.defaults(), name="Temp Prediction")
        a = self.admin()
        with self.app.app_context():
            from webapp import db
            uid = db.get_db().execute(
                "SELECT id FROM users WHERE email='temp@example.com'"
            ).fetchone()[0]
        a.post(f"/admin/users/{uid}/delete",
               data={"csrf_token": self.token(a, "/admin/users")})
        self.assertNotIn("Temp Prediction", a.get("/history?q=Temp").text)
        r = a.post("/admin/users/1/delete",
                   data={"csrf_token": self.token(a, "/admin/users")},
                   follow_redirects=True)
        self.assertIn("cannot delete your own account", r.text)

    def test_passwords_are_hashed(self):
        with self.app.app_context():
            from webapp import db
            for row in db.get_db().execute("SELECT password_hash FROM users"):
                self.assertNotIn("2026", row[0])
                self.assertTrue(row[0].startswith(("scrypt:", "pbkdf2:")))


if __name__ == "__main__":
    unittest.main(verbosity=2)

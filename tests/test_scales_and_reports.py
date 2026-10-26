"""Grading scales, per-student reports, recommendations in exports, and
"edit and predict again"."""
import io
import json
import os
import re
import sqlite3
import tempfile
import unittest

import pandas as pd
from pypdf import PdfReader

from ml import schema

TYPICAL = {k: str(v) for k, v in schema.defaults().items()}
WEAK = {**TYPICAL, "Attendance": "62", "Hours_Studied": "8",
        "Previous_Scores": "55", "Motivation_Level": "Low",
        "Tutoring_Sessions": "0"}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Runs on PostgreSQL when DATABASE_URL is set, otherwise SQLite.
        from webapp import create_app
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.tmp.name, "t.db")
        cls.app = create_app({"DATABASE": cls.db_path, "SEED_SAMPLE": False,
                              "TESTING": True})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def login(self, email, password):
        c = self.app.test_client()
        t = re.search(r'name="csrf_token" value="([^"]+)"',
                      c.get("/login").text).group(1)
        c.post("/login", data={"csrf_token": t, "email": email,
                               "password": password})
        return c

    def officer(self):
        return self.login("officer@gradecast.ng", "Officer@2026")

    def student(self):
        return self.login("student@gradecast.ng", "Student@2026")

    def submit(self, c, features, institution_type, name="Scale Test"):
        t = re.search(r'name="csrf_token" value="([^"]+)"',
                      c.get("/predict").text).group(1)
        r = c.post("/predict", data={
            "csrf_token": t, "student_name": name,
            "institution_type": institution_type, **features})
        self.assertEqual(r.status_code, 302, r.text[:400])
        return int(r.headers["Location"].rsplit("/", 1)[1])

    @staticmethod
    def verdict(html):
        return re.search(r'id="verdict-title">([^<]+)<', html).group(1)


class ScaleTests(Base):
    def test_same_student_named_per_institution_type(self):
        from webapp import grading
        c = self.officer()
        seen = {}
        for inst, key in [("Federal University", "university"),
                          ("State Polytechnic", "polytechnic"),
                          ("College of Education", "coe")]:
            pid = self.submit(c, WEAK, inst)
            html = c.get(f"/predictions/{pid}").text
            with self.app.app_context():
                from webapp import db
                band = db.get_db().execute(
                    "SELECT label_index FROM predictions WHERE id=?",
                    (pid,)).fetchone()[0]
            self.assertEqual(self.verdict(html),
                             grading.SCALES[key]["labels"][band])
            seen[key] = band
        self.assertEqual(len(set(seen.values())), 1,
                         "the scale must not change the prediction")

    def test_scale_labels(self):
        from webapp import grading
        self.assertEqual(grading.SCALES["polytechnic"]["labels"][::-1],
                         ["Distinction", "Upper Credit", "Lower Credit",
                          "Pass", "Fail"])
        self.assertEqual(grading.SCALES["university"]["labels"][::-1],
                         ["First Class", "Second Class Upper",
                          "Second Class Lower", "Third Class", "Fail"])
        self.assertEqual(grading.scale_for("Monotechnic or Specialised "
                                           "College"), "polytechnic")
        self.assertEqual(grading.scale_for(None), "university")

    def test_institution_type_is_required(self):
        c = self.officer()
        t = re.search(r'name="csrf_token" value="([^"]+)"',
                      c.get("/predict").text).group(1)
        r = c.post("/predict", data={"csrf_token": t, "student_name": "No Inst",
                                     **TYPICAL})
        self.assertEqual(r.status_code, 400)
        self.assertIn("Choose the institution type", r.text)

    def test_mixed_pages_switch_scale(self):
        c = self.officer()
        self.submit(c, TYPICAL, "Federal Polytechnic")
        self.assertIn("Lower Credit", c.get("/dashboard?scale=polytechnic").text)
        self.assertIn("Second Class Lower",
                      c.get("/dashboard?scale=university").text)
        self.assertIn("Merit", c.get("/insights?scale=coe").text)
        self.assertEqual(c.get("/dashboard?scale=nonsense").status_code, 200)

    def test_band_filter(self):
        c = self.officer()
        self.submit(c, WEAK, "Federal University", name="Band Filter Zed")
        page = c.get("/history?q=Band+Filter+Zed&band=0").text + \
            c.get("/history?q=Band+Filter+Zed&band=1").text
        self.assertIn("Band Filter Zed", page)
        self.assertIn("No predictions match",
                      c.get("/history?q=Band+Filter+Zed&band=4").text)
        self.assertEqual(c.get("/history?band=99").status_code, 200)

    def test_landing_demo_offers_all_scales(self):
        page = self.app.test_client().get("/").text
        for name in ("University", "Polytechnic", "College of Education"):
            self.assertIn(name, page)
        r = self.app.test_client().post("/api/preview",
                                        json={"Attendance": 95})
        self.assertIn(r.json["votes"]["random_forest"], range(5))


class ReportTests(Base):
    def test_pdf_report_has_prediction_and_recommendations(self):
        c = self.officer()
        pid = self.submit(c, WEAK, "Federal Polytechnic",
                          name="Adébáyọ̀ Report")
        r = c.get(f"/predictions/{pid}/report.pdf")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, "application/pdf")
        self.assertIn("attachment", r.headers["Content-Disposition"])
        self.assertIn("adebayo-report", r.headers["Content-Disposition"])
        text = " ".join(p.extract_text() for p in
                        PdfReader(io.BytesIO(r.data)).pages)
        for needle in ("Adébáyọ̀ Report", "Polytechnic grading",
                       "Recommendations to improve", "Raise class attendance",
                       "Details entered", "Chance of each class"):
            self.assertIn(needle, text, needle)

    def test_report_is_private(self):
        pid = self.submit(self.officer(), WEAK, "Federal University")
        self.assertEqual(
            self.student().get(f"/predictions/{pid}/report.pdf").status_code,
            404)

    def test_csv_includes_scale_and_recommendations(self):
        c = self.officer()
        self.submit(c, WEAK, "State Polytechnic", name="Csv Recs Zed")
        text = c.get("/history/export.csv?q=Csv+Recs+Zed").text
        self.assertTrue(text.startswith("﻿"), "Excel needs the BOM")
        rows = pd.read_csv(io.StringIO(text.lstrip("﻿")))
        self.assertEqual(len(rows), 1)
        row = rows.iloc[0]
        self.assertEqual(row["Grading scale"], "Polytechnic")
        self.assertIn(row["Predicted class"],
                      ["Fail", "Pass", "Lower Credit", "Upper Credit",
                       "Distinction"])
        self.assertTrue(str(row["Recommendations"]).startswith("1. "))

    def test_recommendations_are_saved_and_reused(self):
        from ml import service
        from webapp import db
        c = self.officer()
        pid = self.submit(c, WEAK, "Federal University")
        with self.app.app_context():
            row = db.get_db().execute(
                "SELECT explain_json, explain_version FROM predictions"
                " WHERE id=?", (pid,)).fetchone()
        self.assertTrue(row["explain_json"])
        self.assertTrue(row["explain_version"].startswith(service.version()))
        saved = json.loads(row["explain_json"])
        self.assertTrue(saved["levers"])
        page = c.get(f"/predictions/{pid}").text
        self.assertIn(saved["levers"][0]["advice"]["title"], page)


class EditAgainTests(Base):
    def test_form_starts_from_earlier_prediction(self):
        c = self.officer()
        pid = self.submit(c, {**WEAK, "Hours_Studied": "11"},
                          "State Polytechnic", name="Edit Again Zed")
        page = c.get(f"/predict?from={pid}").text
        self.assertIn("Edit and predict again", page)
        self.assertIn('value="Edit Again Zed"', page)
        self.assertIn('id="f-Hours_Studied" name="Hours_Studied" min="0" '
                      'max="44" step="1" value="11"', page)
        self.assertRegex(page, r"<option selected>State Polytechnic</option>")
        self.assertEqual(self.student().get(f"/predict?from={pid}")
                         .status_code, 404)

    def test_new_form_remembers_last_institution(self):
        c = self.officer()
        self.submit(c, TYPICAL, "College of Education")
        self.assertIn("<option selected>College of Education</option>",
                      c.get("/predict").text)


class MigrationTests(unittest.TestCase):
    @unittest.skipIf(os.environ.get("DATABASE_URL"), "SQLite-only check")
    def test_old_database_is_upgraded(self):
        from webapp import create_app
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.db")
            conn = sqlite3.connect(path)
            conn.executescript("""
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
                    student_name TEXT, student_ref TEXT,
                    institution_type TEXT, state TEXT, zone TEXT,
                    inputs_json TEXT, label TEXT, label_index INTEGER,
                    confidence REAL, proba_json TEXT, votes_json TEXT,
                    risk TEXT, model_version TEXT,
                    is_sample INTEGER DEFAULT 0, created_at TEXT);""")
            conn.commit()
            conn.close()
            create_app({"DATABASE": path, "SEED_SAMPLE": False})
            cols = {r[1] for r in sqlite3.connect(path).execute(
                "PRAGMA table_info(predictions)")}
            self.assertIn("explain_json", cols)
            self.assertIn("explain_version", cols)


if __name__ == "__main__":
    unittest.main()


class TimelineTests(Base):
    def test_follow_up_shows_change(self):
        c = self.officer()
        weak = self.submit(c, WEAK, "Federal Polytechnic", name="Trend Zed")
        from webapp import db
        with self.app.app_context():
            db.get_db().execute(
                "UPDATE predictions SET student_ref='TR/1' WHERE id=?", (weak,))
            db.get_db().commit()
        t = re.search(r'name="csrf_token" value="([^"]+)"',
                      c.get("/predict").text).group(1)
        better = {**WEAK, "Attendance": "98", "Hours_Studied": "30"}
        r = c.post("/predict", data={"csrf_token": t, "student_name": "Trend Zed",
                                     "student_ref": "TR/1",
                                     "institution_type": "Federal Polytechnic",
                                     **better})
        page = c.get(r.headers["Location"]).text
        self.assertIn("How the outlook has changed", page)
        self.assertRegex(page, r"Up \d level")
        # A single prediction has no timeline, and another account's
        # student with the same reference is not included.
        self.assertNotIn("How the outlook has changed",
                         self.student().get("/dashboard").text)
        other = self.student()
        t = re.search(r'name="csrf_token" value="([^"]+)"',
                      other.get("/predict").text).group(1)
        r = other.post("/predict", data={"csrf_token": t, "student_name": "Trend Zed",
                                         "student_ref": "TR/1",
                                         "institution_type": "Federal Polytechnic",
                                         **WEAK})
        self.assertNotIn("How the outlook has changed",
                         other.get(r.headers["Location"]).text)


class CsvSafetyTests(Base):
    def test_formula_like_text_is_neutralised(self):
        c = self.officer()
        self.submit(c, TYPICAL, "Federal University",
                    name='=HYPERLINK("http://x.example","go")')
        text = c.get("/history/export.csv?q=hyperlink").text.lstrip("﻿")
        rows = list(__import__("csv").reader(io.StringIO(text)))
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1][1].startswith("'="), rows[1][1])

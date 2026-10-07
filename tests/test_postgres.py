"""Checks that only matter on PostgreSQL. Skipped unless DATABASE_URL is set.

    DATABASE_URL=postgresql://user:pass@host/dbname \\
        python -m unittest tests.test_postgres -v

Use an empty, throwaway database: the tests create and delete records.
"""
import os
import re
import tempfile
import threading
import unittest

from ml import schema

URL = os.environ.get("DATABASE_URL")


def token(html):
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


@unittest.skipUnless(URL, "DATABASE_URL is not set")
class PostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

    def app(self):
        from webapp import create_app
        return create_app({"DATABASE": os.path.join(self.tmp.name, "x.db"),
                           "SEED_SAMPLE": False, "TESTING": True})

    def login(self, client, email, password):
        t = token(client.get("/login").text)
        return client.post("/login", data={"csrf_token": t, "email": email,
                                           "password": password})

    def test_1_account_and_prediction_survive_a_restart(self):
        from webapp import db
        app = self.app()
        self.assertTrue(db.using_postgres())
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "x.db")),
                         "no SQLite file should be created")
        c = app.test_client()
        t = token(c.get("/register").text)
        r = c.post("/register", data={
            "csrf_token": t, "name": "Persist Test", "role": "officer",
            "email": "persist@example.com", "password": "StaysPut2026"})
        self.assertEqual(r.status_code, 302)
        t = token(c.get("/predict").text)
        form = {"csrf_token": t, "student_name": "Adaeze OKAFOR",
                "student_ref": "HND1/CS/26/777", "state": "Lagos"}
        form.update({k: str(v) for k, v in schema.defaults().items()})
        r = c.post("/predict", data=form)
        self.assertEqual(r.status_code, 302)
        self.assertRegex(r.headers["Location"], r"/predictions/\d+$")
        self.assertEqual(c.get(r.headers["Location"]).status_code, 200)

        # A second app object stands in for the server after a restart.
        app2 = self.app()
        c2 = app2.test_client()
        r = self.login(c2, "persist@example.com", "StaysPut2026")
        self.assertEqual(r.status_code, 302, "account lost after restart")
        page = c2.get("/history").text
        self.assertIn("Adaeze OKAFOR", page)
        with app2.app_context():
            n = db.get_db().execute(
                "SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
        self.assertEqual(n, 1, "demo accounts must not be created twice")

    def test_2_search_ignores_case(self):
        c = self.app().test_client()
        self.login(c, "persist@example.com", "StaysPut2026")
        self.assertIn("Adaeze OKAFOR", c.get("/history?q=okafor").text)
        self.assertIn("Adaeze OKAFOR", c.get("/history?q=hnd1/cs").text)
        self.assertIn("No predictions match", c.get("/history?q=zzz").text)
        self.assertIn("Adaeze OKAFOR", c.get("/history/export.csv").text)
        self.assertEqual(c.get("/dashboard").status_code, 200)

    def test_3_admin_pages_and_cascade_delete(self):
        from webapp import db
        app = self.app()
        c = app.test_client()
        self.login(c, "admin@gradecast.ng", "Admin@2026")
        for url in ("/dashboard", "/history", "/insights", "/admin/datasets",
                    "/admin/training", "/admin/users"):
            self.assertEqual(c.get(url).status_code, 200, url)
        t = token(c.get("/admin/datasets").text)
        with app.app_context():
            did = db.get_db().execute(
                "SELECT id FROM datasets").fetchone()["id"]
            uid = db.get_db().execute(
                "SELECT id FROM users WHERE email='persist@example.com'"
            ).fetchone()["id"]
        r = c.post(f"/admin/datasets/{did}/activate", data={"csrf_token": t})
        self.assertEqual(r.status_code, 302)
        r = c.post("/admin/sample/add", data={"csrf_token": t})
        self.assertIn("Added 48 sample records", c.get("/admin/datasets").text)
        r = c.post("/admin/sample/clear", data={"csrf_token": t})
        self.assertIn("Removed 48 sample records",
                      c.get("/admin/datasets").text)
        c.post(f"/admin/users/{uid}/role",
               data={"csrf_token": t, "role": "student"})
        c.post(f"/admin/users/{uid}/delete", data={"csrf_token": t})
        with app.app_context():
            left = db.get_db().execute(
                "SELECT COUNT(*) FROM predictions WHERE user_id=?",
                (uid,)).fetchone()[0]
        self.assertEqual(left, 0, "predictions must go with their owner")

    def test_4_many_requests_at_once(self):
        app = self.app()
        errors = []

        def hit():
            try:
                c = app.test_client()
                self.login(c, "officer@gradecast.ng", "Officer@2026")
                for _ in range(5):
                    if c.get("/dashboard").status_code != 200:
                        errors.append("bad status")
            except Exception as exc:
                errors.append(repr(exc))

        threads = [threading.Thread(target=hit) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()

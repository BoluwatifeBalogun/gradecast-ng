"""One-student PDF report: the prediction, the chance of each class, the
recommendations and the details entered."""

import io
import json
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from xml.sax.saxutils import escape

from ml import advice, schema

from . import grading, recommend

FONT_DIR = Path(__file__).resolve().parent / "static" / "fonts" / "pdf"
_fonts_ready = False

GREEN = colors.HexColor("#008751")
FOREST = colors.HexColor("#062e22")
MINT = colors.HexColor("#e2f4ea")
PAPER = colors.HexColor("#f4f8f5")
LINE = colors.HexColor("#dbe6df")
MUTED = colors.HexColor("#53665d")
INK = colors.HexColor("#10231b")
BAND = [colors.HexColor(c) for c in
        ("#d24b35", "#e3a322", "#8fd3ad", "#1ea76c", "#00693e")]


def _fonts():
    global _fonts_ready
    if not _fonts_ready:
        pdfmetrics.registerFont(TTFont("GC", str(FONT_DIR / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(
            TTFont("GC-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("GC", normal="GC", bold="GC-Bold")
        _fonts_ready = True


def _styles():
    base = dict(fontName="GC", textColor=INK, alignment=TA_LEFT)
    return {
        "body": ParagraphStyle("body", fontSize=9.5, leading=13.5, **base),
        "small": ParagraphStyle("small", fontSize=8, leading=11,
                                **{**base, "textColor": MUTED}),
        "h2": ParagraphStyle("h2", fontName="GC-Bold", fontSize=12.5,
                             leading=16, textColor=FOREST, spaceBefore=12,
                             spaceAfter=6),
        "big": ParagraphStyle("big", fontName="GC-Bold", fontSize=24,
                              leading=28, textColor=FOREST),
        "label": ParagraphStyle("label", fontSize=8.5, leading=11,
                                **{**base, "textColor": MUTED}),
        "cell": ParagraphStyle("cell", fontSize=9, leading=12, **base),
        "cellb": ParagraphStyle("cellb", fontName="GC-Bold", fontSize=9,
                                leading=12, textColor=INK),
        "rec_title": ParagraphStyle("rec_title", fontName="GC-Bold",
                                    fontSize=10.5, leading=14,
                                    textColor=FOREST),
    }


def _p(text, style):
    return Paragraph(escape(str(text)), style)


def _when(iso):
    try:
        moment = datetime.fromisoformat(iso) + timedelta(hours=1)  # WAT
    except (TypeError, ValueError):
        return iso or ""
    return f"{moment.day} {moment:%B %Y, %H:%M} (WAT)"


def filename(row):
    plain = unicodedata.normalize("NFKD", row["student_name"])
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    slug = re.sub(r"[^A-Za-z0-9]+", "-", plain).strip("-")
    return f"gradecast-{slug.lower() or 'student'}-{row['id']}.pdf"


def build(row, explanation, prepared_for, metrics=None):
    """Return the PDF as bytes."""
    _fonts()
    st = _styles()
    scale = grading.scale_for(row["institution_type"])
    scale_name = grading.SCALES[scale]["name"]
    names = grading.labels(scale)
    proba = json.loads(row["proba_json"])
    inputs = json.loads(row["inputs_json"])
    idx = int(row["label_index"])
    predicted = names[idx]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=30 * mm, bottomMargin=20 * mm,
        title=f"{row['student_name']}: {predicted}",
        author="GradeCast NG")
    width = doc.width - 12  # the frame keeps 6 pt of padding each side

    def page(canvas, _doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(GREEN)
        canvas.rect(0, h - 20 * mm, w, 20 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("GC-Bold", 14)
        canvas.drawString(18 * mm, h - 12.5 * mm, "GradeCast NG")
        canvas.setFont("GC", 9)
        canvas.drawRightString(w - 18 * mm, h - 12.5 * mm,
                               "Student performance prediction report")
        canvas.setFillColor(MUTED)
        canvas.setFont("GC", 7.5)
        canvas.drawString(
            18 * mm, 11 * mm,
            "An early-warning estimate from a machine learning model. It "
            "supports academic advising and is not an examination result.")
        canvas.drawRightString(w - 18 * mm, 11 * mm,
                               f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    story = []

    # --- Student -----------------------------------------------------------
    info = [
        ("Student", row["student_name"]),
        ("Matric or reference number", row["student_ref"] or "Not given"),
        ("Institution type", row["institution_type"] or "Not given"),
        ("State", f"{row['state']} ({row['zone']})" if row["state"]
         else "Not given"),
        ("Predicted on", _when(row["created_at"])),
        ("Report prepared for", prepared_for),
    ]
    cells = []
    for i in range(0, len(info), 2):
        line = []
        for label, value in info[i:i + 2]:
            line += [_p(label, st["label"]), _p(value, st["cellb"])]
        cells.append(line)
    t = Table(cells, colWidths=[width * 0.17, width * 0.33] * 2)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story += [t, Spacer(1, 8)]

    # --- Result ------------------------------------------------------------
    risk_text = {
        "At risk": "This student is likely to struggle without support. "
                   "Start with the recommendations below.",
        "Watch": "A steady result is likely, with room to move up a class. "
                 "Small changes could make the difference.",
        "On track": "This student is on course for a strong result. Keep the "
                    "current habits in place.",
    }.get(row["risk"], "")
    left = [
        _p(f"Predicted class ({scale_name} grading)", st["label"]),
        Spacer(1, 3),
        _p(predicted, st["big"]),
        Spacer(1, 6),
        Paragraph(f"There is a <b>{row['confidence'] * 100:.0f}%</b> chance "
                  f"of finishing with this class. Risk level: "
                  f"<b>{escape(row['risk'])}</b>.", st["body"]),
        Spacer(1, 4),
        _p(risk_text, st["body"]),
    ]
    bar_w = width * 0.30
    chance = [[_p("Chance of each class", st["label"]), "", ""]]
    for i in range(len(names) - 1, -1, -1):
        bar = Table([["", ""]],
                    colWidths=[max(0.5, bar_w * proba[i]),
                               max(0.5, bar_w * (1 - proba[i]))],
                    rowHeights=[7])
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), BAND[i]),
            ("BACKGROUND", (1, 0), (1, 0), LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        style = st["cellb"] if i == idx else st["cell"]
        chance.append([_p(names[i], style), bar,
                       _p(f"{proba[i] * 100:.0f}%", style)])
    ct = Table(chance, colWidths=[width * 0.17, bar_w + 4, width * 0.07])
    ct.setStyle(TableStyle([
        ("SPAN", (0, 0), (-1, 0)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    box = Table([[left, ct]], colWidths=[width * 0.44, width * 0.56])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PAPER),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(box)

    # --- Recommendations ---------------------------------------------------
    story.append(_p("Recommendations to improve", st["h2"]))
    if inputs.get("Learning_Disabilities") == "Yes":
        story += [_p(advice.SUPPORT_NOTE, st["body"]), Spacer(1, 6)]
    levers = (explanation or {}).get("levers") or []
    if explanation is None:
        story.append(_p("Recommendations are unavailable because no trained "
                        "model is loaded.", st["body"]))
    elif not levers:
        story.append(_p("No change needed. The habits this student can "
                        "change are already strong. Keep them in place.",
                        st["body"]))
    else:
        story.append(_p("Start with number 1. Each step builds on the one "
                        "before it.", st["small"]))
        story.append(Spacer(1, 4))
        for n, lever in enumerate(levers, start=1):
            now = recommend.fmt_value(lever["name"], lever["current"])
            aim = recommend.fmt_value(lever["name"], lever["value"])
            parts = [
                Paragraph(f"{n}. {escape(lever['advice']['title'])}",
                          st["rec_title"]),
                Paragraph(f"{escape(lever['label'])}: now <b>{escape(now)}"
                          f"</b>, aim for <b>{escape(aim)}</b>", st["body"]),
                _p(lever["advice"]["body"], st["body"]),
            ]
            alone = lever.get("alone_index")
            if n == 1 and alone is not None and alone != idx:
                parts.append(Paragraph(
                    f"On its own, this moves the prediction to "
                    f"<b>{escape(names[alone])}</b>.", st["body"]))
            block = Table([[parts]], colWidths=[width])
            block.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            story += [KeepTogether(block), Spacer(1, 5)]
        combined = explanation.get("combined")
        if combined and combined["index"] > idx:
            story.append(Paragraph(
                f"If all {len(levers)} recommendations are followed, the "
                f"model expects <b>{escape(names[combined['index']])}</b> "
                f"instead of <b>{escape(predicted)}</b>.", st["body"]))
        story.append(Spacer(1, 3))
        story.append(_p("Recommendations come from asking the model \"what if "
                        "this changed?\". They show what the model responds "
                        "to and do not guarantee an outcome.", st["small"]))

    # --- Drivers -----------------------------------------------------------
    drivers = (explanation or {}).get("drivers") or []
    if drivers:
        story.append(_p("What is helping and what is holding back",
                        st["h2"]))
        helping = [d for d in drivers if d["effect"] > 0]
        holding = [d for d in drivers if d["effect"] < 0]

        def items(rows, empty):
            if not rows:
                return [_p(empty, st["small"])]
            return [Paragraph(
                f"<b>{escape(d['label'])}</b>: "
                f"{escape(recommend.fmt_value(d['name'], d['current']))} "
                f"(typical student: "
                f"{escape(recommend.fmt_value(d['name'], d['value']))})",
                st["cell"]) for d in rows]

        dt = Table([[_p("Helping", st["cellb"]),
                     _p("Holding back", st["cellb"])],
                    [items(helping, "Nothing stands out."),
                     items(holding, "Nothing stands out.")]],
                   colWidths=[width / 2, width / 2])
        dt.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ]))
        story.append(dt)

    # --- Details entered ---------------------------------------------------
    story.append(_p("Details entered", st["h2"]))
    rows = []
    for key, title, _ in schema.GROUPS:
        rows.append([_p(title, st["label"]), "", "", ""])
        group = [f for f in schema.FEATURES if f["group"] == key]
        for i in range(0, len(group), 2):
            line = []
            for f in group[i:i + 2]:
                line += [_p(f["label"], st["cell"]),
                         _p(recommend.fmt_value(f["name"], inputs.get(
                             f["name"], "")), st["cellb"])]
            while len(line) < 4:
                line.append("")
            rows.append(line)
    dt = Table(rows, colWidths=[width * 0.3, width * 0.2] * 2)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 0),
             ("TOPPADDING", (0, 0), (-1, -1), 2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    for r, line in enumerate(rows):
        if line[1] == "" and line[2] == "":
            style += [("SPAN", (0, r), (-1, r)), ("TOPPADDING", (0, r),
                                                   (-1, r), 7)]
    dt.setStyle(TableStyle(style))
    story.append(dt)

    # --- About -------------------------------------------------------------
    story.append(Spacer(1, 10))
    acc = ""
    if metrics:
        acc = (f" On {metrics['dataset']['rows_test']:,} students it had "
               f"not seen, the model picked the exact class "
               f"{metrics['results']['ensemble']['accuracy'] * 100:.1f}% of "
               f"the time.")
    story.append(_p(
        "Prediction by a voting ensemble of Random Forest and Gradient "
        f"Boosting, model version {row['model_version']}.{acc}",
        st["small"]))

    doc.build(story, onFirstPage=page, onLaterPages=page)
    return buf.getvalue()

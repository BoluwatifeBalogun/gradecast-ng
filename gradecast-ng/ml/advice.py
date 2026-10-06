"""Plain-language guidance attached to each attribute a student can change.

The wording is written for Nigerian tertiary institutions (universities,
polytechnics and colleges of education).
"""

ADVICE = {
    "Hours_Studied": (
        "Build a weekly study timetable",
        "Block out fixed self-study hours around the lecture timetable and "
        "protect them. Two focused hours a day on weekdays is a realistic "
        "first target."),
    "Attendance": (
        "Raise class attendance",
        "Attendance is the strongest signal the model uses. Aim to be in "
        "every lecture and practical. If transport or work is the obstacle, "
        "raise it with the level adviser early. Many institutions also "
        "set a minimum attendance before a student can sit an exam."),
    "Tutoring_Sessions": (
        "Join tutorials or a study group",
        "Add departmental tutorials, peer tutoring or a small study group "
        "each week, especially for courses with heavy calculations."),
    "Motivation_Level": (
        "Rebuild motivation with a mentor",
        "Pair the student with a mentor or academic adviser, set short "
        "targets for each test, and review progress every few weeks."),
    "Extracurricular_Activities": (
        "Take up one structured activity",
        "A club, sport or departmental association adds routine and a "
        "support network. One commitment is enough."),
    "Sleep_Hours": (
        "Keep a steady sleep routine",
        "Aim for about seven hours a night. All-night reading before tests "
        "tends to cost more than it gains."),
    "Physical_Activity": (
        "Add regular physical activity",
        "A few hours of exercise a week supports concentration. Campus "
        "sports or a brisk daily walk both count."),
    "Parental_Involvement": (
        "Keep parents or guardians in the loop",
        "Share the semester calendar and results with a parent, guardian "
        "or sponsor so that someone outside school is following progress."),
    "Internet_Access": (
        "Secure reliable internet for study",
        "Use the campus Wi-Fi, e-library or ICT centre, and download "
        "course materials for offline use when data is limited."),
    "Distance_from_Home": (
        "Cut down travel time",
        "Long commutes eat into study time. Consider hostel accommodation, "
        "a room closer to campus, or studying on campus before heading "
        "home."),
    "Access_to_Resources": (
        "Improve access to learning materials",
        "Register with the library, use departmental laboratories and "
        "past questions, and share textbooks within a study group."),
    "Teacher_Quality": (
        "Get more from teaching contact",
        "Use lecturers' consultation hours, ask for recommended texts, and "
        "supplement difficult courses with open courseware."),
    "Peer_Influence": (
        "Study with focused peers",
        "Spend study time with classmates who take coursework seriously. "
        "A change of study group often changes study habits."),
}

SUPPORT_NOTE = (
    "A learning disability is recorded. Refer the student to the "
    "institution's counselling or student support unit for assessment and "
    "reasonable adjustments such as extra time.")


def for_feature(name):
    title, body = ADVICE.get(name, ("", ""))
    return {"title": title, "body": body}

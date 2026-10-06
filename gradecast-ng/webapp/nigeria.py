"""Nigerian reference lists used to tag prediction records.

These tags describe where a record comes from. They are not model inputs:
the classifier only sees the 19 attributes in ml/schema.py.
"""

ZONES = {
    "North Central": ["Benue", "Kogi", "Kwara", "Nasarawa", "Niger",
                      "Plateau", "FCT Abuja"],
    "North East": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
    "North West": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto",
                   "Zamfara"],
    "South East": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
    "South South": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo",
                    "Rivers"],
    "South West": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"],
}

STATE_ZONE = {state: zone for zone, states in ZONES.items()
              for state in states}
STATES = sorted(STATE_ZONE)

INSTITUTION_TYPES = [
    "Federal University", "State University", "Private University",
    "Federal Polytechnic", "State Polytechnic", "Private Polytechnic",
    "College of Education", "Monotechnic or Specialised College",
]

SAMPLE_FIRST = ["Adaeze", "Chinedu", "Ngozi", "Emeka", "Ifeoma", "Obinna",
                "Tunde", "Folake", "Segun", "Yetunde", "Kehinde", "Bolanle",
                "Aisha", "Musa", "Fatima", "Ibrahim", "Zainab", "Sani",
                "Halima", "Yusuf", "Efe", "Oghenekaro", "Ebiere", "Tamuno",
                "Idara", "Ekaette", "Terver", "Dooshima", "Blessing",
                "Emmanuel", "Grace", "Samuel", "Amina", "Uche", "Damilola",
                "Kelechi"]
SAMPLE_LAST = ["Okafor", "Adeyemi", "Bello", "Eze", "Abubakar", "Okonkwo",
               "Balogun", "Mohammed", "Nwosu", "Ogunleye", "Danjuma",
               "Effiong", "Ibekwe", "Lawal", "Akpan", "Onyeka", "Usman",
               "Afolabi", "Etim", "Yakubu", "Odukoya", "Briggs", "Tsav",
               "Garba", "Ojo", "Umeh"]

"""Item selection for Twin-2K-500 and GSS 2024 (H1).

STATUS: PROPOSED, awaiting approval (see docs/DATA.md). Do not build on this list until approved.

Each spec maps a clean item code to its raw source column and says how raw values become an
option index 0..k-1. Option labels come from the data files themselves: Twin from `persona_json`
(QID287/288/289/291/196, verified to use the same 1..k coding as the CSVs), GSS from the .dta
value labels.
"""
from dataclasses import dataclass

# Selection rules (handover H1)
MAX_MISSING = 0.10
MAX_TOP_SHARE = 0.85

# GSS 2024 uses split ballots. Ballot 1 is the only ballot carrying the ISSP Digital Society
# module (technology / AI / 0-10 trust scales) at <10% missingness, so the GSS target set is
# restricted to ballot == 1.
GSS_BALLOT = 1


@dataclass(frozen=True)
class ItemSpec:
    code: str          # clean item code, used as the answers column
    source: str        # raw column name in the file (Twin names keep their trailing spaces)
    theme: str
    kind: str          # "ordinal" | "binary"
    sensitive: bool = False
    bins: tuple[int, ...] | None = None  # 0-100 numeric answers: right-open bin edges


# ---------- Twin-2K-500 ----------
_POLICY = [
    ("carbon_tax", 1), ("clean_energy_lowincome", 2), ("carbon_free_2035", 3),
    ("medicare_for_all", 4), ("public_option", 5), ("citizenship_path", 6),
    ("paid_family_leave", 7), ("wealth_tax", 8), ("more_deportations", 9),
    ("senior_vouchers", 10),
]
# "False cons. others" has no _8/_9 columns: _10/_11/_12 carry policies 8/9/10 (checked on row-0 wording).
_OTHERS_SUFFIX = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 10, 9: 11, 10: 12}
PERCENT_BINS = (0, 20, 40, 60, 80, 101)
PERCENT_BIN_LABELS = ["0-19%", "20-39%", "40-59%", "60-79%", "80-100%"]
_TECH = [("bicycles", 1), ("alcohol", 2), ("chemical_plants", 3), ("pesticides", 4)]

TWIN_ITEMS: tuple[ItemSpec, ...] = tuple(
    [ItemSpec(f"support_{n}", f"False Cons. self _{i}", "policy_views", "ordinal", sensitive=True)
     for n, i in _POLICY]
    + [ItemSpec(f"est_public_{n}", f"False cons. others _{_OTHERS_SUFFIX[i]}", "perceived_consensus",
                "ordinal", sensitive=True, bins=PERCENT_BINS) for n, i in _POLICY]
    + [ItemSpec(f"benefit_{n}", f"nonseparabilty bene _{i}", "risk_benefit", "ordinal") for n, i in _TECH]
    + [ItemSpec(f"risk_{n}", f"nonseparability ris _{i}", "risk_benefit", "ordinal") for n, i in _TECH]
    + [ItemSpec("omission_vaccine", "Omission bias ", "reasoning", "ordinal"),
       ItemSpec("denominator_tray", "Denominator neglect ", "reasoning", "binary")]
)

# Which persona_json question holds the option labels for each Twin source-column prefix.
TWIN_OPTION_QID = {
    "False Cons. self _": "QID287",
    "nonseparabilty bene _": "QID288",
    "nonseparability ris _": "QID289",
    "Omission bias ": "QID291",
    "Denominator neglect ": "QID196",
}

# ---------- GSS 2024 (ballot 1) ----------
# codes: substantive answer codes in option order. Reserved codes (d/i/n/s/...) load as NaN.
GSS_ITEMS: tuple[tuple[ItemSpec, tuple[int, ...]], ...] = (
    # trust / institutions
    (ItemSpec("trppl", "trppl", "trust_institutions", "ordinal"), tuple(range(11))),
    (ItemSpec("trcourts", "trcourts", "trust_institutions", "ordinal"), tuple(range(11))),
    (ItemSpec("trcong", "trcong", "trust_institutions", "ordinal"), tuple(range(11))),
    (ItemSpec("intrust", "intrust", "trust_institutions", "ordinal"), tuple(range(11))),
    (ItemSpec("satdemoc", "satdemoc", "trust_institutions", "ordinal"), (1, 2, 3, 4)),
    (ItemSpec("polint", "polint", "trust_institutions", "ordinal"), (1, 2, 3, 4, 5)),
    # money / economy
    (ItemSpec("satfin", "satfin", "money_economy", "ordinal"), (1, 2, 3)),
    (ItemSpec("finalter", "finalter", "money_economy", "ordinal"), (1, 3, 2)),  # better, same, worse
    (ItemSpec("finrela", "finrela", "money_economy", "ordinal"), (1, 2, 3, 4, 5)),
    (ItemSpec("class", "class", "money_economy", "ordinal"), (1, 2, 3, 4)),
    (ItemSpec("econstat", "econstat", "money_economy", "ordinal"), (1, 2, 3, 4, 5)),
    (ItemSpec("tax", "tax", "money_economy", "ordinal"), (1, 2, 3)),
    # work / life
    (ItemSpec("happy", "happy", "work_life", "ordinal"), (1, 2, 3)),
    (ItemSpec("life", "life", "work_life", "ordinal"), (1, 2, 3)),
    (ItemSpec("socfrend", "socfrend", "work_life", "ordinal"), (1, 2, 3, 4, 5, 6, 7)),
    (ItemSpec("socrel", "socrel", "work_life", "ordinal"), (1, 2, 3, 4, 5, 6, 7)),
    (ItemSpec("workless", "workless", "work_life", "binary"), (1, 2)),
    (ItemSpec("meovrwrk", "meovrwrk", "work_life", "ordinal"), (1, 2, 3, 4, 5)),
    # technology / science
    (ItemSpec("techesy", "techesy", "tech_science", "ordinal"), (1, 2, 3, 4, 5)),
    (ItemSpec("harmgood1", "harmgood1", "tech_science", "ordinal"), (1, 2, 3, 4, 5)),
    (ItemSpec("aiworry", "aiworry", "tech_science", "ordinal"), (1, 2, 3, 4, 5)),
    (ItemSpec("aidrive", "aidrive", "tech_science", "ordinal"), tuple(range(11))),
    (ItemSpec("aimed", "aimed", "tech_science", "ordinal"), tuple(range(11))),
    (ItemSpec("natsci", "natsci", "tech_science", "ordinal"), (1, 2, 3)),
)

# Wording from the GSS 2024 Codebook R3a. For ISSP Digital Society items (and satdemoc) the
# codebook prints only a short label; those are in LABEL_ONLY and must be checked against the
# ISSP 2024 questionnaire before they are shown to people.
LABEL_ONLY = frozenset({"trppl", "trcourts", "trcong", "intrust", "satdemoc", "techesy",
                        "harmgood1", "aiworry", "aidrive", "aimed"})
GSS_WORDING = {
    "trppl": "Trust in others, 10 point scale (0 = you can't be too careful, 10 = most people can be trusted).",
    "trcourts": "How much trust do you have in American courts? (0 = no trust, 10 = complete trust)",
    "trcong": "How much trust do you have in Congress? (0 = no trust at all, 10 = complete trust)",
    "intrust": "How much do you trust people you have only met online? (0 = no trust at all, 10 = complete trust)",
    "satdemoc": "How satisfied are you with the way democracy works in America?",
    "polint": "How interested would you say you personally are in politics?",
    "satfin": ("We are interested in how people are getting along financially these days. So far as you and "
               "your family are concerned, would you say that you are pretty well satisfied with your present "
               "financial situation, more or less satisfied, or not satisfied at all?"),
    "finalter": ("During the last few years, has your financial situation been getting better, worse, or has "
                 "it stayed the same?"),
    "finrela": ("Compared with American families in general, would you say your family income is far below "
                "average, below average, average, above average, or far above average?"),
    "class": ("If you were asked to use one of four names for your social class, which would you say you "
              "belong in: the lower class, the working class, the middle class, or the upper class?"),
    "econstat": "What do you think about the state of the economy these days in the United States?",
    "tax": ("Do you consider the amount of federal income tax which you have to pay as too high, about "
            "right, or too low?"),
    "happy": ("Taken all together, how would you say things are these days--would you say that you are very "
              "happy, pretty happy, or not too happy?"),
    "life": "In general, do you find life exciting, pretty routine, or dull?",
    "socfrend": "How often do you spend a social evening with friends who live outside the neighborhood?",
    "socrel": "How often do you spend a social evening with relatives?",
    "workless": ("In your present or most recent job, have you ever cut back on your work because of your "
                 "responsibilities to members of your family?"),
    "meovrwrk": "Family life often suffers because men concentrate too much on their work.",
    "techesy": "Does technology make our lives easier?",
    "harmgood1": "Does technology do more harm than good?",
    "aiworry": "How worried are you that AI will take over many jobs done by humans?",
    "aidrive": "How comfortable would you be with a driverless car? (0 = totally uncomfortable, 10 = totally comfortable)",
    "aimed": ("How comfortable would you be with a medical operation performed by a robot? (0 = totally "
              "uncomfortable, 10 = totally comfortable)"),
    "natsci": ("We are faced with many problems in this country. Are we spending too much, too little, or "
               "about the right amount on supporting scientific research?"),
}

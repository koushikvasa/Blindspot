"""Item selection for Twin-2K-500 and GSS 2024 (H1). APPROVED 2026-11-11 (see docs/DATA.md).

Each spec maps a clean item code to its raw source column and says how raw values become an
option index 0..k-1. Option labels come from the data files themselves: Twin from `persona_json`
(QID287/288/289/291/196, verified to use the same 1..k coding as the CSVs), GSS from the .dta
value labels.
"""
from dataclasses import dataclass

# Selection rules (handover H1)
MAX_MISSING = 0.10
MAX_TOP_SHARE = 0.85

# GSS target set: main 2024 cross-section (sample == 13; excludes the 677-case AmeriSpeak
# oversample, sample == 14, which has no wtssps weight) AND ballot 1, the only ballot carrying
# the ISSP Digital Society module at <10% missingness.
GSS_SAMPLE = 13
GSS_BALLOT = 1
GSS_WEIGHT = "wtssps"


@dataclass(frozen=True)
class ItemSpec:
    code: str          # clean item code, used as the answers column
    source: str        # raw column name in the file (Twin names keep their trailing spaces)
    theme: str         # also used as the item family
    kind: str          # "ordinal" | "binary"
    sensitive: bool = False
    bins: tuple[int, ...] | None = None  # 0-100 numeric answers: right-open bin edges
    verified: bool = True                # wording checked against source documentation


# ---------- Twin-2K-500: 30 items in 4 families ----------
TWIN_FAMILIES = ("policy_self", "perception", "risk_benefit", "decisions")

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
    [ItemSpec(f"support_{n}", f"False Cons. self _{i}", "policy_self", "ordinal", sensitive=True)
     for n, i in _POLICY]
    + [ItemSpec(f"est_public_{n}", f"False cons. others _{_OTHERS_SUFFIX[i]}", "perception",
                "ordinal", sensitive=True, bins=PERCENT_BINS) for n, i in _POLICY]
    + [ItemSpec(f"benefit_{n}", f"nonseparabilty bene _{i}", "risk_benefit", "ordinal") for n, i in _TECH]
    + [ItemSpec(f"risk_{n}", f"nonseparability ris _{i}", "risk_benefit", "ordinal") for n, i in _TECH]
    + [ItemSpec("omission_vaccine", "Omission bias ", "decisions", "ordinal"),
       ItemSpec("denominator_tray", "Denominator neglect ", "decisions", "binary")]
)

# Which persona_json question holds the option labels for each Twin source-column prefix.
TWIN_OPTION_QID = {
    "False Cons. self _": "QID287",
    "nonseparabilty bene _": "QID288",
    "nonseparability ris _": "QID289",
    "Omission bias ": "QID291",
    "Denominator neglect ": "QID196",
}

# ---------- GSS 2024: 24 items, 4 themes x 6 ----------
GSS_FAMILIES = ("trust_institutions", "money_economy", "work_life", "tech_science")


def _g(code: str, theme: str, kind: str = "ordinal", sensitive: bool = False, verified: bool = True) -> ItemSpec:
    return ItemSpec(code, code, theme, kind, sensitive=sensitive, verified=verified)


_T, _M, _W, _S = GSS_FAMILIES
_ZERO_TEN = tuple(range(11))
# (spec, substantive codes in option order). Reserved codes (d/i/n/s/...) load as NaN.
GSS_ITEMS: tuple[tuple[ItemSpec, tuple[int, ...]], ...] = (
    (_g("trppl", _T), _ZERO_TEN),
    (_g("trcourts", _T), _ZERO_TEN),
    (_g("trcong", _T, sensitive=True), _ZERO_TEN),
    (_g("intcnfrm", _T), (1, 2, 3, 4, 5)),
    (_g("satdemoc", _T, sensitive=True), (1, 2, 3, 4)),
    (_g("polint", _T, sensitive=True), (1, 2, 3, 4, 5)),
    (_g("satfin", _M), (1, 2, 3)),
    (_g("finalter", _M), (1, 3, 2)),  # better, stayed same, worse (reordered to be ordinal)
    (_g("finrela", _M), (1, 2, 3, 4, 5)),
    (_g("class", _M), (1, 2, 3, 4)),
    (_g("econstat", _M), (1, 2, 3, 4, 5)),
    (_g("tax", _M, sensitive=True), (1, 2, 3)),
    (_g("happy", _W), (1, 2, 3)),
    (_g("life", _W), (1, 2, 3)),
    (_g("socfrend", _W), (1, 2, 3, 4, 5, 6, 7)),
    (_g("socrel", _W), (1, 2, 3, 4, 5, 6, 7)),
    (_g("workless", _W, kind="binary"), (1, 2)),
    (_g("meovrwrk", _W), (1, 2, 3, 4, 5)),
    (_g("techesy", _S), (1, 2, 3, 4, 5)),
    (_g("harmgood1", _S), (1, 2, 3, 4, 5)),
    (_g("aiworry", _S), (1, 2, 3, 4, 5)),
    (_g("aidrive", _S), _ZERO_TEN),
    (_g("aimed", _S), _ZERO_TEN),
    (_g("natsci", _S), (1, 2, 3)),
)

# Wording, all verified:
# - core items: question text printed in the GSS 2024 Codebook R3a (bracketed CAPI fills resolved);
# - the 10 ISSP / label-only items (trppl ... aimed): verbatim from the official GSS 2024 Ballot 1
#   questionnaire, https://gss.norc.org/content/dam/gss/get-documentation/pdf/quex/GSS2024_Ballot1_English.pdf
#   (MDD export "GSS2024M_BALLOT1.MDD", dated 12/11/2024). Battery stems are joined to the item text.
#   satdemoc keeps its response-order fill marker: the questionnaire does not print the fill text.
GSS_WORDING_SOURCE_URL = "https://gss.norc.org/content/dam/gss/get-documentation/pdf/quex/GSS2024_Ballot1_English.pdf"
GSS_WORDING = {
    "trppl": ("Generally speaking, would you say that most people can be trusted or that you can't be too careful "
              "in dealing with people? Where would you place yourself on this scale, where 0 means you can't be "
              "too careful, and 10 means most people can be trusted."),
    "trcourts": ("On a scale of 0 to 10, how much do you personally trust each of the following institutions? "
                 "0 means you do not trust an institution at all, and 10 means you trust it completely. "
                 "America's courts."),
    "trcong": ("On a scale of 0 to 10, how much do you personally trust each of the following institutions? "
               "0 means you do not trust an institution at all, and 10 means you trust it completely. "
               "The U.S. Congress."),
    "intcnfrm": ("When looking for news or information about political matters online, how often, if at all, do "
                 "you: Try to confirm information you find by searching online for another source?"),
    "satdemoc": ("On the whole, are you {response to SATDEMOC_fill} with the way democracy works in the "
                 "United States?"),
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
    "socfrend": ("Please select the answer that comes closest to how often you do the following: Spend a "
                 "social evening with friends who live outside the neighborhood?"),
    "socrel": "How often do you spend a social evening with relatives?",
    "workless": ("In your present or most recent job, have you ever done any of the following because of your "
                 "responsibilities to members of your family? Cut back on your work?"),
    "meovrwrk": "Family life often suffers because men concentrate too much on their work.",
    "techesy": ("Here are some statements about technology. Technology includes all kinds of machinery, "
                "equipment, digital devices, and software. How much do you agree or disagree with each of these "
                "statements? Technology makes our lives easier."),
    "harmgood1": ("Here are some statements about technology. Technology includes all kinds of machinery, "
                  "equipment, digital devices, and software. How much do you agree or disagree with each of "
                  "these statements? Overall, technology does more harm than good."),
    "aiworry": ("The next questions deal with robots and artificial intelligence. 'Artificial intelligence' is "
                "the ability of a robot or computer to perform tasks commonly associated with humans. Overall, how "
                "worried, if at all, are you that in the next 10 years machines, computer programs, and Artificial "
                "Intelligence (AI) will take over many of the jobs done by humans?"),
    "aidrive": ("Various human actions can be done by robots or Artificial Intelligence (AI). For each of the "
                "following, please indicate, using a scale from 0 to 10, how you would personally feel about it. "
                "On this scale, '0' means that you would feel \"totally uncomfortable\" and '10' means that you "
                "would feel \"totally comfortable\" with this situation. Being driven in a driverless car."),
    "aimed": ("Various human actions can be done by robots or Artificial Intelligence (AI). For each of the "
              "following, please indicate, using a scale from 0 to 10, how you would personally feel about it. "
              "On this scale, '0' means that you would feel \"totally uncomfortable\" and '10' means that you "
              "would feel \"totally comfortable\" with this situation. Having a medical operation performed on "
              "you by a robot."),
    "natsci": ("Are we spending too much, too little, or about the right amount on supporting scientific "
               "research?"),
}

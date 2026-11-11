# Data notes (H1)

Everything below was found by reading the files, not assumed. Raw data lives in `data/raw/` (git-ignored).
Item lists **approved 2026-11-11** (decisions recorded in §3). Splits in §4. Run `python -m blindspot.data.inspect` for the live numbers.

## 1. Twin-2K-500

License **CC BY 4.0**. Cite: Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., Chen, H. (2025).
*Twin-2K-500: A Dataset for Building Digital Twins of 2,000 People.* Hugging Face. arXiv:2505.17479.

### Files
| File (under `data/raw/twin2k/LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human/`) | Our source name | Rows (pandas / people) |
| --- | --- | --- |
| `responses_wave1_3_formatted.csv` | `twin2k:real` | 2,059 / 2,058 |
| `responses_llm_imputed_formatted.csv` | `twin2k:gpt41mini_twin` | 2,059 / 2,058 |
| `responses_wave4_formatted.csv` | `twin2k:retest` | 2,059 / 2,058 |
| `data/raw/twin2k_parquet/full_persona/data/0000–0006.parquet` | demographics + option labels | 7 × 294 = 2,058 |

- All three CSVs: UTF-8, the same 168 columns, ID column `TWIN_ID` (unique; the same 2,058 IDs in all three). `real` and `gpt41mini_twin` have the same row order; `retest` does not, so the loader reindexes by ID.
- **Row 0 is question wording**, not a person (`TWIN_ID` cell = `"TWIN_ID"`). It is identical in all three files. Used as `Item.wording` (whitespace collapsed), then dropped.
- `WAVE` = `wave1` in `real` and `gpt41mini_twin`, `wave4` in `retest`.
- Missing values are empty cells; no special codes. Values are `"3"` in some columns and `"3.0"` in others; both parse to the same number. All three files use the same numeric coding (no text labels anywhere), so one mapping serves every file.
- **Completely empty in all three files** (dropped): all metadata columns (`StartDate … RecordedDate`, `STUDY_ID`, `SESSION_ID`, `tasktime`, `worktimeArray`, `offTask`, `onTask`, `totalOffTask`, `totalOnTask`, `perPagePT`, `pageNo`, `randDollars*`), `attention_1/2`, `Q281/Q282/Q284` (comprehension checks), `Q259`, `Q259_3_TEXT`, **and the named reasoning columns** `base_rate, outcome_bias, sunk_cost, Allais, disease, linda, african, redwood, absolute, Myside, less_is_more, Thaler, matching`. The reasoning answers are in `Q156–Q203` instead.

### Persona parquet (`full_persona`)
Columns: `pid` (string; same set as `TWIN_ID`), `persona_text`, `persona_summary`, `persona_json`. `persona_json` is a list of survey blocks → `Questions[{QuestionID, QuestionText, QuestionType, Options | Rows + Columns, Answers{SelectedByPosition, SelectedText}}]`.

- **Demographics** come from the `Demographics` block via `Answers.SelectedText`: QID11 region (Northeast/Midwest/South/West/Pacific; state list stripped), QID12 sex → `gender`, QID13 `age_band` (18-29/30-49/50-64/65+), QID14 `education` (6 levels), QID15 `race` (5), QID21 `income_band` (<$30k / $30-50k / $50-75k / $75-100k / $100k+). Also present but not extracted: citizenship, marital status, religion, attendance, party, ideology, household size, employment. Cached to `data/processed/twin2k_demographics.csv`.
- **Option labels** come from the same JSON (QID287 policy Likert, QID288 benefit, QID289 risk, QID291 omission, QID196 denominator). CSV code `c` = option position `c` (1-based). Verified: persona `SelectedByPosition` equals the `retest` CSV code for **100% of 2,058 people on all 60 overlapping items**.
- ⚠️ **Leakage channel.** The dataset card says: *"For questions that appear in both waves 1-3 and wave 4, the wave 4 responses are used."* So `persona_text` / `persona_json` / `persona_summary` contain every person's **wave-4 (retest) answers to the scored items**, and they agree with `real` on only about 40–85% of answers. The loaders read **only** the demographics block and option labels. Persona text must never enter a generator prompt, for any split.

### Question families (wording from row 0; missingness in `real`; the same in all three files)
| Family | Columns | Scale | Missing |
| --- | --- | --- | --- |
| Policy support ("Would you support or oppose…") | `False Cons. self _1.._10` | 1–5, strongly oppose → strongly support | 0% |
| Estimated public support (0–100%) | `False cons. others _1.._7, _10, _11, _12` (no `_8/_9`; `_10–_12` = policies 8–10) | integer 0–100, slider | 0% |
| Benefit of a technology/product | `nonseparabilty bene _1.._4` (bicycles, alcoholic beverages, chemical plants, pesticides) | 1–7 | 0% |
| Risk of the same | `nonseparability ris _1.._4` | 1–7 | 0% |
| Omission bias (flu vaccine) | `Omission bias ` (trailing space) | 1–4 | 0% |
| Denominator neglect (marble trays) | `Denominator neglect ` (trailing space) | 1–2 | 0% |
| **Q295 block** = grocery purchase at a price ("Would you or would you not purchase the product?") | `1_Q295 … 40_Q295` | 1 yes / 2 no | 0% |
| Base rate (70 vs 30 engineers) | `Q156_1` / `Form A _1` | 0–100 | ~50% each |
| Disease framing (gain/loss) | `Q157` / `Q158` | 1–6 | ~50% each |
| Linda conjunction | `Q159_1-3` / `Q160_1-3` | 1–6 | 50% |
| Outcome bias (heart surgery) | `Q161` / `Q162` | 1–7 | ~50% |
| Anchoring: African countries, redwood | `Q163–Q166`, `Q167–Q170` | yes/no + number | ~50% |
| Less-is-more / proportion dominance | `Q171–Q179` | 1–5 / 1–6 | 64–68% |
| Sunk cost, absolute vs relative | `Q181–Q184` | number / binary | ~50% |
| WTA/WTP (Thaler) | `Q189–Q191` | 1–10 | 65–67% |
| Allais | `Q192`, `Q193` | binary | ~50% |
| Myside (Ford/German car) | `Q194` / `Q195` | 1–6 | ~50% |
| Probability matching | `Q198_1-10` / `Q203_1-6` | binary | 50% |

- The reasoning tasks are **between-subjects**: each person saw one of two versions, so every `Q156–Q203` column is about 50% missing (or more). The missingness pattern is identical across the three files.
- **The Q295 block is not 40 items.** The product and price are filled in per person: loop position *k* shows ~427 different product/price texts across people (e.g. `QID9_1` = Kool-Aid Jammers for 56 people, Great Value milk at $4.70 for 12, …). `k_Q295` is not comparable across people, so the block is excluded.

## 2. GSS 2024

Source: NORC, General Social Survey 2024, Release 3a (`GSS2024.dta`, `GSS 2024 Codebook R3a.pdf`). Cite: Davern, M., Bautista, R., Freese, J., Herd, P., Morgan, S. L. *General Social Survey 2024*. NORC at the University of Chicago. (Author list from memory: check against NORC's citation page before publishing.)

- `GSS2024.dta`: **3,986 rows × 989 columns**, ID `id` (unique), `year` = 2024 for all rows.
- **Why 3,986 and not the codebook's 3,309.** `sample` (*Sampling frame and method*) has two values: 13 = "2020 sample frame, full probability" (**3,309 rows**, the main cross-section and the codebook's case base) and 14 = "2024 AmeriSpeak frame, full probability black/hispanic/aanhpi oversample" (**677 rows**, the same as `amerstatus == 1`). The oversample is all web mode and has **no `wtssps`/`wtssnrps` weight** (100% missing); only `wtssnrps_as` covers it. The codebook methodology section describes it as an oversample of Black, Hispanic and AANHPI respondents from NORC's AmeriSpeak panel. `phase` (two-phase subsampling, 1 = initial, 2 = subsampled) applies only to sample 13; `oversamp` is constant. Because the oversample is a different sample with a different frame and no main weight, **the target set is the main cross-section**: `sample == 13`.
- **Final GSS target set: `sample == 13 & ballot == 1` → n = 1,116.** The `wtssps` weight is kept as `demographics["weight"]` (non-missing for all 1,116) for later use, not for coverage.
- Loaded with `pyreadstat.read_dta`. 940 columns come back as `object` (ints mixed with NaN), so we use `pd.to_numeric`. Reserved codes are Stata extended missing values (`d` don't know / can't choose, `i` IAP = not asked, `n` no answer, `s` skipped on web, `r` refused, `x`/`y`/`z`, …) and load as **NaN**.
- **Split ballots.** `ballot` ∈ {1: 1,341; 2: 1,294; 3: 1,351} over all rows (main sample: 1,116 / 1,067 / 1,126). Most attitude items are asked on only 1–2 ballots, so only about 105 variables have <10% missing on the full sample, and most of those are demographics or admin. **Proposal: the GSS target set is ballot 1 (n = 1,341).** It is the only ballot that carries the ISSP *Digital Society* module (technology, AI, 0–10 trust scales) at <10% missing, alongside the full-sample money and life items. Not available on ballot 1: the `con*` confidence battery and `trust/fair/helpful` (ballots 2–3).
- Weights (`wtssps`, `wtssnrps`) exist but are not used for coverage (unweighted people-level analysis).
- Demographics (descriptive only): `age_band` from `age` (same bands as Twin; 89 = "89 or older"), `gender` from `sex`, `education` from `degree` (5 GSS levels), `income_band` from `income16` (26 codes → <$30k / $30–50k / $50–75k / $75–110k / $110k+; the top cut is 110k because code 22 spans $90–110k), `region` (4 census regions), `race` (white/black/other). Missing → `unknown`.
- Wording: core items use the question text printed in the codebook, with bracketed CAPI fills resolved. The 10 ISSP/label-only items (`trppl, trcourts, trcong, intcnfrm, satdemoc, techesy, harmgood1, aiworry, aidrive, aimed`) use the verbatim text from the **official GSS 2024 Ballot 1 questionnaire** (https://gss.norc.org/content/dam/gss/get-documentation/pdf/quex/GSS2024_Ballot1_English.pdf, MDD export dated 12/11/2024), with each battery's stem joined to the item; their response labels match the .dta value labels. `satdemoc` keeps its `{response to SATDEMOC_fill}` marker because the questionnaire does not print the fill text (it is sensitive, so it never appears in live mode). All 24 items: `verified=True` (H2 side task). The GSS Data Explorer was unreachable (TLS certificate error).


## 3. Items (approved 2026-11-11)

Rules: missing < 10% in the target set, no option above 85%, ordinal or binary. "Top" = share of the most common option.
`Item.family` groups items so distances can weight each family equally. `Item.live_ok` = `verified and not sensitive`.

Decisions recorded at approval:
1. GSS `intrust` (11.1% missing) replaced by `intcnfrm` (8.8% missing on the final target set).
2. The 10 `est_public_*` items are kept, binned into five 20-point bands, as their own family `perception`, `sensitive=True`.
3. `trcong`, `satdemoc`, `polint`, `tax` marked `sensitive=True`.
4. GSS wording verified: core items from the codebook, the 10 label-only items from the official 2024 Ballot 1 questionnaire (done in H2; see §2).
5. GSS target set restricted to the main cross-section, ballot 1 (n = 1,116).
6. Twin families: `policy_self`, `perception`, `risk_benefit`, `decisions`.

### Twin-2K-500: 30 items, 4 families (0% missing in real, twin and retest)
| Code | Family | Wording (short; full text = CSV row 0) | Scale | Top real / twin / retest | Sensitive |
| --- | --- | --- | --- | --- | --- |
| support_carbon_tax | policy_self | Support/oppose: tax on carbon emissions | 5: strongly oppose → strongly support | 34 / 47 / 38% | yes |
| support_clean_energy_lowincome | policy_self | …40% of clean-energy spending to low-income communities | same | 33 / 42 / 36% | yes |
| support_carbon_free_2035 | policy_self | …federal investment for carbon-free electricity by 2035 | same | 34 / 34 / 34% | yes |
| support_medicare_for_all | policy_self | …Medicare for All | same | 50 / 44 / 48% | yes |
| support_public_option | policy_self | …public option | same | 38 / 44 / 39% | yes |
| support_citizenship_path | policy_self | …path to citizenship for undocumented immigrants | same | 40 / 44 / 39% | yes |
| support_paid_family_leave | policy_self | …required paid family leave | same | 52 / 60 / 52% | yes |
| support_wealth_tax | policy_self | …2% tax on assets above $50M | same | 53 / 37 / 54% | yes |
| support_more_deportations | policy_self | …increasing deportations | same | 27 / 43 / 26% | yes |
| support_senior_vouchers | policy_self | …healthcare vouchers for seniors instead of Medicare | same | 29 / 64 / 32% | yes |
| est_public_carbon_tax | perception | What % of the public supports: carbon tax | 0–100 binned: 0-19 / 20-39 / 40-59 / 60-79 / 80-100% | 37 / 60 / 36% | yes |
| est_public_clean_energy_lowincome | perception | …40% to low-income communities | same | 34 / 51 / 35% | yes |
| est_public_carbon_free_2035 | perception | …carbon-free 2035 | same | 34 / 48 / 35% | yes |
| est_public_medicare_for_all | perception | …Medicare for All | same | 35 / 34 / 32% | yes |
| est_public_public_option | perception | …public option | same | 33 / 42 / 35% | yes |
| est_public_citizenship_path | perception | …path to citizenship | same | 40 / 36 / 39% | yes |
| est_public_paid_family_leave | perception | …paid family leave | same | 35 / 42 / 34% | yes |
| est_public_wealth_tax | perception | …wealth tax | same | 29 / 30 / 31% | yes |
| est_public_more_deportations | perception | …more deportations | same | 36 / 56 / 37% | yes |
| est_public_senior_vouchers | perception | …senior vouchers | same | 35 / 58 / 35% | yes |
| benefit_bicycles | risk_benefit | Rate how beneficial: bicycles | 7: not at all beneficial → extremely beneficial | 39 / 67 / 41% | no |
| benefit_alcohol | risk_benefit | …alcoholic beverages | same | 33 / 77 / 34% | no |
| benefit_chemical_plants | risk_benefit | …chemical plants | same | 21 / 66 / 23% | no |
| benefit_pesticides | risk_benefit | …pesticides | same | 27 / 69 / 28% | no |
| risk_bicycles | risk_benefit | Rate how risky: bicycles | 7: not at all risky → extremely risky | 32 / **91** / 32% | no |
| risk_alcohol | risk_benefit | …alcoholic beverages | same | 36 / 79 / 37% | no |
| risk_chemical_plants | risk_benefit | …chemical plants | same | 34 / 84 / 32% | no |
| risk_pesticides | risk_benefit | …pesticides | same | 33 / 79 / 34% | no |
| omission_vaccine | decisions | Flu with 10% death risk vs vaccine with 5% risk: take it? | 4: definitely not → definitely take | 38 / **92** / 39% | no |
| denominator_tray | decisions | Draw from small tray (1/10 black) or large tray (8/100)? | small / large tray | 64 / 84 / 62% | no |

- All Twin wording is verified (row 0 of the CSVs). Live mode: the 10 non-sensitive items (`risk_benefit` + `decisions`).
- Not used: the Q295 block (not a fixed item) and every `Q156–Q203` reasoning task (≥49% missing, between-subjects).
- The rules apply to `real`. The GPT twin breaks the 85% rule on `risk_bicycles` and `omission_vaccine`. That is a finding about the twin, not a reason to drop the items.

### GSS 2024: 24 items, 4 families × 6 (target set n = 1,116)
| Code | Family | Wording | Scale | Missing | Top | Sensitive | Verified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trppl | trust_institutions | Generally speaking, would you say that most people can be trusted…? (0–10) | 0 can't be too careful → 10 most people can be trusted | 7% | 24% | no | yes |
| trcourts | trust_institutions | How much do you personally trust… America's courts (0–10) | 0 no trust → 10 complete trust | 9% | 23% | no | yes |
| trcong | trust_institutions | How much do you personally trust… The U.S. Congress (0–10) | 0 no trust at all → 10 complete trust | 9% | 22% | yes | yes |
| intcnfrm | trust_institutions | …political matters online, how often… Try to confirm information you find by searching online for another source? | very often / often / sometimes / rarely / never | 9% | 32% | no | yes |
| satdemoc | trust_institutions | On the whole, are you {fill} with the way democracy works in the United States? | very / fairly / not very / not at all satisfied | 5% | 37% | yes | yes |
| polint | trust_institutions | How interested would you say you personally are in politics? | very → not at all interested (5) | 5% | 26% | yes | yes |
| satfin | money_economy | …would you say that you are pretty well satisfied with your present financial situation, more or less satisfied, or not satisfied at all? | 3 | 0% | 45% | no | yes |
| finalter | money_economy | During the last few years, has your financial situation been getting better, worse, or has it stayed the same? | better / stayed same / worse | 1% | 38% | no | yes |
| finrela | money_economy | Compared with American families in general, would you say your family income is far below … far above average? | 5 | 1% | 40% | no | yes |
| class | money_economy | …which would you say you belong in: the lower class, the working class, the middle class, or the upper class? | 4 | 2% | 45% | no | yes |
| econstat | money_economy | What do you think about the state of the economy these days in the United States? | very good → very bad (5) | 2% | 33% | no | yes |
| tax | money_economy | Do you consider the amount of federal income tax which you have to pay as too high, about right, or too low? | 3 | 3% | 67% | yes | yes |
| happy | work_life | …would you say that you are very happy, pretty happy, or not too happy? | 3 | 1% | 60% | no | yes |
| life | work_life | In general, do you find life exciting, pretty routine, or dull? | 3 | 1% | 54% | no | yes |
| socfrend | work_life | …how often: Spend a social evening with friends who live outside the neighborhood? | almost daily → never (7) | 1% | 23% | no | yes |
| socrel | work_life | How often do you spend a social evening with relatives? | almost daily → never (7) | 0% | 21% | no | yes |
| workless | work_life | In your present or most recent job, have you ever … because of your responsibilities to members of your family? Cut back on your work? | yes / no | 4% | 72% | no | yes |
| meovrwrk | work_life | Family life often suffers because men concentrate too much on their work. | strongly agree → strongly disagree (5) | 1% | 39% | no | yes |
| techesy | tech_science | Agree/disagree: Technology makes our lives easier. | strongly agree → strongly disagree (5) | 5% | 51% | no | yes |
| harmgood1 | tech_science | Agree/disagree: Overall, technology does more harm than good. | same | 5% | 35% | no | yes |
| aiworry | tech_science | …how worried… that in the next 10 years machines, computer programs, and AI will take over many of the jobs done by humans? | very → not at all worried (5) | 8% | 39% | no | yes |
| aidrive | tech_science | Comfort (0–10): Being driven in a driverless car. | 0 totally uncomfortable → 10 totally comfortable | 5% | 45% | no | yes |
| aimed | tech_science | Comfort (0–10): Having a medical operation performed on you by a robot. | same | 6% | 36% | no | yes |
| natsci | tech_science | Are we spending too much, too little, or about the right amount on supporting scientific research? | too little / about right / too much | 5% | 45% | no | yes |

- **Live mode (GSS): 20 items** (all except the sensitive `trcong`, `satdemoc`, `polint`, `tax`).
- Excluded by rule: sexual behaviour, health/mental-health conditions (including the ballot-1 mental-health vignettes), immigration status, religious identity.

## 4. Splits

`blindspot/eval/splits.py`: `make_splits(ids, seed=42, fractions=(0.5, 0.3, 0.2))` sorts the ids, permutes them with `numpy.random.default_rng(seed)`, and takes floor(n·0.5) for pool, floor(n·0.3) for test, and the rest for holdout. Saved once to `data/processed/splits_{dataset}.json` (`dataset, seed, fractions, n, ids_fingerprint, pool, test, holdout`); `get_or_make_splits` refuses to reuse a saved split if the id set changed. `assert_no_leak(ids_used, splits, allowed=("pool",))` raises `LeakError` for any id outside the allowed splits. This differs from the handover signature: it takes `splits` explicitly.

| Dataset | n | pool | test | holdout |
| --- | --- | --- | --- | --- |
| twin2k (one split shared by real, gpt41mini_twin, retest) | 2,058 | 1,029 | 617 | 412 |
| gss (sample 13, ballot 1) | 1,116 | 558 | 334 | 224 |

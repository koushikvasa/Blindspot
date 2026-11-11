# Data notes (H1)

Everything below was found by reading the files, not assumed. Raw data lives in `data/raw/` (git-ignored).
Item lists are **PROPOSED, awaiting approval**. Splits and tests are not built yet.

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

- `GSS2024.dta`: **3,986 rows × 989 columns**, ID `id` (unique). Note: the codebook's frequency tables total 3,309 cases, so the release file holds more rows than the codebook counts. Not investigated further; flagged.
- Loaded with `pyreadstat.read_dta`. 940 columns come back as `object` (ints mixed with NaN), so we use `pd.to_numeric`. Reserved codes are Stata extended missing values (`d` don't know / can't choose, `i` IAP = not asked, `n` no answer, `s` skipped on web, `r` refused, `x`/`y`/`z`, …) and load as **NaN**.
- **Split ballots.** `ballot` ∈ {1: 1,341; 2: 1,294; 3: 1,351}. Most attitude items are asked on only 1–2 ballots, so only about 105 variables have <10% missing on the full sample, and most of those are demographics or admin. **Proposal: the GSS target set is ballot 1 (n = 1,341).** It is the only ballot that carries the ISSP *Digital Society* module (technology, AI, 0–10 trust scales) at <10% missing, alongside the full-sample money and life items. Not available on ballot 1: the `con*` confidence battery and `trust/fair/helpful` (ballots 2–3).
- Weights (`wtssps`, `wtssnrps`) exist but are not used for coverage (unweighted people-level analysis).
- Demographics (descriptive only): `age_band` from `age` (same bands as Twin; 89 = "89 or older"), `gender` from `sex`, `education` from `degree` (5 GSS levels), `income_band` from `income16` (26 codes → <$30k / $30–50k / $50–75k / $75–110k / $110k+; the top cut is 110k because code 22 spans $90–110k), `region` (4 census regions), `race` (white/black/other). Missing → `unknown`.
- Wording: taken from the codebook. For ISSP items (and `satdemoc`) the codebook prints only a short label; those are marked *label-only* below and need checking against the ISSP 2024 questionnaire before anyone sees them.

## 3. Proposed items (awaiting approval)

Rules: missing < 10% in the target set, no option above 85%, ordinal or binary. "Top" = share of the most common option.

### Twin-2K-500: 30 items (missing shown as real / retest / twin)
| Code | Theme | Wording (short) | Scale | Missing | Top real | Top twin | Sensitive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| support_carbon_tax | policy_views | Support/oppose: tax on carbon emissions | 5: strongly oppose → strongly support | 0/0/0% | 34% | 47% | yes |
| support_clean_energy_lowincome | policy_views | …40% of clean-energy spending to low-income communities | same | 0% | 33% | 42% | yes |
| support_carbon_free_2035 | policy_views | …federal investment for carbon-free electricity by 2035 | same | 0% | 34% | 34% | yes |
| support_medicare_for_all | policy_views | …Medicare for All | same | 0% | 50% | 44% | yes |
| support_public_option | policy_views | …public option | same | 0% | 38% | 44% | yes |
| support_citizenship_path | policy_views | …path to citizenship for undocumented immigrants | same | 0% | 40% | 44% | yes |
| support_paid_family_leave | policy_views | …required paid family leave | same | 0% | 52% | 60% | yes |
| support_wealth_tax | policy_views | …2% tax on assets above $50M | same | 0% | 53% | 37% | yes |
| support_more_deportations | policy_views | …increasing deportations | same | 0% | 27% | 43% | yes |
| support_senior_vouchers | policy_views | …healthcare vouchers for seniors instead of Medicare | same | 0% | 29% | 64% | yes |
| est_public_carbon_tax | perceived_consensus | What % of the public supports: carbon tax | 0–100 binned into 5: 0-19 / 20-39 / 40-59 / 60-79 / 80-100% | 0% | 37% | 60% | yes |
| est_public_clean_energy_lowincome | perceived_consensus | …40% to low-income communities | same | 0% | 34% | 51% | yes |
| est_public_carbon_free_2035 | perceived_consensus | …carbon-free 2035 | same | 0% | 34% | 48% | yes |
| est_public_medicare_for_all | perceived_consensus | …Medicare for All | same | 0% | 35% | 34% | yes |
| est_public_public_option | perceived_consensus | …public option | same | 0% | 33% | 42% | yes |
| est_public_citizenship_path | perceived_consensus | …path to citizenship | same | 0% | 40% | 36% | yes |
| est_public_paid_family_leave | perceived_consensus | …paid family leave | same | 0% | 35% | 42% | yes |
| est_public_wealth_tax | perceived_consensus | …wealth tax | same | 0% | 29% | 30% | yes |
| est_public_more_deportations | perceived_consensus | …more deportations | same | 0% | 36% | 56% | yes |
| est_public_senior_vouchers | perceived_consensus | …senior vouchers | same | 0% | 35% | 58% | yes |
| benefit_bicycles | risk_benefit | Rate how beneficial: bicycles | 7: not at all beneficial → extremely beneficial | 0% | 39% | 67% | no |
| benefit_alcohol | risk_benefit | …alcoholic beverages | same | 0% | 33% | 77% | no |
| benefit_chemical_plants | risk_benefit | …chemical plants | same | 0% | 21% | 66% | no |
| benefit_pesticides | risk_benefit | …pesticides | same | 0% | 27% | 69% | no |
| risk_bicycles | risk_benefit | Rate how risky: bicycles | 7: not at all risky → extremely risky | 0% | 32% | **91%** | no |
| risk_alcohol | risk_benefit | …alcoholic beverages | same | 0% | 36% | 79% | no |
| risk_chemical_plants | risk_benefit | …chemical plants | same | 0% | 34% | 84% | no |
| risk_pesticides | risk_benefit | …pesticides | same | 0% | 33% | 79% | no |
| omission_vaccine | reasoning | Flu with 10% death risk vs vaccine with 5% risk: take it? | 4: definitely not → definitely take | 0% | 38% | **92%** | no |
| denominator_tray | reasoning | Draw from small tray (1/10 black) or large tray (8/100)? | 2: small / large tray | 0% | 64% | 84% | no |

- Not proposed: the Q295 block (not a fixed item, see above) and every `Q156–Q203` reasoning task (≥49% missing, between-subjects). Pooling the two versions of a task into one item would give 0% missing but mix framings, so we don't.
- Sensitive: all policy items and the public-support estimates (also political). That leaves **10 non-sensitive Twin items for live mode**.
- The rules apply to `real`. The GPT twin breaks the 85% rule on `risk_bicycles` and `omission_vaccine`. That is a finding about the twin, not a reason to drop the items.

### GSS 2024 (ballot 1, n = 1,341): 24 items, 4 themes × 6
| Code | Theme | Wording (codebook) | Scale | Missing | Top | Sensitive |
| --- | --- | --- | --- | --- | --- | --- |
| trppl | trust_institutions | Trust in others, 0–10 *(label-only)* | 0 can't be too careful → 10 most people can be trusted | 7.4% | 25% | no |
| trcourts | trust_institutions | How much trust in American courts *(label-only)* | 0 no trust → 10 complete trust | 8.8% | 24% | no |
| trcong | trust_institutions | Trust in Congress *(label-only)* | 0 no trust at all → 10 complete trust | 9.3% | 23% | no |
| **intrust** ❌ | trust_institutions | How much R trusts people only met online *(label-only)* | 0–10 | **11.1%** (fails <10%) | 31% | no |
| satdemoc | trust_institutions | Satisfied with the way democracy works in America *(label-only)* | very / fairly / not very / not at all satisfied | 4.2% | 38% | no |
| polint | trust_institutions | How interested would you say you personally are in politics? | very → not at all interested (5) | 5.6% | 27% | no |
| satfin | money_economy | …pretty well satisfied with your present financial situation, more or less satisfied, or not satisfied at all? | 3 | 0.4% | 47% | no |
| finalter | money_economy | During the last few years, has your financial situation been getting better, worse, or stayed the same? | better / stayed same / worse (reordered to be ordinal) | 0.7% | 37% | no |
| finrela | money_economy | Compared with American families in general, your family income is…? | far below → far above average (5) | 1.0% | 41% | no |
| class | money_economy | Which social class do you belong in? | lower / working / middle / upper | 1.7% | 45% | no |
| econstat | money_economy | State of the economy these days in the US? | very good → very bad (5) | 2.2% | 32% | no |
| tax | money_economy | Federal income tax you pay: too high, about right, too low? | 3 | 2.8% | 67% | no |
| happy | work_life | Very happy, pretty happy, or not too happy? | 3 | 1.0% | 60% | no |
| life | work_life | Life exciting, pretty routine, or dull? | 3 | 0.9% | 57% | no |
| socfrend | work_life | How often a social evening with friends outside the neighborhood? | almost daily → never (7) | 1.0% | 23% | no |
| socrel | work_life | How often a social evening with relatives? | almost daily → never (7) | 0.9% | 21% | no |
| workless | work_life | In present/most recent job, ever cut back on work because of family responsibilities? | yes / no | 3.9% | 72% | no |
| meovrwrk | work_life | Family life often suffers because men concentrate too much on their work. | strongly agree → strongly disagree (5) | 0.7% | 38% | no |
| techesy | tech_science | Does technology make our lives easier? *(label-only)* | strongly agree → strongly disagree (5) | 5.4% | 50% | no |
| harmgood1 | tech_science | Does technology do more harm than good? *(label-only)* | same | 5.7% | 37% | no |
| aiworry | tech_science | Worried that AI will take over many jobs done by humans *(label-only)* | very → not at all worried (5) | 7.8% | 39% | no |
| aidrive | tech_science | Comfort with a driverless car *(label-only)* | 0 totally uncomfortable → 10 totally comfortable | 5.3% | 42% | no |
| aimed | tech_science | Comfort with a medical operation by a robot *(label-only)* | same | 6.3% | 35% | no |
| natsci | tech_science | Spending on supporting scientific research: too much, too little, about right? | too little / about right / too much | 4.5% | 45% | no |

- `intrust` fails the missingness rule. Ballot-1 replacements that pass: `intcnfrm` (how often R confirms online information with other sources; 8.6% missing, top 33%), `inttrig` (reads political opinions they disagree with online; 8.8%, 40%), `infoprofit` (would give personal info companies could profit from; 6.8%, 38%). Or use 5 trust items + 7 tech.
- Excluded by rule: sexual behaviour, health/mental-health conditions (the ballot-1 mental-health vignettes), immigration status, religious identity.
- Possibly sensitive (not flagged yet): `trcong`, `satdemoc`, `polint`, `tax`. They touch politics but are not partisan policy positions.

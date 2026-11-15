"""Build the sprint report (DOCX on Apart's official template); PDF export is done in Word.

    D:/ComputationalLab/automation/.venv/Scripts/python.exe experiments/ai_fabrication/report/build_report.py

Numbers quoted in the text come from results/test/ and results/dev/ (see analysis.py); figures from figures.py.
"""

from __future__ import annotations

import copy
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "apart_template.docx"
OUT = HERE / "Akber_2026_AI_fabricated_interviews.docx"

TITLE = "When the Curbstoner Has a Chatbot: Survey Fabrication Checks Miss AI-Written Interviews, and Three Answer-Level Signals Catch Them"
AUTHOR = "Syed Hasnain Akber"
AFFILIATION = "Karachi School of Business and Leadership (KSBL)"

ABSTRACT = (
    "Household surveys run by field interviewers inform aid, health and education decisions in low- and "
    "middle-income countries. Their main safeguard against interviewers who invent interviews (curbstoning) is a "
    "battery of statistical checks designed around how humans fabricate. We ask whether these checks still work "
    "when the interviewer asks a language model to fill in the interviews. Using FieldGuard, an open-source "
    "implementation of 14 standard high-frequency checks, we injected interviews written by Claude Opus 5.5 under "
    "three prompting strategies (naive; informed by colleagues' summary statistics; asked to look genuine) into "
    "synthetic surveys of 30 enumerators, and compared them with rule-based fabricators. When the fabricated "
    "interviews carried honest-looking timestamps and GPS, FieldGuard flagged 0 of 25 AI fabricators across three "
    "held-out surveys (5 of 9 rule-based fabricators); with the usual paradata traces it flagged all 25. The "
    "model-written answers were nonetheless distinctive: ratings varied less within an interview (mean SD 0.59-0.83 "
    "versus 1.04), households were more alike, and amounts were rounded more often. Three answer-level signals, "
    "chosen on two development surveys and frozen before testing, flagged 25 of 25 AI fabricators on the held-out "
    "surveys while flagging 4 of 88 honest enumerators. The model declined 2 of 45 requests, both asking for data "
    "that would look genuine. Answer-level checks remain a cheap layer of trust infrastructure for survey data, but "
    "they need recalibrating for model-written fabrication, and paradata and back-checks stay essential."
)

# Blocks: ("h2"|"h3"|"p"|"bullets"|"numbered"|"figure"|"table"|"caption"|"refs", payload)
BODY: list[tuple[str, object]] = [
    ("h2", "1. Introduction"),
    ("p",
     "Much of what governments and donors know about poor households comes from interviewer-administered surveys: "
     "enumerators visit homes with a tablet, ask a questionnaire and upload the answers. Survey teams have long known "
     "that some interviewers invent interviews instead of conducting them, a practice called curbstoning, and they "
     "defend against it with high-frequency checks run during fieldwork: duplicated or straight-lined answers, "
     "digit heaping, Benford's law, implausibly short interviews, GPS points at the interviewer's home, and "
     "enumerator-level anomaly scores. Flags send supervisors to re-interview households (back-checks). These checks "
     "are a working piece of trust infrastructure: they decide which numbers reach a programme report."),
    ("p",
     "The checks were designed around human fabricators, whose shortcuts leave characteristic traces. A language "
     "model removes the labour that produces those traces. An enumerator can paste the questionnaire into a chat "
     "assistant and ask for twenty \"realistic\" interviews, and the request is indistinguishable from an ordinary "
     "request for synthetic test data. If the checks miss such interviews, fabricated data flows into decisions with "
     "the same apparent quality as real fieldwork. Recent work shows that language-model agents can pass as "
     "respondents in online surveys (Westwood, 2025; Zhang et al., 2026). We study the neglected interviewer side: "
     "the fabricator is the data collector, not the respondent, and the defence is the statistical quality-control "
     "layer that field surveys already run."),
    ("p", "Our main contributions are:"),
    ("numbered", [
        "A threat model and reproducible test bed for AI-assisted curbstoning: three prompting strategies, two "
        "paradata conditions and an answer key, built on an open-source survey quality tool and released with every "
        "prompt and every model answer.",
        "Evidence that a standard battery of 14 fabrication checks, which flags rule-based fabricators, flags none of "
        "25 AI fabricators on held-out surveys when timestamps and GPS look honest, and that the gap is one of "
        "calibration as much as coverage: the risk score still ranks AI fabricators above honest enumerators.",
        "Three answer-level signals (within-interview rating spread, similarity of an enumerator's households to "
        "each other, and closeness to the median household), chosen on development data and frozen before testing, "
        "that flag all 25 held-out AI fabricators while flagging 4 of 88 honest enumerators.",
    ]),
    ("h2", "2. Related Work"),
    ("p",
     "Interviewer falsification detection. Survey methodology has a mature literature on detecting fabricated "
     "interviews from the data: near-duplicate interviews (Kuriakose and Robbins, 2016), Benford's law and digit "
     "preference (Judge and Schechter, 2009; Schräpler, 2011), response simplification such as straight-lining "
     "(Blasius and Thiessen, 2015), multivariate indicators combined by cluster analysis (Bredl et al., 2012), "
     "comparisons of detection methods on a survey with known falsifiers (Schwanhäuser et al., 2022), and questions "
     "with known distributions planted to catch fabrication (Hernandez et al., 2022). All of these assume a human "
     "fabricator. We test whether their premises survive a fabricator who delegates the inventing to a model."),
    ("p",
     "Language models and survey data. Models can emulate the answer distributions of human subgroups, the "
     "\"silicon sample\" idea (Argyle et al., 2023), but their samples are homogenised and compress response scales "
     "(Li et al., 2025; Lukauskas and Šarkauskaitė, 2026), and simple classifiers separate them from human samples. "
     "A separate line of work studies respondents who use models or agents to answer online surveys: Westwood "
     "(2025) shows agents evading standard attention checks, Zhang et al. (2026) estimate agent prevalence across "
     "platforms, Sehgal et al. (2026) observe AI assistance directly, and Wang et al. (2026) benchmark detectors. We "
     "found no study of interviewers using models to fabricate interviewer-administered surveys, where the "
     "defence is not text detection but the statistical checks run on the whole dataset."),
    ("p",
     "Our work sits in the trust-infrastructure agenda of this sprint: early verification tools accumulate trust "
     "whether or not they deserve it, so they need to be audited against the adversaries they will actually meet "
     "(Cotton-Barratt and Finnveden, 2025)."),
    ("h2", "3. Methods"),
    ("p",
     "Test bed. FieldGuard (v0.1.0, written before the sprint and unchanged in this study) reads survey exports, "
     "runs 14 checks and scores each enumerator with a rule score and an Isolation-Forest anomaly score combined into "
     "a risk score; an enumerator is flagged at a risk score of 30 or more. Its synthetic generator mimics a "
     "KoboToolbox household survey from rural central Kenya: 30 enumerators with about 25 interviews each across "
     "15 villages, with honest heterogeneity in speed, don't-know rates, rounding habits and village wealth. The "
     "questionnaire has demographics, five money questions (income, business income, livestock value, food and "
     "other spending), land, water access, phones, an eight-item wellbeing scale with three reverse-worded items, and "
     "six service-satisfaction items. All data in the study are synthetic; no real survey or personal data is used."),
    ("p",
     "Threat model and fabrication. In each survey, three enumerators (10%) fabricate 60-100% of their interviews, "
     "matching FieldGuard's benchmark. For each fabricator we wrote the request they might type into a chat "
     "assistant, under three strategies: S0 naive (the questionnaire and the number of households), S1 informed "
     "(plus summary statistics of colleagues' interviews in the same two villages, which a fabricator could obtain "
     "from a past round or a supervisor's report), and S2 careful (S1 plus a request that the records look like "
     "genuine field data, with realistic rounding, some don't-know answers and consistent households). No purpose "
     "is stated in any request (Appendix B). Each request was answered by Claude Opus 5.5 through a fresh Claude "
     "Code subagent that saw only the request file, stored under a neutral name, and wrote its JSON answer "
     "directly to disk. This route replaced the API because no API key was available during the sprint; it means "
     "the model ran inside an agent harness with its own system prompt, not a bare chat window. The answers "
     "replaced the content of the chosen interviews; identifiers were kept."),
    ("p",
     "Paradata conditions. In content-only, fabricated interviews keep the honest interview's timestamps and GPS, "
     "isolating what the answers alone reveal (a fabricator who also fakes paradata well, or who visits but invents "
     "the answers). In with-paradata, they also carry the usual curbstoning traces as FieldGuard models them: "
     "interviews a quarter to half as long, entered in evening batches, with GPS points clustered at home. The "
     "comparison is FieldGuard's own rule-based fabricators (copying, straight-lining, heaping, Benford violations, "
     "low variance and similar patterns), restricted to answer patterns in the content-only condition."),
    ("p",
     "New signals. We computed seven candidate enumerator-level signals from the answers only, each standardised "
     "against the enumerator's peers by a robust z-score (median and median absolute deviation across enumerators), "
     "two-sided. They were motivated by the synthetic-sample literature: rounding share, Cronbach's alpha of each "
     "rating grid, mean within-interview standard deviation of the 14 rating items, spread of the food-to-income "
     "ratio, typicality (mean distance of an interview from the median household in robust units) and spread (mean "
     "distance from each interview to its nearest neighbour among the same enumerator's interviews)."),
    ("p",
     "Protocol. Five surveys were generated: two for development (seeds 101-102) and three held out (seeds "
     "201-203), fixed before any model call. On the development surveys we chose the rule \"flag an enumerator with "
     "at least 8 interviews if any of rating spread, household similarity or typicality has |z| of 3.5 or more\" "
     "(13 of 18 AI and 5 of 6 rule-based fabricators, 0 of 216 honest enumerator-runs in development). The rule was "
     "committed to the repository (commit b0ea6d1) before the held-out surveys were scored once. We report flagged "
     "counts with Wilson 95% intervals and the per-survey ROC-AUC of FieldGuard's risk score."),
    ("h2", "4. Results"),
    ("p",
     "Generation. The model answered 43 of 45 requests with complete records (807 interviews) and declined 2, both "
     "S2 requests (2 of 15). Its stated reason was that records meant to blend into colleagues' data and \"look "
     "like genuine field data\" amount to fabricating fieldwork. In six of the answered requests it added a note "
     "that the records were invented and should not be passed off as real interviews; a fabricator would simply "
     "discard such a note. For a declined request the enumerator stays honest in that run, so the S2 rows below "
     "have 7 fabricators instead of 9."),
    ("table", "fieldguard"),
    ("caption",
     "Table 1. FieldGuard as released, on the three held-out surveys. Flagged: fabricating enumerators whose risk "
     "score reached 30. Honest flagged: honest enumerators flagged in the same runs. AUC: ROC-AUC of the risk "
     "score separating fabricators from honest enumerators, mean over surveys."),
    ("p",
     "FieldGuard misses AI-written answers. With honest-looking paradata, FieldGuard flagged none of the 25 AI "
     "fabricators, against 5 of 9 rule-based fabricators (Table 1). The miss is largely one of calibration: the "
     "risk score still ranked AI fabricators above honest enumerators (AUC 0.86-1.00), but their scores stayed "
     "below the flag line (median 7.4-18.9, against 1.6-1.7 for honest enumerators and a threshold of 30). Among the "
     "built-in checks, only heaping fired often (6 of 9 S0 and 7 of 7 S2 fabricators, 0 honest): the model rounded "
     "money amounts more than simulated respondents did, most of all when asked to round \"the way people do\". "
     "Straight-lining, Benford and the enumerator-effect checks, which caught rule-based fabricators, never fired "
     "for AI fabricators. Once the usual paradata traces were present, every AI fabricator "
     "was flagged, as were all rule-based ones: the paradata checks carry the load."),
    ("figure", "fig1_recall.png"),
    ("caption",
     "Figure 1. Fabricating enumerators flagged on the held-out surveys when only the answers are fabricated "
     "(content-only). Blue: FieldGuard as released (risk score of 30 or more). Orange: the three new answer-level "
     "signals, frozen on development surveys. False flags among honest enumerators: 0 for FieldGuard; 4 of 88 "
     "distinct honest enumerators for the new signals across all held-out runs."),
    ("p",
     "Model-written answers are too regular. The AI interviews differ from honest ones in ways the synthetic-sample "
     "literature predicts (Table 2): ratings within an interview vary less (mean standard deviation 0.59-0.83 "
     "against 1.04), the food share of income and household size vary less, and incomes are rounded more often. "
     "At the enumerator level these show up as interviews that are closer to each other and to the median household "
     "than any honest enumerator's (Figure 2)."),
    ("table", "contrasts"),
    ("caption",
     "Table 2. Answer features of the held-out AI interviews and of the honest interviews in the same surveys."),
    ("figure", "fig2_signals.png"),
    ("caption",
     "Figure 2. The three frozen signals on the held-out surveys (content-only), as robust z-scores against the "
     "enumerator's peers. Dashed lines: the frozen threshold of |z| = 3.5. Each honest enumerator appears once; AI "
     "fabricators pool the three strategies."),
    ("p",
     "The frozen signals catch them. On the held-out surveys the frozen rule flagged 9 of 9 S0, 9 of 9 S1 and 7 of "
     "7 S2 AI fabricators (25 of 25; Wilson 95% interval 0.87-1.00 pooled) and 7 of 9 rule-based fabricators. "
     "Every AI flag was in the too-regular direction (negative z), and household similarity alone caught all 25. "
     "On the three held-out surveys with no fabrication at all, the rule flagged 2 of 90 honest enumerators (2%); "
     "counting every held-out run, 4 of 88 distinct honest enumerators were flagged at least once, compared with 1 "
     "of 60 in development. A supervisor would follow these up with a back-check. Informing the model with colleagues' "
     "statistics (S1) or asking it to look genuine (S2) did not make it harder to catch by these signals: matching "
     "the marginal distributions does not restore the within-household and between-household variation of real "
     "interviews."),
    ("h2", "5. Discussion and Limitations"),
    ("p",
     "The result cuts both ways for trust infrastructure. The bad news is that a standard class of checks, "
     "tuned for human shortcuts, gave every AI fabricator a clean bill when the paradata looked honest. The good news "
     "is that model-written interviews are not hard to tell apart from real ones in aggregate: they are too "
     "regular, and three cheap statistics that any survey team can compute caught them. Paradata (timestamps, GPS, "
     "audio audits) and back-checks remain the backbone; answer-level checks are the layer that still works when a "
     "fabricator fakes the paradata or visits households but invents the answers."),
    ("h3", "Limitations"),
    ("bullets", [
        "Synthetic honest data. \"Honest\" interviews come from a simulator, so detection is measured against simulated "
        "people. Real honest enumerators differ more from each other (enumerator effects), which would raise false "
        "flags, and real respondents may be more regular than the simulator in some respects. The result needs "
        "confirming on real survey rounds with back-check outcomes.",
        "One model, one access route. All answers came from Claude Opus 5.5 inside the Claude Code harness. Other "
        "models, chat interfaces, sampling settings or a fine-tuned generator could produce more varied data. A "
        "local open-weights model was planned but was too slow on our laptop to finish in the sprint.",
        "Small numbers. 25 held-out AI fabricators in 3 surveys; intervals are wide and the honest false-flag rate "
        "(4 of 88) is imprecise. The development rule was chosen on 18 fabricators.",
        "Non-adaptive fabricator. No request named the checks. A fabricator who knows them can ask for more spread "
        "and less similarity, or mix in real interviews. We expect an arms race; the signals buy time and raise the "
        "cost of fabrication rather than ending it.",
        "Fixed design. 10% fabricators fabricating 60-100% of their interviews; lower shares dilute every signal.",
    ]),
    ("h3", "Future Work"),
    ("p",
     "Validate on real survey rounds with known back-check outcomes; add more models, including open-weights models "
     "a fabricator could run offline, and adversarial prompts that target the checks; recalibrate FieldGuard's "
     "scoring so that answer-level regularity counts towards the flag, and add record-level detection; and study "
     "how supervisors act on these flags in practice."),
    ("h2", "6. Conclusion"),
    ("p",
     "Statistical fabrication checks are part of the trust infrastructure behind household survey data. Tuned for "
     "human curbstoners, a standard battery missed every interviewer who had a language model invent the answers, "
     "unless the paradata gave them away. The model's answers were too regular, and three answer-level signals "
     "frozen in advance caught all of them on held-out surveys at a modest false-flag cost. Survey teams should "
     "assume that some fabrication will now be model-written, keep paradata and back-checks, and add checks for "
     "data that is too regular, not only too sloppy."),
    ("h2", "Code and Data"),
    ("bullets", [
        "Code repository: github.com/hasnainakber9/fieldguard, branch sprint/ai-fabrication, folder "
        "experiments/ai_fabrication (new during the sprint; FieldGuard itself is preexisting and marked as such in "
        "the README).",
        "Data: every prompt payload and every model answer, verbatim, in experiments/ai_fabrication/data; all data are "
        "synthetic.",
        "Info-hazard note: the prompts are ordinary requests for synthetic data and add nothing a fabricator could "
        "not write; see Appendix A.",
    ]),
    ("h2", "References"),
    ("refs", [
        "Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J., Rytting, C., and Wingate, D. (2023). Out of One, Many: "
        "Using Language Models to Simulate Human Samples. Political Analysis. arXiv:2209.06899.",
        "Blasius, J., and Thiessen, V. (2015). Should we trust survey data? Assessing response simplification and data "
        "fabrication. Social Science Research, 52.",
        "Bredl, S., Winker, P., and Kötschau, K. (2012). A statistical approach to detect interviewer falsification "
        "of survey data. Survey Methodology, 38(1), 1-10.",
        "Cotton-Barratt, O., and Finnveden, L. (2025). AI for AI for Epistemics. Forethought. "
        "https://www.forethought.org/research/ai-for-ai-for-epistemics",
        "Hernandez, I., Ristow, T., and Hauenstein, M. (2022). Curbing curbstoning: Distributional methods to detect "
        "survey data fabrication by third-parties. Psychological Methods. doi:10.1037/met0000403.",
        "Judge, G., and Schechter, L. (2009). Detecting problems in survey data using Benford's Law. Journal of Human "
        "Resources, 44(1).",
        "Kuriakose, N., and Robbins, M. (2016). Don't get duped: Fraud through duplication in public opinion surveys. "
        "Statistical Journal of the IAOS, 32(3). doi:10.3233/SJI-160978.",
        "Li, D., Li, L., and Qiu, H. S. (2025). ChatGPT is not A Man but Das Man: Representativeness and Structural "
        "Consistency of Silicon Samples Generated by Large Language Models. arXiv:2507.02919.",
        "Lukauskas, M., and Šarkauskaitė, V. (2026). Plausible but Not Valid: A Psychometric Audit of LLMs as "
        "Synthetic Survey Respondents. arXiv:2608.14606.",
        "Schräpler, J.-P. (2011). Benford's Law as an instrument for fraud detection in surveys using the data of the "
        "Socio-Economic Panel (SOEP). Jahrbücher für Nationalökonomie und Statistik, 231(5-6).",
        "Schwanhäuser, S., Sakshaug, J. W., and Kosyakova, Y. (2022). How to Catch a Falsifier: Comparison of "
        "Statistical Detection Methods for Interviewer Falsification. Public Opinion Quarterly, 86(1), 51-81. "
        "doi:10.1093/poq/nfab066.",
        "Sehgal, N. K. R., Tonneau, M., Folk, D., Ungar, L., and Guntuku, S. C. (2026). AI-Assisted Submissions in "
        "Online Research Are Rare and Highly Concentrated but Routinely Approved. arXiv:2610.09279.",
        "Wang, Q., Mamaev, B., and Leckie, C. (2026). Towards Detecting AI-Assisted Responses in Online Surveys. "
        "arXiv:2609.17317.",
        "Westwood, S. J. (2025). The potential existential threat of large language models to online survey "
        "research. Proceedings of the National Academy of Sciences. doi:10.1073/pnas.2518075122.",
        "Zhang, G., Walatka, R., Chen, S. Y., Moore, A. K., Urminsky, O., Fernandez, K., Low, A., Bogard, J. E., and "
        "Fox, C. R. (2026). Estimating the threat of AI-agent responding across online survey platforms. Proceedings "
        "of the National Academy of Sciences. doi:10.1073/pnas.2606460123.",
    ]),
    ("h2", "Appendix A. Limitations and Dual-Use / Ethical Considerations"),
    ("p",
     "What the result does and does not establish. It shows that, on a realistic simulator, a standard check battery "
     "misses model-written fabrication when paradata look honest, and that three answer-level signals separate it "
     "from simulated honest data. It does not establish detection rates on real surveys, against other models, or "
     "against a fabricator who adapts to these signals."),
    ("p",
     "Dual use. Publishing which checks miss model-written interviews, and which signals catch them, could help a "
     "fabricator tune their requests. We judge the benefit to survey teams larger: the checks are already public, "
     "the requests we used are ordinary requests for synthetic data, and survey managers, not fabricators, are the "
     "ones who currently lack this information. We do not publish prompts that target specific checks. Flags are "
     "leads for supervisors to verify by back-check, never proof of misconduct, and should not be used to sanction "
     "anyone on their own."),
    ("p",
     "Eliciting the data. The requests to the model did not state that they were part of a study, because the "
     "question is what a fabricating interviewer would receive. The village and its households are synthetic, so no "
     "real dataset could be corrupted. When the model declined, we recorded the refusal and did not rephrase or "
     "retry the request."),
    ("p", "Personal data. None. All interviews, names and phone numbers are synthetic and labelled as such."),
    ("h2", "Appendix B. Prompts"),
    ("p",
     "Every request began with the questionnaire below (wording abbreviated here; full text in "
     "data/prompts and in each answer file), preceded by: \"I need N completed household survey interviews from two "
     "villages in rural central Kenya: n1 from A and n2 from B (list the A households first). Fill in realistic "
     "answers for each household using the questionnaire below. Each record is a different household. Amounts are "
     "in Kenyan shillings (KES).\" The questionnaire lists 29 fields: age, sex, household size, children, business "
     "and its income, total monthly income, livestock value, food and other spending, land and its size, water source "
     "and minutes to it, phones, eight wellbeing statements (1 = strongly disagree to 5 = strongly agree; items 2, 5 "
     "and 7 are worded negatively) and six service-satisfaction items, with -99 for don't know and -98 for refused."),
    ("p",
     "S1 adds: \"For reference, so your records fit these villages:\" followed by colleagues' medians, quartiles and "
     "ranges for the numeric questions, yes-shares, water-source shares, item means and the don't-know share. S2 adds "
     "to S1: \"Make the records look like genuine field data: people round amounts the way they do in real interviews, "
     "a few answers are don't know, households differ from each other, and the answers within each household are "
     "consistent with each other.\" All requests end with \"Return exactly N records\" and a line asking for one JSON "
     "object with the listed keys."),
    ("h2", "LLM Usage Statement"),
    ("p",
     "This project was built with substantial help from Claude (Anthropic) through Claude Code: it wrote the "
     "experiment code, ran the analysis and drafted this report under the author's direction; the author chose the "
     "question and design, reviewed the code and results, and is responsible for the final text. Claude Opus 5.5 "
     "also served as the fabricating model under study, through separate subagents that saw only the request. All "
     "numbers were produced by the released code, and every reference was checked against its source."),
]


def _clear_after(body, keep) -> None:
    for el in list(body):
        if el is keep or el.tag == qn("w:sectPr"):
            continue
        body.remove(el)


def _set_cell_text(cell, text: str) -> None:
    ps = cell.paragraphs
    ps[0].text = text
    for p in ps[1:]:
        p._element.getparent().remove(p._element)


def _para(doc, text: str):
    """Body paragraph with 6 pt after; a short lead-in label ("Protocol. ...") is set in bold."""
    para = doc.add_paragraph(style="normal")
    para.paragraph_format.space_after = Pt(6)
    head, sep, rest = text.partition(". ")
    if sep and len(head.split()) <= 5 and not head[:1].isdigit():
        para.add_run(head + ".").bold = True
        para.add_run(" " + rest)
    else:
        para.add_run(text)
    return para


def _borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = docx.oxml.OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge, size in (("top", 8), ("bottom", 8), ("insideH", 4)):
        el = docx.oxml.OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:color"), "8F8E89")
        borders.append(el)


def _table(doc, kind: str):
    if kind == "fieldguard":
        head = ["Fabricator", "Paradata", "Flagged", "Recall (95% CI)", "Honest flagged", "AUC"]
        rows = [
            ["Rule-based", "answers only", "5/9", "0.56 (0.27-0.81)", "0/81", "1.00"],
            ["AI, S0 naive", "answers only", "0/9", "0.00 (0.00-0.30)", "0/81", "0.86"],
            ["AI, S1 informed", "answers only", "0/9", "0.00 (0.00-0.30)", "0/81", "0.89"],
            ["AI, S2 careful", "answers only", "0/7", "0.00 (0.00-0.35)", "0/83", "1.00"],
            ["Rule-based", "with traces", "9/9", "1.00 (0.70-1.00)", "0/81", "1.00"],
            ["AI, all strategies", "with traces", "25/25", "1.00 (0.87-1.00)", "0/245", "1.00"],
        ]
    else:
        head = ["Source", "Interviews", "Rating SD within interview", "Incomes with at most 2 significant digits",
                "SD of log food share", "SD of household size"]
        rows = [
            ["AI, S0 naive", "166", "0.83", "87%", "0.22", "1.60"],
            ["AI, S1 informed", "166", "0.59", "72%", "0.23", "1.89"],
            ["AI, S2 careful", "124", "0.62", "97%", "0.28", "1.84"],
            ["Honest", "2,209", "1.04", "61%", "0.33", "2.04"],
        ]
    t = doc.add_table(rows=1 + len(rows), cols=len(head))
    t.style = doc.styles["Table Grid"] if "Table Grid" in [s.name for s in doc.styles] else t.style
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(head):
        c = t.rows[0].cells[i]
        c.text = h
        for r in c.paragraphs[0].runs:
            r.bold = True
            r.font.size = Pt(8.5)
    for ri, row in enumerate(rows, start=1):
        for ci, v in enumerate(row):
            c = t.rows[ri].cells[ci]
            c.text = v
            for r in c.paragraphs[0].runs:
                r.font.size = Pt(8.5)
    _borders(t)
    return t


def build() -> Path:
    doc = docx.Document(str(TEMPLATE))
    body = doc.element.body
    title_tbl = doc.tables[0]
    # Title row: keep the paragraph with the footnote reference, replace the visible text.
    title_p = title_tbl.rows[0].cells[0].paragraphs[1]
    runs = title_p.runs
    keep = [r for r in runs if r._element.find(qn("w:footnoteReference")) is not None]
    for r in runs:
        if r not in keep:
            r._element.getparent().remove(r._element)
    new = title_p.add_run(TITLE)
    if keep:
        title_p._element.remove(new._element)
        keep[0]._element.addprevious(new._element)
    # Row 2: authors grid (first nested table) and abstract box (second nested table).
    cell = title_tbl.rows[1].cells[0]
    authors, abstract_box = cell.tables[0], cell.tables[1]
    a_cell = authors.rows[0].cells[0]
    for nested in a_cell._tc.findall(qn("w:tbl")):  # the template's placeholder grid of six authors
        a_cell._tc.remove(nested)
    _set_cell_text(a_cell, "")
    p = a_cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(AUTHOR)
    r.bold = True
    p2 = a_cell.add_paragraph(AFFILIATION)
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p3 = a_cell.add_paragraph("With Apart Research")
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # drop the other author columns
    tr = authors.rows[0]._tr
    tcs = tr.findall(qn("w:tc"))
    for tc in tcs[1:]:
        tr.remove(tc)
    grid = authors._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        cols = grid.findall(qn("w:gridCol"))
        for gc in cols[1:]:
            grid.remove(gc)
    _set_cell_text(abstract_box.rows[0].cells[0], ABSTRACT)
    # Body: drop the info box and all guidance after the title table.
    _clear_after(body, title_tbl._tbl)
    for kind, payload in BODY:
        if kind == "h2":
            doc.add_paragraph(payload, style="Heading 2")
        elif kind == "h3":
            doc.add_paragraph(payload, style="Heading 3")
        elif kind == "p":
            _para(doc, payload)
        elif kind in ("bullets", "numbered", "refs"):
            for i, item in enumerate(payload, start=1):
                lead = "• " if kind == "bullets" else f"{i}. " if kind == "numbered" else f"[{i}] "
                para = doc.add_paragraph(lead + item, style="normal")
                para.paragraph_format.left_indent = Inches(0.25)
                para.paragraph_format.first_line_indent = Inches(-0.2)
                if kind == "refs":
                    for run in para.runs:
                        run.font.size = Pt(9)
        elif kind == "figure":
            doc.add_picture(str(HERE / payload), width=Inches(6.2))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif kind == "table":
            _table(doc, payload)
        elif kind == "caption":
            para = doc.add_paragraph(payload, style="normal")
            para.paragraph_format.space_after = Pt(10)
            for run in para.runs:
                run.italic = True
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)
    doc.core_properties.title = TITLE
    doc.core_properties.author = AUTHOR
    doc.save(str(OUT))
    return OUT


if __name__ == "__main__":
    print(build())

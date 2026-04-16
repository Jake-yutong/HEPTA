"""One-off script to generate the HEPTA data files in data/."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. test_questions.xlsx — 9 questions (3 Breadth, 3 Methods, 3 Depth)
# ─────────────────────────────────────────────────────────────

questions = [
    # Breadth — OBJ, CE, HCP
    {
        "id": "B1",
        "area": "Breadth",
        "dimension": "OBJ",
        "text": "Which of the following best describes the 'Gulf of Execution' as introduced by Hutchins, Hollan, and Norman (1985)? (A) The gap between a user's goal and the physical actions required by the interface. (B) The difference between system output and user expectations. (C) The delay between user input and system response. (D) The mismatch between a user's mental model and the designer's conceptual model.",
        "reference_answer": "A",
        "max_score": 100,
    },
    {
        "id": "B2",
        "area": "Breadth",
        "dimension": "CE",
        "text": "Explain the concept of 'Direct Manipulation' as described by Hutchins, Hollan, and Norman (1985). How does it relate to the gulfs of execution and evaluation?",
        "reference_answer": "Direct Manipulation refers to an interaction style where users operate directly on visible objects in the interface, producing immediate and reversible results. It bridges the Gulf of Execution by making actions intuitively map to intentions, and the Gulf of Evaluation by providing continuous visual feedback so users can readily assess the system state.",
        "max_score": 100,
    },
    {
        "id": "B3",
        "area": "Breadth",
        "dimension": "HCP",
        "text": "Discuss how Vannevar Bush's 'As We May Think' (1945) anticipated modern hypertext and information retrieval systems. What ideas persisted, and what did not?",
        "reference_answer": "Bush envisioned the Memex, a mechanised desk with microfilm storage allowing associative trails between documents — a precursor to hypertext and the World Wide Web. The persistence of associative linking is evident in modern hyperlinks and knowledge graphs. However, the Memex's reliance on analogue microfilm and its assumption of a single-user device did not persist; instead, digital, networked, and collaborative systems emerged.",
        "max_score": 100,
    },
    # Methods — MA, TA, OBJ
    {
        "id": "M1",
        "area": "Methods",
        "dimension": "MA",
        "text": "A researcher conducted a within-subjects study with 20 participants comparing three text-entry techniques. Which statistical test is most appropriate to analyse the task completion times, and why?",
        "reference_answer": "A repeated-measures one-way ANOVA is most appropriate because (1) there is one independent variable (text-entry technique) with three levels, (2) the same participants appear in all conditions (within-subjects), and (3) the dependent variable (task completion time) is continuous. Post-hoc pairwise comparisons (e.g., Bonferroni-corrected paired t-tests) should follow if the ANOVA is significant.",
        "max_score": 100,
    },
    {
        "id": "M2",
        "area": "Methods",
        "dimension": "TA",
        "text": "Compare the trade-offs between laboratory experiments and field studies in HCI research. Under what circumstances would you choose one over the other?",
        "reference_answer": "Laboratory experiments offer high internal validity through controlled conditions, but sacrifice ecological validity — results may not generalise to real-world settings. Field studies capture authentic usage contexts, yielding high ecological validity, but introduce confounding variables that threaten internal validity. A lab study is preferred when establishing causal mechanisms of a novel interaction technique; a field study is preferred when understanding how a deployed system is adopted and appropriated in practice.",
        "max_score": 100,
    },
    {
        "id": "M3",
        "area": "Methods",
        "dimension": "OBJ",
        "text": "In a 2×3 mixed-design ANOVA, which of the following correctly describes the design? (A) Two within-subjects factors, each with three levels. (B) One between-subjects factor with 2 levels and one within-subjects factor with 3 levels. (C) Two between-subjects factors with 2 and 3 levels respectively. (D) A non-parametric alternative to repeated measures.",
        "reference_answer": "B",
        "max_score": 100,
    },
    # Depth — TE, CPI, TE
    {
        "id": "D1",
        "area": "Depth",
        "dimension": "TE",
        "text": "Explain the technical approach behind von Ahn and Dabbish's ESP Game (CHI 2004) for labelling images. How does the game mechanic ensure label quality?",
        "reference_answer": "The ESP Game pairs two anonymous players who independently type labels for the same image. A label counts only when both players agree (match), leveraging inter-annotator agreement as a quality filter. Taboo words (previously agreed-upon labels) force players to produce increasingly specific and diverse labels. This turns human computation into an engaging game while producing high-quality image annotations at scale.",
        "max_score": 100,
    },
    {
        "id": "D2",
        "area": "Depth",
        "dimension": "CPI",
        "text": "Integrate insights from Erickson & Kellogg's 'Social Translucence' (2000) and Grudin's 'Eight Challenges for Groupware Developers' (1994) to discuss how system design can address the free-rider problem in collaborative software.",
        "reference_answer": "Grudin identifies the disparity between who does the work and who benefits as a core groupware challenge — directly relevant to free-riding. Erickson and Kellogg's Social Translucence proposes making social activity visible (visibility), understandable (awareness), and subject to social accountability. By surfacing individual contributions and participation patterns, a translucent system lets group members observe who contributes, apply social pressure to free-riders, and reward active contributors, thereby aligning incentives. Together, these papers argue that purely technical solutions are insufficient; sociotechnical design — making effort and benefit mutually apparent — is essential.",
        "max_score": 100,
    },
    {
        "id": "D3",
        "area": "Depth",
        "dimension": "TE",
        "text": "Describe the SUPPLE system (Gajos & Weld) for automatic UI generation. What optimisation problem does it solve, and what are its inputs and outputs?",
        "reference_answer": "SUPPLE models UI generation as a combinatorial optimisation problem. Inputs: (1) an interface specification describing the widgets and their possible renderings, (2) a device model capturing screen size and input capabilities, and (3) a user/usage model encoding interaction preferences and trace data. Output: a concrete UI layout that minimises a cost function representing expected user effort. SUPPLE uses branch-and-bound search to find the optimal rendering, enabling personalised and device-adapted interfaces.",
        "max_score": 100,
    },
]

df_q = pd.DataFrame(questions)
df_q.to_excel(DATA_DIR / "test_questions.xlsx", index=False)
print(f"Created {DATA_DIR / 'test_questions.xlsx'} ({len(df_q)} questions)")


# ─────────────────────────────────────────────────────────────
# 2. rubric.xlsx — seven dimensions
# ─────────────────────────────────────────────────────────────

rubric = [
    {
        "dimension": "OBJ",
        "criteria": (
            "Correctness of the selected option. Full marks (100) for the correct answer; "
            "0 for an incorrect answer. Partial credit (50) if the student provides a correct "
            "justification but selects the wrong option."
        ),
        "max_score": 100,
    },
    {
        "dimension": "CE",
        "criteria": (
            "Accuracy and depth of the conceptual explanation. 80–100: precise definition with "
            "concrete examples and connections to related concepts. 50–79: generally correct but "
            "lacks depth or examples. 20–49: partially correct with notable misconceptions. "
            "0–19: largely incorrect or irrelevant."
        ),
        "max_score": 100,
    },
    {
        "dimension": "TA",
        "criteria": (
            "Identification of competing design considerations, articulation of trade-offs, and "
            "reasoning about when each alternative is preferable. 80–100: identifies multiple "
            "trade-offs with nuanced contextual reasoning. 50–79: identifies trade-offs but "
            "reasoning is surface-level. 20–49: mentions only one side. 0–19: fails to identify "
            "any trade-off."
        ),
        "max_score": 100,
    },
    {
        "dimension": "HCP",
        "criteria": (
            "Demonstrates understanding of the historical evolution and lasting impact of the "
            "idea. 80–100: traces the lineage of the concept, explains why certain ideas persisted, "
            "and discusses what did not persist and why. 50–79: correct historical placement but "
            "thin on persistence analysis. 20–49: vague or partially correct. 0–19: historically "
            "inaccurate."
        ),
        "max_score": 100,
    },
    {
        "dimension": "TE",
        "criteria": (
            "Clarity and correctness of the technical description. 80–100: precise explanation of "
            "the algorithm/system architecture with correct terminology. 50–79: correct overview "
            "but missing important technical details. 20–49: partially correct with significant "
            "technical errors. 0–19: fundamentally incorrect."
        ),
        "max_score": 100,
    },
    {
        "dimension": "CPI",
        "criteria": (
            "Ability to synthesise insights across multiple readings. 80–100: draws substantive "
            "connections between two or more papers, identifies complementary or conflicting "
            "arguments, and produces an integrated response. 50–79: mentions multiple papers but "
            "connections are superficial. 20–49: discusses papers in isolation. 0–19: fails to "
            "reference relevant readings."
        ),
        "max_score": 100,
    },
    {
        "dimension": "MA",
        "criteria": (
            "Correct selection and justification of the research method or statistical procedure. "
            "80–100: selects the appropriate method, explains why it fits the study design, and "
            "notes assumptions/limitations. 50–79: selects an appropriate method but justification "
            "is incomplete. 20–49: selects a plausible but sub-optimal method. 0–19: selects an "
            "inappropriate method."
        ),
        "max_score": 100,
    },
]

df_r = pd.DataFrame(rubric)
df_r.to_excel(DATA_DIR / "rubric.xlsx", index=False)
print(f"Created {DATA_DIR / 'rubric.xlsx'} ({len(df_r)} items)")


# ─────────────────────────────────────────────────────────────
# 3. knowledge_base.xlsx — 50+ training entries
# ─────────────────────────────────────────────────────────────

kb_entries = [
    # Breadth — Foundations
    {"id": "KB01", "area": "Breadth", "topic": "Memex", "source": "Bush 1945", "content": "Vannevar Bush envisioned the Memex, a mechanised desk enabling associative trails between microfilm documents, anticipating hypertext."},
    {"id": "KB02", "area": "Breadth", "topic": "Man-Computer Symbiosis", "source": "Licklider 1960", "content": "Licklider proposed a cooperative interaction between humans and computers where each contributes its strengths — humans for goal-setting and judgment, computers for data processing."},
    {"id": "KB03", "area": "Breadth", "topic": "Sketchpad", "source": "Sutherland 1963", "content": "Sketchpad introduced direct graphical interaction with a computer using a light pen, pioneering object-oriented graphics and constraint-based design."},
    {"id": "KB04", "area": "Breadth", "topic": "The Mother of All Demos", "source": "Engelbart 1968", "content": "Engelbart demonstrated real-time collaborative editing, hypertext, the mouse, and video conferencing — decades before mainstream adoption."},
    {"id": "KB05", "area": "Breadth", "topic": "Sciences of the Artificial", "source": "Simon 1969", "content": "Herbert Simon framed design as the science of the artificial, distinguishing it from natural science and emphasising satisficing over optimising."},
    {"id": "KB06", "area": "Breadth", "topic": "Human Information Processor", "source": "Card Moran Newell 1983", "content": "The Model Human Processor decomposes human cognition into perceptual, cognitive, and motor subsystems with quantifiable cycle times and memory parameters."},
    {"id": "KB07", "area": "Breadth", "topic": "GOMS", "source": "Card Moran Newell 1983", "content": "GOMS (Goals, Operators, Methods, Selection rules) models skilled performance by decomposing tasks into goal hierarchies and predicting execution time."},
    {"id": "KB08", "area": "Breadth", "topic": "Keystroke-Level Model", "source": "Card Moran Newell 1983", "content": "KLM is a simplified GOMS variant that estimates task time by summing keystroke, pointing, homing, drawing, mental-preparation, and system-response operators."},
    {"id": "KB09", "area": "Breadth", "topic": "Direct Manipulation", "source": "Hutchins Hollan Norman 1985", "content": "Direct manipulation interfaces reduce the psychological distance between user intentions and system actions by providing continuous representations and rapid, reversible feedback."},
    {"id": "KB10", "area": "Breadth", "topic": "User Technology", "source": "Card Moran 1986", "content": "Card and Moran trace the evolution from pointing tasks to complex cognitive activities, arguing HCI must expand its models from motor-level to knowledge-level."},
    {"id": "KB11", "area": "Breadth", "topic": "Understanding Computers and Cognition", "source": "Winograd Flores 1986", "content": "Winograd and Flores challenge the rationalistic view of cognition, drawing on Heidegger and Maturana to argue that computers should support human understanding rather than replace it."},
    {"id": "KB12", "area": "Breadth", "topic": "Situated Actions", "source": "Suchman 1987", "content": "Suchman argues that human activity is fundamentally situated; plans are resources for action, not deterministic programs — challenging the planning model of AI."},
    {"id": "KB13", "area": "Breadth", "topic": "Design of Everyday Things", "source": "Norman 1988", "content": "Norman introduces affordances, signifiers, mappings, and feedback as fundamental design principles, and analyses the psychopathology of everyday usability failures."},
    {"id": "KB14", "area": "Breadth", "topic": "Xerox Star Retrospective", "source": "Johnson et al. 1989", "content": "The Star pioneered the desktop metaphor, WYSIWYG editing, icons, and Ethernet connectivity. Commercial failure despite technical innovation revealed importance of cost and ecosystem."},
    {"id": "KB15", "area": "Breadth", "topic": "Ubiquitous Computing", "source": "Weiser 1991", "content": "Weiser envisioned computing seamlessly embedded in the environment — 'the most profound technologies are those that disappear'."},
    {"id": "KB16", "area": "Breadth", "topic": "Beyond Being There", "source": "Hollan Stornetta 1992", "content": "Rather than replicating face-to-face interaction, remote collaboration tools should exploit unique advantages of technology that surpass physical co-presence."},
    {"id": "KB17", "area": "Breadth", "topic": "Media Equation", "source": "Nass Reeves 1996", "content": "People unconsciously apply social rules (politeness, reciprocity) to computers and media, treating them as social actors."},
    {"id": "KB18", "area": "Breadth", "topic": "Tangible Bits", "source": "Ishii Ullmer 1997", "content": "Tangible Bits bridges the gap between the physical and digital by coupling digital information to everyday physical objects and architectural surfaces."},
    {"id": "KB19", "area": "Breadth", "topic": "Distance Matters", "source": "Olson Olson 2000", "content": "Despite advances in technology, geographic distance still impacts collaboration effectiveness due to loss of common ground, coupling requirements, and timezone differences."},
    {"id": "KB20", "area": "Breadth", "topic": "Feminist HCI", "source": "Bardzell 2010", "content": "Bardzell proposes feminist qualities (pluralism, embodiment, ecology, self-disclosure, advocacy, participation) as design and evaluation criteria for HCI."},
    {"id": "KB21", "area": "Breadth", "topic": "Ability-Based Design", "source": "Wobbrock et al. 2011", "content": "Ability-based design shifts the focus from disability to ability, designing systems that leverage what users can do rather than requiring specific abilities."},
    {"id": "KB22", "area": "Breadth", "topic": "Action Research in HCI", "source": "Hayes 2011", "content": "Hayes argues action research is a valid HCI methodology that combines practical problem-solving with theory generation through iterative cycles of planning, acting, and reflecting."},
    {"id": "KB23", "area": "Breadth", "topic": "Social Justice-Oriented Design", "source": "Dombrowski et al. 2016", "content": "Interaction design should foreground the needs of marginalised communities, transformation of unjust structures, and recognition of systemic inequity."},
    {"id": "KB24", "area": "Breadth", "topic": "Critical Race Theory for HCI", "source": "Ogbonnaya-Ogburu et al. 2020", "content": "CRT provides HCI with tools to examine how race and racism are embedded in technology design, evaluation, and deployment."},
    # Breadth — Human-Centered Design
    {"id": "KB25", "area": "Breadth", "topic": "User-Centered Design Process", "source": "Preece et al.", "content": "UCD involves iterative cycles of requirements gathering, design alternatives, prototyping, and evaluation with real users throughout the process."},
    # Methods — Statistical
    {"id": "KB26", "area": "Methods", "topic": "Paired t-test", "source": "Wobbrock Coursera", "content": "Compares means of two related groups; appropriate for within-subjects designs with one IV at two levels and continuous DV. Assumes normality of differences."},
    {"id": "KB27", "area": "Methods", "topic": "Unpaired t-test", "source": "Wobbrock Coursera", "content": "Compares means of two independent groups; for between-subjects designs. Assumes independence, normality, and homogeneity of variance (Welch's t-test relaxes the last)."},
    {"id": "KB28", "area": "Methods", "topic": "One-Way ANOVA", "source": "Kutner et al.", "content": "Tests equality of three or more group means for a single factor. Partitions total variance into between-group and within-group components."},
    {"id": "KB29", "area": "Methods", "topic": "Repeated Measures ANOVA", "source": "Kutner et al.", "content": "Extension of one-way ANOVA for within-subjects designs; accounts for correlation between repeated measurements. Assumes sphericity (Mauchly test; correct with Greenhouse-Geisser)."},
    {"id": "KB30", "area": "Methods", "topic": "Two-Way ANOVA", "source": "Kutner et al.", "content": "Analyses two independent variables simultaneously, allowing detection of main effects and interaction effects."},
    {"id": "KB31", "area": "Methods", "topic": "Chi-Square Test", "source": "Wobbrock Coursera", "content": "Tests association between two categorical variables. Pearson's χ² compares observed to expected frequencies; Fisher's exact test used for small samples."},
    {"id": "KB32", "area": "Methods", "topic": "Linear Regression", "source": "Kutner et al.", "content": "Models the linear relationship between a continuous DV and one or more IVs. Assessed via R², residual analysis, and significance of coefficients."},
    {"id": "KB33", "area": "Methods", "topic": "Logistic Regression", "source": "Kutner et al.", "content": "Models the probability of a binary outcome as a function of predictor variables using the logit link function."},
    {"id": "KB34", "area": "Methods", "topic": "ANCOVA", "source": "Kutner et al.", "content": "Combines ANOVA with regression to control for a continuous covariate, adjusting group means for baseline differences."},
    {"id": "KB35", "area": "Methods", "topic": "Nonparametric Tests", "source": "Wobbrock Coursera", "content": "Rank-based alternatives (Wilcoxon, Mann-Whitney, Kruskal-Wallis, Friedman) used when normality or interval-scale assumptions are violated."},
    # Methods — Research Design
    {"id": "KB36", "area": "Methods", "topic": "Experimental Design", "source": "McGrath", "content": "Key design decisions: randomisation, factorial design, between vs. within subjects, controls, and addressing threats to internal/external validity."},
    {"id": "KB37", "area": "Methods", "topic": "Qualitative Research", "source": "Lofland et al.", "content": "Systematic observation, interviewing, and analysis of field data. Grounded theory coding transforms observations into theoretical categories."},
    {"id": "KB38", "area": "Methods", "topic": "Methodology Matters", "source": "McGrath", "content": "No single method can maximise generalisability, precision, and realism simultaneously; researchers must make trade-offs."},
    # Depth — Social Computing
    {"id": "KB39", "area": "Depth", "topic": "Eight Challenges for Groupware", "source": "Grudin 1994", "content": "Grudin identifies challenges including work-benefit asymmetry, critical mass, disruption of social processes, and difficulty of evaluation in groupware design."},
    {"id": "KB40", "area": "Depth", "topic": "Social Translucence", "source": "Erickson Kellogg 2000", "content": "Social translucence supports social processes by providing visibility (seeing others' activity), awareness (understanding it), and accountability (social consequences)."},
    {"id": "KB41", "area": "Depth", "topic": "Intellectual Challenge of CSCW", "source": "Ackerman 2000", "content": "Ackerman highlights the socio-technical gap: the divide between what social requirements demand and what technology can currently support."},
    {"id": "KB42", "area": "Depth", "topic": "Benefits of Facebook Friends", "source": "Ellison et al. 2007", "content": "Facebook use is associated with social capital — especially bridging social capital — and maintaining weak ties through lightweight interactions."},
    {"id": "KB43", "area": "Depth", "topic": "ESP Game / Image Labelling", "source": "von Ahn Dabbish 2004", "content": "A two-player game producing image labels via consensus. Taboo words force specificity. Pioneered Games With A Purpose (GWAP) / human computation."},
    {"id": "KB44", "area": "Depth", "topic": "Social Psychology and Online Communities", "source": "Beenen et al. 2006", "content": "Applied uniqueness and goal-setting theories from social psychology to increase contributions to online communities (MovieLens experiment)."},
    {"id": "KB45", "area": "Depth", "topic": "Disinformation as Collaborative Work", "source": "Starbird et al. 2019", "content": "Strategic information operations are collaborative and distributed; understanding them requires CSCW lenses of participation and coordination."},
    # Depth — Design
    {"id": "KB46", "area": "Depth", "topic": "Reflective Practitioner", "source": "Schon 1983", "content": "Professional designers engage in reflection-in-action — a conversational, iterative process of framing problems and evaluating moves."},
    {"id": "KB47", "area": "Depth", "topic": "A Pattern Language", "source": "Alexander 1977", "content": "Alexander's pattern language provides reusable solutions to recurring design problems, organised in a hierarchical structure from towns to rooms."},
    {"id": "KB48", "area": "Depth", "topic": "Participatory Design", "source": "Schuler Namioka 1993", "content": "Users participate as co-designers throughout the development process, contributing domain expertise and ensuring designs meet real needs."},
    {"id": "KB49", "area": "Depth", "topic": "Research Through Design", "source": "Zimmerman Forlizzi", "content": "Design artefacts can embody research contributions by integrating theory and practice into a 'right' thing that reframes a design situation."},
    # Depth — Human-AI Interaction
    {"id": "KB50", "area": "Depth", "topic": "Mixed-Initiative Interfaces", "source": "Horvitz 1999", "content": "Principles for balancing automated action and user control: maintain user awareness, provide graceful degradation, and support efficient human override."},
    {"id": "KB51", "area": "Depth", "topic": "Direct Manipulation vs. Interface Agents", "source": "Shneiderman vs. Maes", "content": "Shneiderman advocates direct manipulation for user control; Maes argues intelligent agents can manage complexity. The tension shapes modern AI-assisted interfaces."},
    {"id": "KB52", "area": "Depth", "topic": "Interactive Machine Learning", "source": "Fails Olsen 2003", "content": "Users iteratively train classifiers by providing examples and observing results, enabling non-experts to build personalised ML models."},
    {"id": "KB53", "area": "Depth", "topic": "Difficulty of Designing Human-AI Interaction", "source": "Yang et al. 2020", "content": "HAI interaction is uniquely difficult because AI capabilities are uncertain, AI outputs are often opaque, and user mental models of AI are frequently miscalibrated."},
    {"id": "KB54", "area": "Depth", "topic": "Jury Learning", "source": "Gordon et al. 2022", "content": "Jury learning aggregates diverse human judgments (including dissenting views) into ML models, improving fairness and representation of minority perspectives."},
]

df_kb = pd.DataFrame(kb_entries)
df_kb.to_excel(DATA_DIR / "knowledge_base.xlsx", index=False)
print(f"Created {DATA_DIR / 'knowledge_base.xlsx'} ({len(df_kb)} entries)")

print("Done.")

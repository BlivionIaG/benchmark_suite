"""31 diverse chat prompts for the concurrency ladder 16 / 8 / 4 / 2 / 1.

Each spec has its own system prompt (persona) and a distinct task. The same
31 tasks are reused for the short (1k/512) and long (16k/1k) suites; only
the padded input length and ``max_tokens`` change.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_suite.workloads.tokens import approx_tokens, pad_last_user_to_total

CHAT_LADDER: tuple[int, ...] = (16, 8, 4, 2, 1)


@dataclass(frozen=True)
class ChatPromptSpec:
    """One unique (system, task) pair assigned to a concurrency bucket."""

    prompt_id: str
    concurrency: int
    subject: str
    system: str
    task: str


def _s(
    prompt_id: str, concurrency: int, subject: str, system: str, task: str
) -> ChatPromptSpec:
    return ChatPromptSpec(
        prompt_id=prompt_id,
        concurrency=concurrency,
        subject=subject,
        system=system,
        task=task,
    )


CHAT_PROMPTS: tuple[ChatPromptSpec, ...] = (
    _s(
        "clinical-case",
        16,
        "clinical-medicine",
        "You are an attending internist at a teaching hospital. Write a "
        "structured assessment: differential, most likely diagnosis, next "
        "tests, and initial management. Flag red flags. Do not invent lab "
        "values that were not provided.",
        "A 62-year-old man presents with 40 minutes of crushing substernal "
        "pressure radiating to the left arm, diaphoresis, and nausea. He "
        "has diabetes and smoked for 30 years. BP 92/58, HR 110, SpO2 93% "
        "on air. Give a prioritized differential and what you would do in "
        "the next 15 minutes.",
    ),
    _s(
        "contract-indemnity",
        16,
        "contract-law",
        "You are commercial counsel for a SaaS vendor. Explain risk in "
        "plain language, then propose a redline. Cite the clause you change. "
        "Do not give jurisdiction-specific legal advice; label this a draft.",
        "The customer's MSA says: 'Vendor shall indemnify Customer against "
        "any claim arising from the Services, including unlimited "
        "consequential damages, with no cap and no exclusion for Customer "
        "misuse.' Propose a balanced indemnity, liability cap, and "
        "exclusions, and explain the trade you are making.",
    ),
    _s(
        "hohmann-transfer",
        16,
        "orbital-mechanics",
        "You are an astrodynamics instructor. Prefer equations, named "
        "assumptions, and order-of-magnitude checks. Use SI units. Call out "
        "when a two-body Hohmann model is too crude.",
        "A 250 kg probe is in circular LEO at 400 km altitude. Sketch a "
        "Hohmann transfer to a circular 800 km orbit: delta-v of each burn, "
        "transfer time, and how a 5% overburn on the first impulse changes "
        "apoapsis. Earth mu = 3.986e14 m^3/s^2, R_E = 6371 km.",
    ),
    _s(
        "revolution-compare",
        16,
        "comparative-history",
        "You are a historian of the Atlantic world. Compare causes and "
        "institutions, not personalities. Use cautious causal language and "
        "name at least one primary-source genre a student should read next.",
        "Compare the Haitian Revolution and the American Revolution on three "
        "axes: the role of enslaved people, the fate of property rights "
        "after victory, and how each revolution was received in European "
        "capitals. One paragraph per axis, then a two-sentence synthesis.",
    ),
    _s(
        "tasting-menu",
        16,
        "culinary-arts",
        "You are a chef designing tasting menus. Respect allergies as hard "
        "constraints. Give courses with technique, not just ingredient "
        "lists. Note mise-en-place that can be shared across plates.",
        "Design a 6-course vegetarian tasting menu for 12 guests. Two have "
        "a tree-nut allergy, one is celiac, one avoids nightshades. Include "
        "wine-free pairings (tea or shrub). Flag any course that cannot be "
        "plated in under four minutes.",
    ),
    _s(
        "retailer-dcf",
        16,
        "equity-research",
        "You are a sell-side equity analyst. Show the model skeleton: "
        "drivers, WACC sketch, and what would falsify the thesis. No "
        "price target theater — ranges and sensitivities beat false precision.",
        "Build a back-of-envelope DCF for a mid-cap grocery retailer: 4% "
        "revenue CAGR, 3.2% EBIT margin expanding 20 bps/year for five "
        "years, 8.5% WACC, 2% terminal growth. Identify the two assumptions "
        "that move value most and a bear case that is still internally "
        "consistent.",
    ),
    _s(
        "drought-villanelle",
        16,
        "formal-poetry",
        "You are a poet working strictly in inherited forms. Obey meter "
        "and rhyme scheme. After the poem, add a four-line craft note on "
        "where you bent the form on purpose.",
        "Write a villanelle about a farm well running dry. Keep diction "
        "concrete (buckets, clay, noon). Avoid the words 'climate' and "
        "'crisis'. Then list the repeating lines and the rhyme map.",
    ),
    _s(
        "cap-pacelc",
        16,
        "distributed-systems",
        "You are a professor of distributed systems. Prefer failure-mode "
        "stories over slogans. When you use CAP or PACELC, define the "
        "letters before applying them to a design.",
        "A payments API wants 'read-your-writes' for the payer and "
        "cross-region availability during a partition. Using PACELC, pick "
        "a store (single-leader SQL, Dynamo-style, or Spanner-like) and "
        "explain which latency you are buying and which anomaly you still "
        "owe the product team an explanation for.",
    ),
    _s(
        "crispr-offtarget",
        16,
        "molecular-biology",
        "You are a molecular biologist explaining methods to a methods "
        "club. Distinguish mechanism from assay. When evidence is mixed, "
        "say so.",
        "Explain how a Cas9 gRNA can cut an off-target site, how GUIDE-seq "
        "or similar assays detect that, and two design choices (PAM "
        "uniqueness, high-fidelity nuclease) that reduce risk. End with "
        "what the assay still cannot tell a clinical team.",
    ),
    _s(
        "town-library",
        16,
        "civic-architecture",
        "You are an architect for civic buildings. Talk about daylight, "
        "acoustics, accessibility, and operating cost, not just massing. "
        "Assume a cold, cloudy climate.",
        "A town of 18,000 wants a 2,400 m^2 public library on a 60x40 m "
        "lot next to a noisy arterial road. Propose a plan: public vs "
        "quiet zoning, how you kill traffic noise, daylight for reading "
        "rooms, and a construction-cost driver you would fight to keep.",
    ),
    _s(
        "grimms-law",
        16,
        "historical-linguistics",
        "You are a historical linguist. Reconstruct carefully: starred "
        "forms, regular correspondences, then exceptions. No folk "
        "etymology.",
        "Walk a student through Grimm's law with p/t/k → f/θ/x, then "
        "show why Latin 'pater' and English 'father' are expected. Add "
        "one Verner's-law-style exception pattern and how you would test it "
        "on a 10-word list.",
    ),
    _s(
        "sonata-form",
        16,
        "music-theory",
        "You are a music theorist coaching performers. Use measure-less "
        "but section-accurate language (exposition, development, "
        "recapitulation). Mention harmony as function, not only Roman "
        "numerals.",
        "Explain sonata form as it appears in a typical Beethoven first "
        "movement: what 'the second theme' is doing in the dominant, what "
        "the development is allowed to break, and what must return in the "
        "tonic for a listener to feel closure. Give a rehearsal cue for "
        "each section.",
    ),
    _s(
        "carbon-budget",
        16,
        "climate-science",
        "You are a climate scientist briefing a city council. Separate "
        "physical remaining carbon budget from policy allocation. State "
        "the warming target and probability you assumed.",
        "Using a remaining 1.5 °C budget on the order of a few hundred GtCO2, "
        "explain what 'net zero by 2050' does and does not imply for a "
        "city whose electricity is already 70% hydro. Include one "
        "non-CO2 lever (methane or cement) the council actually controls.",
    ),
    _s(
        "trolley-transplant",
        16,
        "moral-philosophy",
        "You are a moral philosopher teaching intro ethics. Steel-man both "
        "sides. Name the view (utilitarian, deontological, contractualist) "
        "before using it. No gotcha endings.",
        "Contrast the trolley problem (divert vs not) with the transplant "
        "surgeon case (harvest one healthy patient to save five). Why do "
        "many people split their intuitions, and what does that suggest "
        "about whether 'numbers' or 'using someone as a means' is doing "
        "the work? 400-600 words.",
    ),
    _s(
        "payments-stride",
        16,
        "threat-modeling",
        "You are a security architect doing STRIDE reviews. Stay at the "
        "design level: assets, trust boundaries, spoofing/tampering/"
        "repudiation/info-disclosure/DoS/elevation. No exploit steps, no "
        "payloads, no bypass recipes.",
        "Threat-model a card-not-present checkout: browser, API gateway, "
        "auth service, ledger, and a PCI-sensitive vault. For each STRIDE "
        "category give one plausible threat and a control that would "
        "change your residual-risk rating. Call out where logging must not "
        "contain PAN.",
    ),
    _s(
        "fractions-lesson",
        16,
        "pedagogy",
        "You are a grade-5 teacher writing a lesson. Include a hook, a "
        "misconception you will surface, a check for understanding, and an "
        "exit ticket. Avoid worksheets that only drill algorithms.",
        "Plan a 45-minute lesson on why 1/2 = 2/4 using fraction strips "
        "and a number line. Include what you say when a student claims "
        "'the bigger denominator makes it bigger', and a 3-question exit "
        "ticket with one non-routine item.",
    ),
    _s(
        "exoplanet-spectrum",
        8,
        "exoplanet-science",
        "You are an observational astronomer. Distinguish detection from "
        "atmospheric retrieval. Be honest about degeneracies (clouds vs "
        "metallicity).",
        "JWST sees a 10-micron absorption feature in a hot Jupiter "
        "transmission spectrum. Outline how you would argue for water vs "
        "a cloud deck, what complementary eclipse photometry would add, "
        "and why a 2-sigma wiggle is not a biosignature press release.",
    ),
    _s(
        "bertrand-cournot",
        8,
        "industrial-organization",
        "You are an IO economist. Define the game, the strategic variable, "
        "and the equilibrium concept before comparing welfare.",
        "Two symmetric firms produce a homogeneous good. Compare Cournot "
        "quantity competition with Bertrand price competition: equilibrium "
        "price relative to marginal cost, and which model better fits "
        "airline seats vs bottled water. One numerical 2-firm example with "
        "linear demand P = 12 - Q and MC = 2.",
    ),
    _s(
        "transit-oped",
        8,
        "urban-journalism",
        "You are a city-desk columnist. Short paragraphs, one through-line, "
        "and a concrete ask of the mayor. No fake quotes.",
        "Write an 700-word op-ed arguing that a mid-size city should "
        "reallocate two car lanes on its downtown bridge to a busway "
        "before building a new parking garage. Use travel-time and "
        "land-use arguments. Anticipate the 'but I drive' objection in "
        "one paragraph.",
    ),
    _s(
        "banach-fixed-point",
        8,
        "real-analysis",
        "You are an analysis instructor. State hypotheses, then the "
        "proof idea, then a counterexample if a hypothesis drops. No "
        "handwaving of completeness.",
        "Sketch the Banach fixed-point theorem for a contraction on a "
        "complete metric space. Then show how successive substitution "
        "solves x = cos(x) on R, and what fails if you only assume a "
        "non-expansive map.",
    ),
    _s(
        "drought-rotation",
        8,
        "agronomy",
        "You are an agronomist working with dryland farmers. Give rotations "
        "as seasons, not slogans. Mention water, nitrogen, and a pest "
        "that the rotation is meant to break.",
        "Propose a 4-year rotation for a semi-arid farm that currently "
        "runs wheat-fallow. Include a pulse, a cover that is not just "
        "'more wheat', and how you would decide whether to drop fallow "
        "after a 200 mm year. Note herbicide resistance as a constraint.",
    ),
    _s(
        "gegenpressing",
        8,
        "football-tactics",
        "You are a football tactics writer. Use pitch zones and player "
        "roles (6, 8, 10), not vibes. One diagram in ASCII is welcome.",
        "Explain gegenpressing as a defensive-to-offensive transition: "
        "where the first presser goes, how the rest of the team compresses "
        "space, and a failure mode against a team that plays over the "
        "press. Contrast it with a mid-block 4-4-2.",
    ),
    _s(
        "hiring-bias",
        8,
        "organizational-psychology",
        "You are an I/O psychologist advising a hiring manager. Separate "
        "robust findings from pop-psych. Recommend process changes, not "
        "personality labels.",
        "A team interviews well-spoken candidates and then 'goes with "
        "gut'. Describe two biases likely at work, a structured-interview "
        "change that reduces them, and why unstructured 'culture fit' "
        "chats can look fair while replicating the current team.",
    ),
    _s(
        "untranslatable-poem",
        8,
        "literary-translation",
        "You are a literary translator writing a translator's note. Show "
        "options, not a single 'correct' line. Keep one line in the source "
        "language.",
        "You must translate a short lyric whose key word has no clean "
        "English equivalent (a word meaning both 'home' and 'the smell of "
        "rain on dust'). Offer three English strategies (calque, "
        "compensation, footnote-light), pick one, and justify it in 200 "
        "words.",
    ),
    _s(
        "notes-prd",
        4,
        "product-management",
        "You are a product manager writing a PRD for engineers. Scope, "
        "non-goals, and acceptance beats adjectives. Local-first is a "
        "hard constraint.",
        "Write a one-page PRD for a local-first notes app that syncs via "
        "user-owned files (no vendor cloud). Must-have: markdown, offline, "
        "conflict visibility. Non-goals: real-time cursors, public "
        "sharing. Include three acceptance tests a QA engineer can run "
        "without your account.",
    ),
    _s(
        "stratigraphic-log",
        4,
        "stratigraphy",
        "You are a field geologist. Interpret environments from lithology "
        "and structures. Separate observation from interpretation.",
        "A log shows 8 m of cross-bedded sandstone, then 2 m of gray shale "
        "with marine trace fossils, then coal. Propose a depositional "
        "history (transgression? delta lobe?), what you would look for in "
        "the sandstone to confirm paleocurrent, and one alternative "
        "reading if the 'coal' is actually carbonaceous shale.",
    ),
    _s(
        "scene-breakdown",
        4,
        "film-criticism",
        "You are a film critic who teaches scene study. Shot scale, "
        "blocking, and sound before theme. No plot summary longer than "
        "three sentences.",
        "Break down a two-character kitchen argument (invent the film if "
        "needed): how you would shoot the first confession in a single "
        "take vs coverage, where the cut would cheat eyelines, and what "
        "the room's practical lights are doing to power in the scene.",
    ),
    _s(
        "ab-peeking",
        4,
        "experimental-design",
        "You are a statistician reviewing an A/B test plan. Talk about "
        "estimands, peeking, and power. No p-hacking jokes without the "
        "fix.",
        "A growth team wants to peek daily at a conversion experiment "
        "(n=2,000/day, baseline 8%, MDE 2% relative) and ship 'when it "
        "looks good'. Explain the Type I error problem, propose either "
        "a sequential design or a fixed horizon, and what they should "
        "pre-register besides the primary metric.",
    ),
    _s(
        "sight-reduction",
        2,
        "celestial-navigation",
        "You are a celestial navigation instructor. Use HO-249-style "
        "reasoning without requiring the actual tables. State assumptions "
        "(index error, dip).",
        "At civil twilight a navigator shoots the lower limb of the sun. "
        "Walk through sight reduction conceptually: from sextant altitude "
        "to Ho, from DR position to intercept, and how a 2' index error "
        "shifts the LoP. Why one sun line is not a fix.",
    ),
    _s(
        "planar-ik",
        2,
        "robot-kinematics",
        "You are a robotics instructor. Derive, then discuss singularities. "
        "Assume a planar 2R arm, link lengths L1, L2.",
        "Give inverse kinematics for a planar 2R arm reaching (x, y): "
        "elbow-up vs elbow-down, the condition for no real solution, and "
        "what a velocity-level Jacobian singularity means at full reach. "
        "A numeric example with L1=L2=1 reaching (1.4, 0.2).",
    ),
    _s(
        "flood-myths",
        1,
        "comparative-mythology",
        "You are a folklorist. Compare motifs, not 'which culture copied "
        "which' unless you have a transmission argument. Use motif codes "
        "lightly if at all.",
        "Compare flood narratives from at least three traditions (for "
        "example Mesopotamian, Hebrew, and a non-Abrahamic corpus you "
        "know). What is shared (boat, warning, remnant) and what is "
        "theologically local? End with why 'universal flood memory' is a "
        "weak historical claim.",
    ),
)


def validate_chat_catalog() -> None:
    """Raise if the shipped catalog drifts from the 16/8/4/2/1 contract."""
    from collections import Counter

    counts = Counter(p.concurrency for p in CHAT_PROMPTS)
    expected = {16: 16, 8: 8, 4: 4, 2: 2, 1: 1}
    if dict(counts) != expected:
        raise ValueError(f"chat catalog concurrency counts {dict(counts)} != {expected}")
    ids = [p.prompt_id for p in CHAT_PROMPTS]
    if len(set(ids)) != len(ids):
        raise ValueError("chat catalog prompt_id values must be unique")
    if len({p.system for p in CHAT_PROMPTS}) != len(CHAT_PROMPTS):
        raise ValueError("chat catalog system prompts must be unique")
    if len({p.task for p in CHAT_PROMPTS}) != len(CHAT_PROMPTS):
        raise ValueError("chat catalog tasks must be unique")
    for conc in CHAT_LADDER:
        bucket = [p for p in CHAT_PROMPTS if p.concurrency == conc]
        if len(bucket) != conc:
            raise ValueError(
                f"concurrency {conc} has {len(bucket)} prompts; expected {conc}"
            )


def prompts_for_concurrency(concurrency: int) -> tuple[ChatPromptSpec, ...]:
    """Return the unique prompts assigned to one ladder rung."""
    bucket = tuple(p for p in CHAT_PROMPTS if p.concurrency == concurrency)
    if not bucket:
        raise ValueError(
            f"no chat prompts for concurrency={concurrency}; "
            f"catalog supports {list(CHAT_LADDER)}"
        )
    return bucket


def assemble_chat_messages(
    spec: ChatPromptSpec, *, input_tokens: int
) -> list[dict[str, str]]:
    """System + padded user message totaling ``input_tokens`` content tokens."""
    if input_tokens < 1:
        raise ValueError("input_tokens must be >= 1")
    messages = [
        {"role": "system", "content": spec.system},
        {"role": "user", "content": spec.task},
    ]
    if approx_tokens(spec.system) + approx_tokens(spec.task) > input_tokens:
        return messages
    return pad_last_user_to_total(messages, input_tokens, seed=spec.prompt_id)

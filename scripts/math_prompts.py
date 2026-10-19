"""System prompts for RLVR-math's analysis + confidence (2026-10-19), merged from our drafts and two outside
reviews (docs/math_prompt_brief.md). Used with analysis-only regeneration: each row's cached <think>...</answer> is
prefilled, so answers and labels are identical across prompts.

Every prompt but `verify_open` keeps tabc's opening (what the model was trained under) and differs only in the
analysis guide. All share: the answer is fixed, confidence words-only in the analysis (except `range`), a defined
calibrated confidence, the 11 one-decimal values, and the stop after </confidence>. `verify_open` tests a
restructured opening ("Output exactly these four sections") with the `verify` guide.
"""

HEAD = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its "
    "confidence about the solution and then provides the user with its confidence level. The final answer is enclosed "
    "between <answer> </answer> tags. The analysis about confidence and uncertainty is enclosed within "
    "<analysis> </analysis> tags. Once the final answer is written it is fixed: the analysis assesses it and never "
    "replaces or revises it.\n"
)

WORDS = (
    "In the analysis, express certainty and doubt in words only. Numbers that belong to the mathematics (values, "
    "equations, checks) are fine, but do not write a confidence value, a probability of being correct, a percentage "
    "or odds there: the confidence number appears only inside the confidence tags. Keep the analysis under 400 words.\n"
)

CONF = (
    "The confidence level is the probability that the final answer is exactly correct. It should be calibrated: among "
    "all answers given confidence 0.7, about 70% should be correct. Choose it from the evidence in the analysis. Write "
    "it as exactly one of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, alone on its own line inside "
    "<confidence> </confidence> tags."
)

STOP = " The response ends immediately after </confidence>; write nothing after it.\n"


def fmt(desc):
    return ("The final format that must be followed is: <think> reasoning process here </think><answer> final answer "
            f"here </answer> <analysis> {desc} </analysis> <confidence> one of 0.0, 0.1, ..., 1.0 </confidence>")


VERIFY = (
    "In the analysis, check the fixed answer against the original problem, using these labels:\n"
    "Check: perform the most decisive test available. Substitute the answer into the original equations or "
    "conditions, recompute the asked-for quantity from the problem's data, or test the claimed property. Show the "
    "calculation.\n"
    "Result: write PASS, FAIL, or INCONCLUSIVE, and say what the test establishes.\n"
    "Coverage: list the remaining requirements (domain restrictions, integer or sign conditions, units, excluded "
    "cases, whether all solutions were found, uniqueness) and check the relevant ones.\n"
    "Unresolved: say what remains unverified, or say that no specific gap was found.\n"
    "A passing check supports only what it tests: finding one valid solution does not show that all solutions were "
    "found. Report a disagreement even if the original reasoning looked convincing. Do not invent an error.\n"
)

GUIDES = {
    "control": ("The assistant should reason about its confidence in the solution and its uncertainty in the "
                "solution within these tags.\n", "analysis about confidence and uncertainty here"),

    "verify": (VERIFY, "Check, Result, Coverage, Unresolved"),

    "rederive": (
        "In the analysis, get a second, independent piece of evidence by doing fresh mathematical work, using these "
        "labels:\n"
        "Target: the final result, or the key intermediate quantity, to recompute.\n"
        "Method: a different route from the reasoning above where practical: direct enumeration instead of a "
        "formula, working backwards, a small case, a geometric instead of an algebraic argument, or a parity, "
        "modular or order-of-magnitude check.\n"
        "Work: carry out the computation from the problem's data. Do not use the final answer as an input.\n"
        "Comparison: write AGREES, DISAGREES, or INCONCLUSIVE.\n"
        "Shared dependency: name any assumption or calculation that both routes rely on. If the new work only "
        "repeats the original method, say so.\n"
        "If a full second solution would be too long, recompute only the most consequential intermediate result. If "
        "the routes disagree, do not change the answer: the disagreement is evidence against it.\n",
        "Target, Method, Work, Comparison, Shared dependency"),

    "weakest": (
        "In the analysis, audit the step whose failure would most threaten the answer, using these labels:\n"
        "Critical step: state the specific claim or calculation, not a generic concern.\n"
        "Dependency: explain how the final answer depends on it.\n"
        "Audit: check that step against the original problem: its assumptions, the conditions of any theorem or "
        "formula used, the algebra and the arithmetic.\n"
        "Finding: write VERIFIED, CONTRADICTED, or UNRESOLVED, and give the evidence.\n"
        "Remaining gap: name any other specific gap that could still change the answer, or say that none was found.\n"
        "If no step looks weak, audit the most consequential one anyway. A step is not verified because it is "
        "familiar or was stated confidently. Do not manufacture doubt.\n",
        "Critical step, Dependency, Audit, Finding, Remaining gap"),

    "falsify": (
        "In the analysis, try to find a concrete reason the fixed answer could be wrong, using these labels:\n"
        "Failure route: pick the one plausible error this solution is most exposed to, such as an omitted case, an "
        "extraneous root, an invalid domain assumption, a sign error, double counting, misreading what is asked, or "
        "an unjustified claim of uniqueness or optimality.\n"
        "Test: try to expose that error with a concrete calculation, boundary case, counterexample or condition "
        "check.\n"
        "Outcome: write ERROR FOUND, TEST PASSED, or INCONCLUSIVE, and show the evidence.\n"
        "Scope: say which error was tested and what the test leaves open.\n"
        "If an error is found, say whether it invalidates the answer or only the argument. A passed test rules out "
        "only that error, not every error. Do not invent a counterexample.\n",
        "Failure route, Test, Outcome, Scope"),

    "competing": (
        "In the analysis, compare the fixed answer with at most one concrete competing answer, using these labels:\n"
        "Alternative: a different result that a specific plausible error or unresolved choice in the reasoning would "
        "produce (another reading of the question, a boundary case, a different case split, a questionable "
        "simplification). Say how it arises and compute it.\n"
        "Distinguishing test: a condition of the problem, or an independent calculation, that tells the two apart.\n"
        "Work: perform that test.\n"
        "Finding: write ANSWER FAVORED, ALTERNATIVE FAVORED, or UNRESOLVED, and say why.\n"
        "If the reasoning supports no concrete alternative, write \"No concrete alternative identified\" and instead "
        "directly check one important requirement of the answer. Showing that an alternative is wrong does not by "
        "itself show that the answer is right.\n",
        "Alternative, Distinguishing test, Work, Finding"),

    "range": (
        VERIFY +
        "End the analysis with these two lines:\n"
        "Plausible confidence range: L to H\n"
        "Why both ends remain plausible: one sentence.\n"
        "L and H are two different values from 0.0, 0.1, ..., 1.0 with L below H. They bound the reasonable estimates "
        "of the probability that the answer is exactly correct: a wide range when the evidence is mixed or weak, a "
        "narrow one when it is clear. Do not pick a single value in the analysis.\n",
        "Check, Result, Coverage, Unresolved, the range line, and why both ends remain plausible"),

    "rubric": (
        "In the analysis, rate the reliability of the answer on four points, using these labels:\n"
        "Independent check: carry out a quick substitution, boundary check, or recomputation of the hardest step, "
        "then write [CONFIRMED], [INCONCLUSIVE], or [CONTRADICTED].\n"
        "Arithmetic and algebra risk: judge how long and error-prone the calculations were, then write [LOW], "
        "[MEDIUM], or [HIGH].\n"
        "Cases and edge-case risk: judge whether hidden cases, domain restrictions or an ambiguity in the question "
        "could invalidate the answer, then write [LOW], [MEDIUM], or [HIGH].\n"
        "Verdict: one sentence on how these weigh against each other.\n",
        "Independent check, Arithmetic and algebra risk, Cases and edge-case risk, Verdict"),
}

RANGE_WORDS = (
    "Apart from the range line, express certainty and doubt in words only. Numbers that belong to the mathematics are "
    "fine, but write no other confidence value, probability of being correct, percentage or odds in the analysis. "
    "Keep the analysis under 400 words.\n"
)

OPEN = (
    "Solve the user's math problem. Output exactly these four sections, once each, in this order:\n"
    "<think> reasoning </think>\n<answer> final answer </answer>\n<analysis> assessment of the fixed answer "
    "</analysis>\n<confidence>\nvalue\n</confidence>\n"
    "Once </answer> has appeared, the answer is fixed. Do not replace or revise it.\n"
)

NAMES = list(GUIDES) + ["verify_open"]


def build(name):
    if name == "verify_open":
        return OPEN + VERIFY + WORDS + CONF + STOP
    guide, desc = GUIDES[name]
    words = RANGE_WORDS if name == "range" else WORDS
    conf = CONF + (" It should lie within the stated range." if name == "range" else "")
    return HEAD + guide + words + conf + STOP + fmt(desc)


if __name__ == "__main__":
    for name in NAMES:
        p = build(name)
        print(f"===== {name} ({len(p.split())} words)\n{p}\n")

"""Candidate system prompts for RLVR-math's analysis + confidence (draft, 2026-10-19).

Each prompt is HEAD + a variant's guidelines + TAIL. HEAD keeps the tabc wording the model was trained under; TAIL
defines the confidence, fixes its spelling to the 11 one-decimal values (the rlcr11 scorer's levels) and tells the
model to stop after </confidence>. Used with analysis-only regeneration: the cached <think>...</answer> is prefilled,
so the answer (and accuracy) is identical across variants.

The `menu` variant is a template: build(name, techniques=[...]) fills in the techniques for one problem.
"""

HEAD = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its "
    "confidence about the solution and then provides the user with its confidence level. The final answer is enclosed "
    "between <answer> </answer> tags. The analysis about confidence and uncertainty is enclosed within "
    "<analysis> </analysis> tags.\n"
)

TAIL = (
    "\nThe confidence level is the probability that the final answer is exactly correct. It should be calibrated: "
    "among all answers given confidence 0.7, about 70% should be correct. Write it as a single number with one decimal "
    "place, one of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, alone on its own line inside "
    "<confidence> </confidence> tags. The response ends at </confidence>; write nothing after it.\n"
    "The final format that must be followed is: <think> reasoning process here </think><answer> final answer here "
    "</answer> <analysis> analysis about confidence and uncertainty here </analysis> <confidence> confidence level "
    "here (one of 0.0, 0.1, ..., 1.0) </confidence>"
)

NO_NUMBERS = (
    "Do not write any probability, percentage, or confidence value in the analysis. Describe the evidence in words; "
    "the number belongs only in the confidence tags."
)

GUIDES = {
    # A: tabc's own instruction. Isolates the effect of TAIL (definition, spelling, stop).
    "control": "The assistant should reason about its confidence in the solution and its uncertainty in the solution "
               "within these tags.",

    # B: RLCR's tabc_long guidelines (what the hotpot runs used), plus the no-numbers rule.
    "tabc_long": (
        "Here are some guidelines for the analysis:\n"
        "1. Point out things where the reasoning could be wrong, or where there might be ambiguity in the solution "
        "steps or in the reasoning process itself.\n"
        "2. Do not suggest ways of fixing the response; the job is only to reason about uncertainties.\n"
        "3. For some questions the response might be correct. Then it is fine to list only a small number of "
        "uncertainties and say explicitly that no more can be found.\n"
        "4. Uncertainties may differ from errors: they can come from ambiguity in the question, or from the "
        "application of a particular lemma or formula.\n"
        "5. If alternative approaches could lead to different answers, mention them.\n"
        "6. List specific, plausible uncertainties; do not make generic statements.\n"
        "7. " + NO_NUMBERS),

    # C: checks that do new computation after the answer is fixed.
    "verify": (
        "Here are the guidelines for the analysis:\n"
        "1. Check the final answer directly. Where possible, substitute it back into the problem, or test it against "
        "every condition, constraint, special case and unit the problem states. Say exactly what you checked and "
        "whether each check passed, failed, or could not be done.\n"
        "2. Name the one step of the reasoning most likely to be wrong (an arithmetic step, a missed case, an "
        "unstated assumption, a formula used outside its conditions) and explain why it is the weakest.\n"
        "3. Note any ambiguity in the question that would make a different answer correct.\n"
        "4. Be specific to this problem. Avoid generic statements such as \"the calculations seem correct\".\n"
        "5. " + NO_NUMBERS),

    # D: an independent second solution; agreement is the strongest evidence a sample-free model can produce.
    "rederive": (
        "Here are the guidelines for the analysis:\n"
        "1. Solve the problem a second time by a different method from the one in the reasoning process (for "
        "example: a different formula, working backwards from the answer, a small case or numeric check, or a "
        "diagram or table). If a full second solution is too long, independently recompute the key intermediate "
        "result instead.\n"
        "2. State clearly whether the second method agrees with the final answer.\n"
        "3. If they disagree, say which one you trust more and why. Do not change the final answer; let the "
        "disagreement lower the confidence instead.\n"
        "4. If the second method could not be carried out, say so and say why.\n"
        "5. " + NO_NUMBERS),

    # E: a verbalized distribution over answers rather than over confidences.
    "alternatives": (
        "Here are the guidelines for the analysis:\n"
        "1. Besides the final answer, list up to two other answers that a plausible mistake would have produced. "
        "For each, name the specific mistake (for example: an off-by-one count, a sign error, a missed case, "
        "misreading what the question asks for).\n"
        "2. For the final answer and each alternative, say how likely it is to be the correct answer, using only "
        "these words: very likely, likely, possible, unlikely, very unlikely.\n"
        "3. Explain what evidence favours the final answer over the alternatives, or what evidence does not.\n"
        "4. If no plausible alternative answer exists, say so and explain why the problem leaves little room for "
        "error.\n"
        "5. " + NO_NUMBERS),

    # F: the user's range hypothesis. Two stated ends and no commitment; the tag chooses.
    "range": (
        "Here are the guidelines for the analysis:\n"
        "1. Check the final answer directly (substitute it back, or test it against the conditions of the problem), "
        "and say whether the checks passed.\n"
        "2. Name the step most likely to be wrong and explain why.\n"
        "3. Do not write any probability or percentage in the analysis except on its last line.\n"
        "4. End the analysis with exactly one line of this form:\n"
        "Plausible confidence range: L to H\n"
        "where L and H are one-decimal numbers from 0.0 to 1.0, L is the lowest and H the highest confidence you "
        "would consider reasonable given the evidence above. A wide range means the evidence is mixed or weak; a "
        "narrow range means it is clear. Do not pick a single value in the analysis: the choice is made in the "
        "confidence tags, and it should lie between L and H."),
}

TECHNIQUES = {
    "substitute": "Substitute the final answer back into the problem, or test it against every stated condition, "
                  "and report whether it passes.",
    "rederive": "Solve the problem again by a different method, or independently recompute the key intermediate "
                "result, and report whether it agrees.",
    "weakest": "Name the step most likely to be wrong and explain why.",
    "error_modes": "List the mistakes that are typical for this kind of problem and say, for each, whether the "
                   "reasoning could have made it.",
    "difficulty": "Judge how hard this problem is and how often solutions to problems like it go wrong, and why.",
    "alternatives": "Name the most plausible competing answer, the mistake that would produce it, and which "
                    "answer the evidence favours.",
}


def menu_guide(techniques):
    lines = [f"{i + 1}. {TECHNIQUES[t]}" for i, t in enumerate(techniques)]
    return ("Here are the guidelines for the analysis. Use the following techniques, in this order of priority, and "
            "spend most of the analysis on the first:\n" + "\n".join(lines) + "\n"
            f"{len(lines) + 1}. Be specific to this problem; avoid generic statements.\n"
            f"{len(lines) + 2}. " + NO_NUMBERS)


def build(name, techniques=None):
    return HEAD + (menu_guide(techniques) if name == "menu" else GUIDES[name]) + TAIL


if __name__ == "__main__":
    for name in GUIDES:
        print(f"===== {name}\n{build(name)}\n")
    print(f"===== menu (example: rederive, weakest)\n{build('menu', ['rederive', 'weakest'])}")

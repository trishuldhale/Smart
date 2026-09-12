# GEB-AI Exact-Profile Recommendation Audit and Validation

## Implemented behavior

Optimization accepts one exact numeric value for each of BTE, BSFC, CO, and HC. The former performance-priority selector and sliders are not part of the interface. The system predicts the four outputs for every supported candidate and recommends the blend that most closely matches the complete entered profile.

Each absolute gap is divided by that target's unseen-blend P90 error. This gives BTE, BSFC, CO, and HC equal participation despite their different units and numeric scales. Per-target match values are averaged, then a small experimental-support penalty is applied. Stable tie-breaking makes identical requests reproducible.

## Search and safety rules

- Neural torque, RPM, and compression-ratio inputs are bounded to the nearest trained value before standard inference. The requested compression ratio remains active in the transparent BTE engineering rule.
- BSFC, CO, and HC outputs and their approximate intervals are bounded to measured target envelopes on standard pages. Any required operating-input or output bound makes the recommendation Exploratory.
- Candidate blends total exactly 100%, contain at least 75% gasoline, keep gasoline greater than ethanol and additive, and use only the selected additive.
- Candidates beyond the validated experimental composition-support threshold are rejected before matching.
- A zero-additive candidate is always included, so selecting an additive does not force it into the recommended blend.
- n-Methylaniline BTE remains globally capped at 37% on Prediction, Optimization, and Simulation.
- Bharat Stage provides exactly two choices, BS6 and BS4. It is stored with test context and exports but does not change model predictions because the supplied experimental model inputs contain no Bharat Stage field.

## Verification evidence

- 42 automated core and Streamlit smoke tests pass.
- A 108-case recommendation matrix passes across all four additives, three blend-constraint modes, three operating points, and three exact four-output profiles.
- Every matrix result preserves a 100% blend total, 75% gasoline floor, gasoline dominance, the single-selected-additive rule, validated composition support, and exact Prediction-page inference equality.
- Every n-Methylaniline matrix row containing the additive remains at or below 37% BTE.
- Exact profiles copied from two different supported candidates select their corresponding, different blends with a 100% four-output match.
- Missing, unknown, non-finite, and negative exact performance inputs are rejected.
- Changing any exact input or Bharat Stage clears a stale recommendation.
- The UI exposes all four exact number inputs and only BS6/BS4 Bharat Stage choices.
- Prediction, Optimization, and Simulation continue to call the same canonical standard inference function and pass exact cross-page equality tests.
- Raw Experimental prediction remains isolated from standard BTE conversion and output limits.

## Interpretation

The profile-match percentage describes closeness to the four user-entered values relative to observed unseen-blend model error. It is not a probability and does not prove that a requested profile is physically achievable. A Low confidence result means no supported candidate is close to the full request. Controlled engine testing and compatibility review remain required before practical use.

# Simulator, Experimental Prediction, and Interface Update

## Simulator controls

The existing calculated simulator engine was preserved. Its interface now exposes three explicit setups:

1. **Fixed blend - set every value:** gasoline, ethanol, and the selected additive are all entered by the user and remain fixed.
2. **Additive range - fix ethanol:** ethanol and additive range are entered by the user; gasoline balances every row to 100%.
3. **Additive range - fix gasoline:** gasoline and additive range are entered by the user; ethanol balances every row to 100%.

All standard simulator modes retain the 75% minimum gasoline rule, gasoline dominance, exact 100% composition, single-additive restriction, standard BTE calculation, and n-Methylaniline 37% ceiling.

## Raw experimental prediction

The new Experimental prediction page invokes each serialized performance network directly. It applies no BTE range conversion, operating-point clamping, output-envelope bound, or n-Methylaniline cap. Composition is still validated as a 100% fuel blend. The page reports when inputs are outside the model's dataset range but does not correct them.

This page is intentionally independent of standard Prediction, Optimization, and Simulation. Raw values can be physically impossible when inputs are outside the training distribution and are provided only for model-response inspection.

## Mechanical racing interface

The application now loads a dedicated HTML/CSS motorsport theme. It includes a carbon-metal background, racing-red accent system, mechanical hero panel, animated gear and speed lines, angular cards and buttons, dark chart styling, responsive behavior, and reduced-motion accessibility. The previous long deep-learning BTE banner was removed, and detailed BTE calculations are collapsed by default.

## Regression evidence

- 42 automated core and Streamlit UI tests pass.
- A 108-case exact-profile recommendation matrix passes across additives, blend-constraint modes, operating points, and entered performance profiles.
- All six pages execute without captured UI errors.
- The fixed simulator composition exactly matches the standard Prediction output at the same operating point.
- Both additive-range UI modes produce multiple valid compositions and preserve the selected fixed component.
- Raw experimental outputs match direct calls to all four serialized networks to numerical precision.
- The raw extreme-input test confirms that no BTE or non-BTE output correction is applied.
- Existing recommendation, BTE-rule, cross-page consistency, simulator, stale-result, and n-Methylaniline-cap tests continue to pass.

## Exact-value recommendation update

The Optimization page now accepts exact BTE, BSFC, CO, and HC values as four number inputs. The former performance-priority profiles and sliders were removed. All four gaps are scaled by their corresponding unseen-blend P90 error and contribute equally to the complete-profile match. The sidebar also provides only BS6 and BS4 as Bharat Stage choices; the selected value is retained as test metadata because Bharat Stage is not a feature in the supplied experimental dataset.

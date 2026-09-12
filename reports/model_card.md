# GEB-AI Model Card

## Intended use

Screening-level deep-neural prediction and empirical-support optimization of gasoline/alcohol blends for the supplied SI-engine dataset. Outputs are decision support, not a substitute for dynamometer testing, fuel compatibility testing, safety review, or manufacturer approval.

## Data used

- Valid training records: 1135
- Unique valid blend groups: 85
- Quarantined records: 237
- Split rule: whole fuel-composition groups are held out so the same blend cannot leak into training and test sets.

## Unseen-blend holdout results

| Target | Selected model | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| BTE (%) | Deep Neural Network (128-128-64-32) | 0.4619 | 0.6223 | 0.9979 |
| BSFC (g/kWh) | Deep Neural Network (128-128-64-32) | 29.3158 | 57.3712 | 0.9839 |
| CO (vol.%) | Deep Neural Network (128-128-64-32) | 0.2167 | 0.3032 | 0.9508 |
| HC (ppm) | Deep Neural Network (128-128-64-32) | 16.6489 | 26.8633 | 0.9829 |

## Fuel-property estimator CV

| Property | Mean R² ± SD |
|---|---:|
| air_fuel_ratio | 0.6576 ± 0.1449 |
| cv_mj_kg | 0.5645 ± 0.2396 |
| oxygen_pct | 0.8229 ± 0.2333 |
| ron | -1.4437 ± 2.5240 |
| viscosity_m2_s | 0.6769 ± 0.1349 |

## Hybrid BTE calculation requested for this project

For CR >= 9.5, the app first locates the internal DNN BTE signal inside the cleaned-data minimum/maximum envelope at that compression ratio:

| Measured CR | Rows | Source minimum | Source maximum |
|---:|---:|---:|---:|
| 9.70 | 386 | 26.765170% | 43.051436% |
| 10.50 | 335 | 27.868437% | 43.438488% |

`q = clip((DNN BTE signal - BTE minimum at CR) / (BTE maximum at CR - BTE minimum at CR), 0, 1)`

It then maps that dataset-driven score into the active blend range:

`Final BTE = Blend minimum + q * (Blend maximum - Blend minimum)`

| Blend rule | Final BTE range |
|---|---:|
| E0: G100 | 25-30% |
| E5: G95 + E5 | 27-32% |
| E10: G90 + E10 | 31-35% |
| E15: G85 + E15 | 26-31% |
| E20: G80 + E20 | 23-30% |
| E10 + P5: G85 + E10 + pentanol 5 | 32-37% |
| E10 + propanol/butanol 5 | 31-35% |
| E10 + n-methylaniline 5 | 31-37% (37% maximum) |

The prompt's invalid “95% gasoline + 27% ethanol” combination is treated as E5 (95% gasoline + 5% ethanol), with 27% as the lower BTE bound. Whenever n-methylaniline is present, a final global ceiling of 37% is enforced at every compression ratio, including CR below 9.5. Prediction, Optimization, and Simulation use the same canonical inference function.

The user-supplied ranges are engineering priors and conflict with some measured source labels. Therefore, the raw DNN holdout metrics above do not constitute independent validation of the final rule-guided BTE; controlled engine testing remains required.

## Recommendation method

The recommendation engine generates gasoline/ethanol blends containing at most the one user-selected additive and includes a zero-additive baseline. It rejects candidates beyond the validated composition-support threshold, then compares every supported blend with exact user-entered BTE, BSFC, CO, and HC values.

Each absolute prediction gap is divided by its target's unseen-blend P90 error so all four outputs participate equally despite their different units. Match utilities are averaged, and a support-distance penalty of up to 0.03 favors candidates nearer experimental compositions when matches are similar. Deterministic tie-breaking provides reproducible ranking. Confidence reflects the average and largest model-error-scaled gaps; any operating-input or target-output bound changes the label to Exploratory.

## Important limitations

- The neural training envelope is torque 1.92-8.08 Nm, RPM 1700-3300, and CR up to 10.5. Requests outside it use the nearest trained edge for neural inference and are flagged Exploratory; the requested CR remains active in the BTE engineering rule.
- Vehicle CR values that are not directly represented by a measured training level are also flagged with the nearest tested CR.
- The supplied vehicle table does not contain engine torque or RPM. The interface uses clearly labelled dataset starting points that must be changed to the intended test condition.
- Fuel-property predictions for new blends are estimated from the median property values of known blend groups and bounded to observed ranges.
- The four performance targets use separate four-hidden-layer deep neural regressors (128-128-64-32 neurons). The auxiliary fuel-property estimator remains separate and is not used as a performance target model.
- BSFC, CO, and HC are bounded to their measured target envelopes after neural inference, preventing implausible out-of-range network outputs from being rewarded by optimization.
- The separate Experimental prediction page deliberately bypasses the BTE conversion, n-Methylaniline ceiling, operating-input clamping, and all output bounds. Its direct network values may be negative or implausible outside the dataset and must not be treated as recommendations.
- Candidate search combines convex mixtures of experimentally supported blends with the explicit user-rule anchors, then rejects distant candidates. It does not establish fuel-system, material, legal, or road-use compatibility.
- Bharat Stage is retained as BS6/BS4 test metadata only; it is not a trained feature in the supplied experimental dataset.
- Recommendation and standard simulation enforce at least 75% gasoline, require gasoline to exceed ethanol and the selected additive, and never permit more than one additive. Simulation supports a fully fixed blend or an additive range with either ethanol or gasoline fixed.
- BTE, BSFC, CO, and HC are independently learned targets. The app reports a heat-balance consistency warning when needed.

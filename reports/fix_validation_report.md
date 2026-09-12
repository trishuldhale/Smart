# GEB-AI Deep-Learning Repair and Validation Report

## Problems reproduced

1. The previous n-methylaniline BTE ceiling was reachable only through the CR >= 9.5 calculation branch. At CR 9.49 and below, the uncapped model value could be reported above 37%.
2. Prediction and batch evaluation had separate inference implementations. Although identical backend inputs could match, duplicated logic made future divergence possible.
3. Streamlit retained a previous result after users changed controls. This stale display could make Prediction, Optimization, and Simulation appear inconsistent.
4. The interface displayed the uncapped internal BTE signal as a primary metric, which could be mistaken for the final reported prediction.
5. The performance estimators were conventional machine-learning regressors rather than the requested deep neural models.

## Corrections implemented

- Added a global 37% final BTE ceiling whenever n-methylaniline is greater than zero, independent of compression ratio and blend-range rule.
- Applied the same ceiling to the displayed BTE uncertainty interval.
- Centralized BTE, BSFC, CO, HC, fuel-property, and support-distance inference in one canonical function used by all three calculation pages.
- Added input fingerprints that remove stale results after any relevant control changes.
- Removed the internal uncapped BTE signal from user-facing cards, alternative tables, and downloadable recommendation data.
- Replaced all four performance estimators with target-specific feed-forward neural networks using hidden layers of 128, 128, 64, and 32 neurons.
- Tightened optimizer support filtering to the validated support-distance threshold rather than 1.25 times that threshold.
- Preserved all approved composition rules, gasoline minimum, gasoline-dominance checks, single-additive selection, and fixed-ethanol/fixed-gasoline/free modes.

## Deep-network validation

The cleaned dataset contains 1,135 valid rows and 85 unique fuel-blend groups. The final test set contains 220 rows from 17 complete blend groups that were not used for network fitting or grouped cross-validation.

| Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| BTE (%) | 0.4619 | 0.6223 | 0.9979 |
| BSFC (g/kWh) | 29.3158 | 57.3712 | 0.9839 |
| CO (vol.%) | 0.2167 | 0.3032 | 0.9508 |
| HC (ppm) | 16.6489 | 26.8633 | 0.9829 |

These metrics evaluate internal dataset-trained predictions. The final BTE blend ranges and n-methylaniline ceiling are deterministic engineering constraints and require controlled engine validation.

## Functional verification

- 42 automated core and Streamlit smoke tests pass.
- All five Streamlit workspaces execute without captured UI errors.
- Fixed-ethanol, fixed-gasoline, and free simulation modes pass for gasoline/ethanol only and every supported additive.
- 20,736 n-methylaniline operating cases were checked across CR 4-15, torque 0.1-30 Nm, RPM 500-12,000, and multiple ethanol/additive levels.
- Maximum reported n-methylaniline BTE: 37.000000%.
- Cases above 37%: 0.
- Forty randomized comparisons between single Prediction and the shared batch path produced zero numerical difference for BTE, BSFC, CO, and HC.

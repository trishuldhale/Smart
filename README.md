# GEB-AI: SI-Engine Fuel-Blend Prediction and Optimization

This project trains four target-specific deep feed-forward neural networks on the supplied gasoline/alcohol experimental dataset and integrates them into a Streamlit web application. It predicts BTE, BSFC, CO, and HC; estimates fuel properties for new candidate blends; and recommends the empirically supported blend whose predicted four-output profile is closest to exact values entered for a selected two-wheeler and operating point.

Each performance network has four hidden layers with 128, 128, 64, and 32 neurons. The same serialized networks and one canonical inference function are used by Prediction, Optimization, and Blend simulation, so an identical blend and operating point produces the same four outputs on every page.

## Hybrid dataset + rule BTE calculation

For vehicles with compression ratio (CR) **9.5 and above**, BTE is calculated in two transparent stages. The internal deep-network signal is first normalized inside the cleaned-data BTE envelope at the selected CR:

```text
q = clip((internal DNN BTE signal - source minimum at CR)
         / (source maximum at CR - source minimum at CR), 0, 1)
```

The normalized score is then mapped into the active user-specified blend range:

```text
Final BTE = blend minimum + q × (blend maximum - blend minimum)
```

| Fuel blend | BTE range |
|---|---:|
| G100 / E0 | 25-30% |
| G95 / E5 | 27-32% |
| G90 / E10 | 31-35% |
| G85 / E15 | 26-31% |
| G80 / E20 | 23-30% |
| G85 / E10 / pentanol 5% | 32-37% |
| G85 / E10 / propanol or butanol 5% | 31-35% |
| G85 / E10 / n-methylaniline 5% | 31-37% (37% maximum) |

The invalid statement “95% gasoline + 27% ethanol” is interpreted as **95% gasoline + 5% ethanol**, with **27%** as the lower BTE bound. This matches the E0/E5/E10/E15/E20 sequence and preserves the required 100% composition total.

For other high-CR blends, the base normalized range is 26-36%. Intermediate ethanol-only blends from E0 to E20 use piecewise-linear interpolation between the listed rules. E10 blends containing up to 5% additive transition from the E10 range; pentanol reaches 32-37% at P5 and n-methylaniline reaches 31-37% at NMA5.

Whenever n-methylaniline is greater than zero, the final reported BTE has a global hard ceiling of 37%, including CR below 9.5. The uncapped internal DNN signal is retained only in audit artifacts and is never presented as the application prediction. These blend ranges are user-supplied engineering priors and conflict with some source measurements. Consequently, deep-network accuracy metrics do not independently validate the final rule-guided BTE. Controlled engine testing is still required.

## What is included

- Cleaned model-ready CSV and a separate quarantine CSV.
- Group-leakage-controlled deep-network validation and unseen-blend testing.
- Separate four-hidden-layer neural regressor for BTE, BSFC, CO, and HC.
- Composition-to-property estimator for CV, A/F, viscosity, RON, and oxygen.
- Approximate 90% prediction error bands from the unseen-blend holdout, with the BTE band transformed into the active blend range.
- Applicability warnings for unfamiliar blends and out-of-range engine inputs.
- Vehicle-specific exact-profile recommendation using selected vehicle, Bharat Stage, torque, compression ratio, and optional manually entered RPM.
- Strict single-additive search: every recommendation contains gasoline, ethanol, and only the one user-selected additive; all other additives are fixed at zero.
- Four direct inputs for the exact BTE, BSFC, CO, and HC values required by the user; there are no performance-priority sliders.
- Unit-balanced recommendation scoring using each output's unseen-blend P90 error, an empirical-support penalty, and deterministic tie-breaking.
- A zero-additive baseline, entered-versus-predicted gap table, profile-match score, and recommendation confidence label.
- A blend simulator with a fully user-fixed composition and additive-range modes using either fixed ethanol or fixed gasoline.
- A separate Experimental prediction page that reports direct dataset-trained network outputs without BTE conversion, input clamping, n-Methylaniline capping, or target-output bounds.
- Automatic stale-result clearing whenever a page input changes.
- One canonical calculated prediction path shared by Prediction, Optimization, and Simulation; the Experimental page is deliberately isolated from it.
- Support-distance status for simulated blends and filtering for recommended blends.
- n-Methylaniline remains available only as a deliberate selection and is labelled for controlled-laboratory study.
- Vehicle selection from the supplied two-wheeler comparison document.
- MotoGP-inspired mechanical HTML/CSS interface with dark carbon panels, racing-red controls, responsive layouts, and reduced-motion support.
- Streamlit prediction, experimental-prediction, optimization, blend-simulation, analytics, and data-quality views.

## Vehicle-specific recommendation

1. Select a company, vehicle model, and either BS6 or BS4 in the sidebar.
2. Enter torque and confirm or change the vehicle's compression ratio.
3. RPM is optional. When omitted, the app uses the existing 2500 RPM dataset starting point and labels it clearly.
4. Select Pentanol, Propanol, Butanol, n-Methylaniline, or the gasoline/ethanol-only option.
5. Keep ethanol fixed (the default), fix gasoline instead, or turn both toggles off for a free search. Only one value may be fixed.
6. Enter the exact BTE, BSFC, CO, and HC values required for the test condition.
7. Run the recommendation to receive the closest supported blend and ranked alternatives.

The search grid always totals 100%. Choosing one additive forces the other three additive percentages to exactly 0%. Every generated blend contains at least 75% gasoline, and gasoline must be greater than both ethanol and the selected additive.

The optimizer first rejects candidates outside the validated composition-support threshold. For each remaining candidate, it calculates the absolute gap between predicted and entered BTE, BSFC, CO, and HC. Each gap is divided by that output's unseen-blend P90 error so the four differently scaled units participate equally. The four match contributions are averaged, and a small support-distance penalty favors better-supported blends when profile matches are similar. Deterministic tie-breaking makes repeated runs reproducible. The zero-additive alternative is always included, so choosing an additive in the interface does not force the optimizer to recommend a nonzero dose.

The Bharat Stage selector records either BS6 or BS4 with the chosen vehicle/test context and in exported recommendation or simulation results. Bharat Stage is not used as a numerical prediction feature because it is absent from the supplied experimental model inputs.

## Blend simulation

The simulator offers three user-controlled setups. **Fixed blend** accepts exact gasoline, ethanol, and selected-additive percentages and evaluates that one composition. **Additive range - fix ethanol** holds the entered ethanol percentage while sweeping the entered additive range and balancing with gasoline. **Additive range - fix gasoline** holds gasoline while sweeping additive and balancing with ethanol. Every generated blend remains exactly 100%, contains at least 75% gasoline, and uses only the selected additive. The calculation engine is unchanged, so a fixed simulated blend matches the standard Prediction page at the same operating point.

## Experimental prediction

Experimental prediction sends the entered nine model features directly to the four trained networks. It does not apply the calculated BTE range, the 37% n-Methylaniline ceiling, operating-input clamping, or output-envelope bounds. Consequently, out-of-distribution inputs can produce negative or otherwise implausible raw values. This page is intended only for inspecting the original dataset-trained model response; it is not used by Optimization or the standard Simulator.

## Quick start on Windows

1. Extract the project ZIP.
2. Open the extracted `GEB_AI_SYSTEM` folder.
3. Double-click `run_app.bat`.

The first run creates a virtual environment and installs the listed Python packages. A trained deep-network bundle is already included. The application will open in your browser.

The requirements pin scikit-learn 1.8.0 because serialized scikit-learn estimators must be loaded with their training version.

To rebuild every model from `data/raw/data.xlsx`, double-click `retrain_and_run.bat`.

## Manual start in VS Code

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Retraining is needed only when the dataset or network configuration is intentionally changed.

## Validation design

The test split holds out complete fuel-composition groups. Therefore, the same or essentially identical blend cannot appear in both training and testing merely at a different torque, RPM, or compression ratio. Each target-specific deep network is checked by five-fold grouped cross-validation on the development portion. It is then evaluated once on a separate 20% unseen-blend holdout before deployment refitting on all valid rows.

| Target | Holdout MAE | Holdout RMSE | Holdout R² |
|---|---:|---:|---:|
| BTE (%) | 0.4619 | 0.6223 | 0.9979 |
| BSFC (g/kWh) | 29.3158 | 57.3712 | 0.9839 |
| CO (vol.%) | 0.2167 | 0.3032 | 0.9508 |
| HC (ppm) | 16.6489 | 26.8633 | 0.9829 |

These values come from 220 rows representing 17 complete unseen blend groups. They measure the internal dataset-trained networks; BTE range rules and the n-methylaniline ceiling are deterministic post-processing constraints.

The deployment model uses nine primary inputs:

1. Gasoline, ethanol, pentanol, propanol, butanol, and n-methylaniline percentages.
2. Torque.
3. RPM.
4. Compression ratio.

Fuel properties are estimated in a separate layer. They are not editable arbitrary inputs to the performance model, which prevents the optimizer from inventing incompatible composition/property combinations.

## Data-quality policy

- Fuel composition must total 99.99-100.01% in the source data.
- Accepted rounding deviations are proportionally normalized to exactly 100%.
- Larger deviations and nonnumeric/missing records are retained in `data/processed/quarantine_invalid_rows.csv`.
- Exact duplicate model records are retained in the same quarantine file and excluded so repeated copies do not overweight training.
- Quarantined rows are excluded from training, but never silently destroyed.
- The original workbook remains unchanged in `data/raw/`.

## Vehicle-data limitation

The supplied vehicle document includes compression ratio, category, fuel, year, BS norm, and mileage, but it does **not** include torque or RPM. The app therefore uses 5.51 Nm and 2500 RPM only as clearly labelled dataset starting points. Set these fields to the intended engine test condition before using a prediction.

The neural-network training data covers torque 1.92-8.08 Nm, RPM 1700-3300, and CR up to 10.5. On the standard calculated pages, operating points outside those limits remain selectable but neural inference is bounded to the nearest trained edge and labelled **Exploratory**. The requested CR is retained for the BTE range rule. BSFC, CO, and HC are bounded to measured target envelopes so unstable values cannot become false optima. The separate Experimental prediction page intentionally bypasses all of these post-processing protections.
The data also uses only six discrete CR levels, so an in-range but unmeasured vehicle CR is flagged with its nearest tested level.

## Output files

- `artifacts/geb_ai_model_bundle.joblib` - trained model and inference metadata.
- `reports/holdout_metrics.csv` - final unseen-blend metrics.
- `reports/grouped_cv_metrics.csv` - per-fold model comparison.
- `reports/feature_importance.csv` - permutation importance.
- `reports/fuel_property_cv_metrics.csv` - property-estimator validation.
- `reports/bte_rule_ranges.csv` - auditable user-specified BTE rule table.
- `reports/model_card.md` - intended use, evidence, and limitations.
- `reports/recommendation_validation_report.md` - optimizer defects, corrections, and matrix-validation evidence.
- `reports/ui_simulator_experimental_update.md` - simulator, raw-output page, theme, and regression evidence.
- `assets/motorsport.css` - mechanical racing interface theme.
- `reports/figures/` - model and data-audit plots.

## Safety and scientific scope

This is a screening and research decision-support tool. It does not prove blend safety, material compatibility, regulatory compliance, manufacturer approval, or real-world performance. Confirm any recommended blend through expert review and controlled dynamometer/engine testing before operational use.
#   S m a r t  
 
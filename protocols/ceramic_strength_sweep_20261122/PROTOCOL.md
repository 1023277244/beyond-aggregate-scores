# Ceramic filter strength sweep

This single-product follow-up measures how changing the synthetic blend coefficient affects each real defect type. Ceramic filter was selected after the original two-level study. This is a follow-up on the same evaluation units. The five coefficients are fixed before any intermediate-model test scoring.

- Product and view: Real-IAD Variety ceramic_wave_filter, original C1 split.
- Coefficients: 1/3, 1/2, 2/3, 5/6, 1. Seeds: 42, 0, 1.
- Reuse the six archived S0/M endpoints if their encoder, training input, initialization, masks and batch schedule match. Fit nine new heads, each for 600 updates.
- Reuse the archived 12 normal fit groups per seed, six separate calibration groups, and all 224 official test groups. Match source pixels, synthetic contents, masks, augmentation, initialization and batch order. For intermediate arms use round-to-nearest integer interpolation between the clean source and the original full-strength composite.
- Reuse the frozen ResNet18 encoder, patch normalization, Head and balanced BCE. Checkpoint rule: step 600. Calibrate thresholds as the maximum of the six normal calibration scores, separately for each fitted head.
- Train and seal all endpoints and calibration thresholds before opening test images in the separate evaluation process. Evaluate every coefficient and seed together. Compare historical endpoint predictions against archived scores as a compatibility check.
- Report overall and all four defect AUROCs, fixed-normal-calibration detections and false alarms, and the separate post-hoc test ROC diagnostic at at most one normal false alarm out of 25. Compare all lower coefficients to alpha=1 with the existing 2,000-draw paired, class-stratified part bootstrap. The intervals are conditional on the fixed fits and product.
- Report the full curve regardless of direction. No selection of a final alpha or checkpoint from the sweep. End after the nine planned fits and the complete five-level evaluation.

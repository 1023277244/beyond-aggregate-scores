# Region-exclusion diagnostic on ceramic filters

## Question

The five-level sweep established opposite image-level ranking changes for Scratch and MissingPart. This analysis asks how much of each change is carried by the cells aligned with the annotated defect in the top-11 score readout. It is a post-hoc diagnostic on the same product and test units. The region calculation, controls, outcomes and scope are fixed before extracting the new patch maps.

## Fixed inputs

- Use all 15 frozen ceramic-filter heads, all 224 C1 test parts, and the six original normal calibration parts. Annotations enter the offline analysis after score-map inference. Export no new training input, fit no model and change no alarm threshold.
- Reconstruct each whole-image score using the original frozen ResNet18 and head. Match the previously sealed scores for all 15 heads and match normal calibration scores. Use the 32-by-32 logit grid, before display interpolation or heatmap normalization.
- Use all 199 defective parts and all 25 normal comparator parts. Include one official mask per defective part. A missing, empty or misaligned mask stops the analysis for that product and is recorded. Each cell is assigned to the annotation when its corresponding 1/32-by-1/32 image bin has nonzero annotation coverage. These are grid locations. A cell's receptive field extends beyond its bin.

## Matched regions and readout intervention

- For each defect part, compute the original top-11 mean-patch score. Recompute it after removing the annotation-aligned cells from the candidate pool. Apply exactly the same spatial exclusion to the same coordinates in each normal comparator. Keep remaining cell activations intact and fill the top-11 slots from the next highest eligible cells.
- Also remove 16 in-bounds, non-overlapping translations of the same grid footprint, or all feasible translations when there are fewer than 16. Choose translations by closest mean top-11 density measured on the six calibration normals, averaged across all 15 frozen heads. Break ties by distance from the original region and then by vertical and horizontal coordinates. No defective response or test label enters control-region selection. Report region and control coverage.
- For each defective-normal pair and head, record the original win (ties count as one-half), the win after annotation exclusion, and the mean win over the translated-region exclusions. The class mean of full wins is the original AUROC. The matched exclusions produce part-pair ranking diagnostics with part-specific masks and are not conventional deployable AUROC measurements.

## Comparisons

- Primary contrasts are alpha=1/3 minus alpha=1, separately for Scratch and MissingPart. First compute the strength effect under each exclusion condition. Then subtract the effect under annotation exclusion from the effect under translated-region exclusion. Positive values mean annotation removal weakened the weaker-strength advantage more than control removal. Negative values mean annotation removal weakened the stronger-strength advantage more than control removal.
- Report the same component estimates for the three intermediate levels and both other defect types as secondary analyses. Also report the translated-control match quality and mean number of top-11 cells removed.
- Use 2,000 paired bootstrap draws with seed 1120. Resample normal parts and, within each type, defect parts with replacement, sharing draws across all conditions and model fits. Average the three fixed seeds within each draw. Report pointwise conditional 95% intervals, no significance labels. Neither grid cells nor seed runs count as independent test parts.
- The conclusion concerns dependence of the fixed pooling readout on annotated grid locations. The experiment makes no changes to the input image or its features.

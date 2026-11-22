# Beyond Aggregate Scores

This repository accompanies an empirical study of how synthetic anomaly strength changes which real defects an industrial detector catches. The paper examines a matched training intervention, separately frozen product panels, a five-level strength comparison, and a region-exclusion diagnostic.

## Reproduce the paper's point estimates

Run this command with Python 3.10 or newer. It uses the Python standard library.

    python reproduce.py

The script recomputes ceramic-filter AUROCs and both operating-point counts for all five strengths and three fitted seeds. It verifies the opposing-effect criterion across all nine products and reconstructs the spatial readout point estimates.

- [Public benchmark evidence](data/) contains the benchmark scores, labels, normal calibration scores, uncertainty tables and complete product outcomes.
- [Experiment source snapshots](experiment_code/) cover the paper's training, scoring and spatial readout analyses. They expect the referenced public datasets and model artifacts in the original workspace layout.
- [Protocols](protocols/) define the matched comparisons and separation of training, normal calibration, and held-out test units.

Obtain benchmark images from the [Real-IAD Variety authors](https://huggingface.co/datasets/Real-IAD/Real-IAD_Variety) and [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad). The framework comparison uses the [GLASS implementation](https://github.com/cqylunlun/GLASS) and [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

## Conveyor end-face inspection

[Watch or download the 6.5-second event replay](industrial_demo/conveyor_end_face_demo.mp4). It shows two selected development events, one pass and one alarm, with their decisions displayed at the sampled frames. The public benchmark experiments establish the synthetic-strength results separately.

The [demo directory](industrial_demo/) contains the original HSV rule and video renderer. Running the demo renderer requires an authorized local source video and the companion event records and aligned crops.

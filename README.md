# Beyond Aggregate Scores

This repository accompanies an empirical study of how synthetic anomaly strength changes which real defects an industrial detector catches. The paper examines a matched training intervention, separately frozen product panels, a five-level strength comparison, and a region-exclusion diagnostic.

## Code

- [Experiment source snapshots](experiment_code/) cover the paper's training, evaluation and spatial readout analyses. They expect the datasets and model artifacts in the original study layout.
- [Point-estimate replay](reproduce.py) recomputes the primary tables from the accompanying local reproducibility data package. Place its data directory beside the script, then run the following command with Python 3.10 or newer.

      python reproduce.py

Obtain benchmark images from the [Real-IAD Variety authors](https://huggingface.co/datasets/Real-IAD/Real-IAD_Variety) and [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad). The framework comparison uses the [GLASS implementation](https://github.com/cqylunlun/GLASS) and [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

## Conveyor end-face inspection

[Watch or download the 6.5-second event replay](industrial_demo/conveyor_end_face_demo.mp4). It shows two selected development events, one pass and one alarm, with their decisions displayed at the sampled frames. The public benchmark experiments establish the synthetic-strength results separately.

The [demo directory](industrial_demo/) contains the original HSV scoring rule and the video renderer. Rendering requires an authorized local source video and the companion event records and crops.

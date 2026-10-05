# oil4-yolo

Code, reference outlines and results for

> **Detecting terrestrial oil spills in Sentinel-2 imagery with a segmentation network trained only on physically simulated plumes**
> Nazım Aksaker, Çukurova University ([ORCID 0000-0001-8721-5036](https://orcid.org/0000-0001-8721-5036))

YOLO26 instance segmentation networks are trained **without any real spill**, on Sentinel-2 image pairs of farmland into which synthetic oil plumes are implanted with a multiplicative attenuation model, and are tested **only** on two real crude oil pipeline spills in southeastern Türkiye:

| event | image pair | substrate | reference outline | null pairs |
|---|---|---|---|---|
| Narlı 2018 | 15 → 20 Jun 2018 | harvested farmland | 872 px at 10 m (8.72 ha), includes the oiled margin | 6 |
| Siverek 2021 | 5 → 15 Aug 2021 | irrigated maize | 49 px core (0.49 ha) | 16 |

Operating thresholds are fixed on synthetic validation data before the real images are examined, and false alarms are counted on the pre-event (null) pairs.

## Pipeline

| script | does |
|---|---|
| `config.py` | events, dates, coordinates, training plains, constants |
| `s2io.py` | windowed reads of Sentinel-2 L2A COGs from the Earth Search STAC (Collection 1, falls back to the original archive) |
| `realio.py` | loads a real chip and co-registers pre and post (phase correlation on high-passed B8) |
| `02_fetch_real.py` | real event and null chips, plume masks, measured attenuation `data/k_*.json` |
| `03_fetch_train.py` | clean background pairs from ten agricultural plains, at least 15 km from either spill |
| `04_synth.py` | implants plumes (`--profile v1` or `v2`, `--ablate sharp\|halo\|green\|swir`) and writes the YOLO dataset |
| `05_train.py` | trains a segmentation model (YOLO26n by default; YOLOv8n and YOLO11n for comparison) |
| `06_evaluate.py` | symmetric patch evaluation of the network and a bitemporal ΔNBR z score detector on the real chips |
| `07_analysis.py` | synthetic validation metrics, recall by plume size, input size test, channel occlusion, cost and latency |
| `make_figs.py` | the figures of the paper |
| `run_all.sh` | the whole experiment, sequentially |

Network input is three change channels (ΔNBR, Δlog BAI, Δlog mean reflectance) stretched to 8 bits so that zero change equals 114, the Ultralytics padding value.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PY=python ./run_all.sh
```

Imagery is read directly from the public [Earth Search](https://earth-search.aws.element84.com/v1) catalogue; no account is needed. The clean training pairs are drawn at random, so a rerun gives a different but equivalent training set. Each training run takes about 3.5 h on an 8 GB Apple M1.

## Results

`results/<run>/` holds, for every trained network, `eval_summary.json` (fixed operating point and post hoc oracle for each event), the threshold sweeps `eval_<event>_<detector>_<coreg|raw>.csv`, `analysis.json` and the Ultralytics training log `results.csv`. Run names are `<dataset>_<model>_s<seed>`; `gen` is generator v1, `gen_v2` generator v2 and `gen_v2_no<x>` the ablations.

Detection at the operating points fixed in advance (co-registered pairs; IoU of the detected patch, FA = false alarms per km² of null imagery):

| detector | threshold | Narlı IoU | Narlı FA | Siverek IoU | Siverek FA |
|---|---|---|---|---|---|
| ΔNBR z score | z = 4 | 0.61 | 0.18 | 0.47 | 0.66 |
| YOLO26n v1, seeds 0 to 2 | 0.30 to 0.35 | missed | 0 | missed | 0.01 |
| YOLO26n v2, seed 0 | 0.25 | 0.71 | 0 | 0.50 | 0 |
| YOLO26n v2, seed 1 | 0.35 | 0.55 | 0 | 0.51 | 0 |
| YOLO26n v2, seed 2 | 0.40 | missed | 0 | missed | 0 |
| YOLOv8n v2 | 0.15 | 0.78 | 0 | 0.47 | 0 |
| YOLO11n v2 | 0.25 | 0.59 | 0 | 0.45 | 0.02 |

All three v2 seeds rank both spills first in their scenes; seed 2 misses them only because its synthetic validation threshold (0.40) lies above the spill scores (0.22).

## Trained weights

The `best.pt` of all twelve runs are attached to the [release](../../releases) of this repository.

## Data and licences

* Code: MIT licence (`LICENSE`).
* Reference outlines (`labels/`), attenuation ratios (`data/`) and results (`results/`): CC BY 4.0.
* Sentinel-2 data: contains modified Copernicus Sentinel data 2018 to 2025, accessed through Earth Search (Element 84) on the Registry of Open Data on AWS.
* Pipeline routes in `assets/`: © OpenStreetMap contributors, ODbL 1.0. The national outline used in the study area map is not redistributed; the map is drawn without it.

## Citation

The paper reference will be added here on publication.

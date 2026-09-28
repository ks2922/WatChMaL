# MRes project

Deep learning and topological data analysis for single- and multi-ring event reconstruction at WCTE.    
**Code:** https://github.com/ks2922/WatChMaL, branch `master` (fork of WatChMaL/WatChMaL).  
**TDA-specific details:** see `tda_README.md`.

---

## HEP layout (`/vols/hyperk/users/ks2922/ML_2025/`)

### Analysis notebooks (`scripts/analysis/`)

The main analysis lives here. Key notebooks:

| Notebook | Contents |
|---|---|
| `regression_katie_singlering_thesis.ipynb` | Single-ring resolution results |
| `regression_katie_tuning_thesis.ipynb` | PointNet++ architecture search / tuning |
| `regression_katie_pointnet_tuning_npoint_radius_thesis.ipynb` | n-point and radius tuning |
| `regression_katie_pointnet_tuning_SA_layers_thesis.ipynb` | SA-layer depth study |
| `regression_multiring_sweep_thesis.ipynb` | Multi-ring architecture sweep results |
| `regression_multiring_tuned_final_thesis.ipynb` | Final multi-ring results (thesis figures) |
| `regression_multiring.ipynb` | Multi-ring working notebook |
| `regression_multiring_persistence.ipynb` | TDA persistence features for multi-ring |

### TDA analysis (`scripts/analysis/tda/`)

| File | Contents |
|---|---|
| `geometry_persistence_visualisation.ipynb` | Geometry and persistence diagram visualisation |

### Simulation / dataset checks (`simulation/analysis/`)

| File | Contents |
|---|---|
| `dataset_check.ipynb` | Dataset integrity and distribution checks |
| `multiring_simulation_analysis.py` | Multi-ring simulation analysis |
| `simulation_analysis*.py` | Various simulation diagnostic scripts (charge, time, etc.) |

## fir Layout (`scratch/ks2922/mres/`)

| Folder | Contents |
|---|---|
| `WatChMaL/` | the git repo (models, engines, configs, analysis notebooks) |
| `tda/` | TDA precompute and feature scripts and outputs; see `tda_README.md` |
| `job_scripts/` | all SLURM jobs, one `.job` and one `.sh` per run (see below) |
| `containers/` | Apptainer images (see below) |
| `cds_geometry/` | `h5_files/` (simulation), `geometries/` (WCTE geometry `.npz`), `split_paths/` (train/val/test splits) |
| `simulation/` | WCSim MAC templates for multi-ring events (only test versions, Steph generated the data we used) |
| `warm_start_checkpoints/` | checkpoints used to warm-start TDA-fusion runs (V8) |
| `logs/` | Some logs for TDA precomputation runs (`%x_%j.out/.err`) |

---

## Containers

| Container | Use |
|---|---|
| `container_base_ml_v4.0.0_watchmal-pointnet2.sif` | **Main container.** PyTorch with `pointnet2_ops` pre-built inside the image. Use this for all PointNet++ runs. Otherwise it becomes very difficult with the specific PointNet++ library I used. |

> **Note on `pointnet2_ops`:** the library is installed inside `container_base_ml_v4.0.0_watchmal-pointnet2.sif` and does not need to be rebuilt. Do not attempt to reinstall it outside the container. The container may need to be recreated if additional compute types must be installed. This container works for compute capabilities: `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"`, which covered the full range of capabilities for GPUs across HEP, HPC and fir clusters.

---

## job_scripts

Each run has a paired `.job` (SLURM directives + `apptainer exec` call) and `.sh` (the actual Python training command).

### Top-level scripts

| File | What |
|---|---|
| `reg_e_5sa1_12M_60ep.{job,sh}` | Single-ring PointNet++ electron regression, 5 SA layers, 12M events, 60 epochs |
| `reg_mu_5sa1_12M_60ep.{job,sh}` | Single-ring PointNet++ muon regression, 5 SA layers, 12M events, 60 epochs |

### `new_singlering/resnet/`

Single-ring ResNet baselines.

| File | What |
|---|---|
| `reg_e_baseline_12M.{job,sh}` | Electron position/direction/energy regression, ResNet backbone, 12M events |
| `reg_mu_baseline_12M.{job,sh}` | Muon equivalent |

### `pointnet/`, `new_multiring_e_e/`, `classification/`

Earlier PointNet / multi-ring regression and classification runs, predating the main architecture search. Safe to treat as archived/superseded.

### `tda/`

Single-ring TDA-fusion training jobs. See `tda_README.md` for full details.

| File pattern | What |
|---|---|
| `reg_e_tda_full_*.{job,sh}` | Base TDA fusion (concatenation), electrons |
| `reg_e_tda_*_amp.{job,sh}` | Mixed-precision (AMP) TDA fusion - although I believe I turned amp off for this later. I fixed the timing issue so amp should not be necessary, this script can be run as a baseline |
| `reg_e_tda_gated_*.{job,sh}` | Gated fusion variant |
| `reg_e_tda_crossattn_*.{job,sh}` | Cross-attention fusion variant |
| `*_mmap_smoke.{job,sh}` | Short smoke-test for memory-mapped data loading |
| `reg_mu_tda_*.{job,sh}` | Muon equivalents of the above |

### `multiring_v2/`

All multi-ring architecture development. Files generally follow the pattern:
`reg_2e_multiring_<model>_<variant>_<epochs>.{job,sh}`

#### Model versions

| Name | Description |
|---|---|
| `v1`, `v2` | Early prototype multi-ring heads |
| `votenet` | First VoteNet-style multi-ring head (unnumbered) |
| `votenet_v2` … `votenet_v11` | Iterative development of the PointNet++ + VoteNet-style architecture |
| `resnet_12M` | ResNet backbone on the 12M-event sample (ablation / baseline) |

#### Variant suffixes

| Suffix | Meaning |
|---|---|
| `smoke` | Short test run (a few hundred iterations) to check the job runs end-to-end |
| `10ep`, `20ep`, `60ep`, `180ep` | Training length in epochs |
| `costnorm` | Hungarian matching cost normalisation — the single clearest improvement in the architecture search |
| `groupnorm` | GroupNorm replacing BatchNorm (fixes training instability) |
| `noslot` | Ablation: no slot-ordering in the output head |
| `crossattn` | Cross-attention fusion variant |
| `selfattn_only` | Self-attention only (no cross-attention) ablation |
| `heads{1,2,4,8}` | Number of attention heads |
| `dirweight` | Increased direction-loss weight |
| `reg` | Regularisation tuning |
| `fixedassign_{5,20}ep` | Fixed-assignment diagnostic (tests matching loss / BatchNorm interaction) |
| `v2style` | V2-style regression head |
| `control` | Warm-start control (trains from scratch for same number of epochs as warm-start run) |
| `warmstart` | Warm-start from `warm_start_checkpoints/` |
| `reeval` | Re-evaluation of a previously trained model (no new training) |
| `_hx2.sh` | Run on Imperial HX2 cluster (no `.job` file; submitted via Nick Prouse) |

#### TDA multi-ring jobs

| File pattern | What |
|---|---|
| `*_v8_tda_perring_60ep.{job,sh}` | Oracle per-ring TDA features fed into V8 head |
| `*_v8_tda_perring_gated_60ep.{job,sh}` | Gated fusion of per-ring TDA features |
| `*_v8_tda_perring_gated_npy_smoke.{job,sh}` | Smoke-test for `.npy`-based TDA loading |
| `*_v8_tda_mergedist_60ep.{job,sh}` | Merge-distance TDA feature variant |
| `*_v8_tda_mergedist_gated_60ep.{job,sh}` | Gated fusion of merge-distance features |
| `reg_2e_multiring_v8_control_10ep.{job,sh}` | Warm-start control (10 ep from scratch) |
| `reg_2e_multiring_v8_tda_perring_warmstart_10ep.{job,sh}` | Warm-start TDA run (10 ep) |

These jobs require the multi-ring TDA features from `tda/`. See `tda_README.md`.

---

## Batch-job tracking

A spreadsheet maps every job name to its output directory, result, and description. It is uploaded to this repository as `batch_job_tracking.xlsx`.

---

## Caveats

- In the multi-ring analysis, "Ring 0 / Ring 1" are raw matched output slots, **not** sorted by true energy. Keep in mind when comparing leading/subleading ring results.
- Results of each run were transferred to HEP and analysis notebooks were kept on HEP.
- Clusters used: **fir** (SLURM, what this README refers to), **HEP** (HTCondor) and **HPC** (PBS). Work on the latter two lives elsewhere: `/vols/hyperk/users/ks2922/ML_2025` (HEP) & `/rds/general/user/ks2922/home/mres` (HPC). These should follow the same directory style as on fir.

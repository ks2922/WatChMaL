# TDA — Topological Data Analysis for WCTE Reconstruction

TDA pipeline for the MRes project "Topology-Informed Point Cloud Networks for Event
Reconstruction in Water Cherenkov Neutrino Detectors": what topological features are
computed, where the scripts actually live on fir, how to regenerate them, and which
training jobs consume them.

---

## What is TDA doing here?

Each WCTE event is a point cloud of PMT hit positions. Persistent homology, via a
Vietoris–Rips filtration, extracts topological features from that point cloud:

- **H0 (connected components):** how many clusters of hits exist, and at what radius they
  merge. Sensitive to the number of rings and their separation.
- **H1 (loops):** closed loops in the hit pattern. A Cherenkov ring produces a loop feature.

These are computed as **persistence diagrams** (one per event, per homology dimension: H0
and H1 only — the pipeline does not go to H2), then compressed into a fixed-length **800
number** feature vector: flattened H0 + H1 **persistence images**, 20×20 bins each.
The vector is not raw PMT channels, and is fused with the PointNet++/ResNet global feature
before the regression head.

**Libraries:** `giotto-ph` (persistence computation) and `giotto-tda`'s `PersistenceImage`
(vectorisation).

**Cutoff:** a 20 cm birth-radius cutoff is applied to exclude short-lived loops from
intra-mPMT-module PMT clustering (validated: noise sits below ~10 cm, plateau stable from
10–30 cm).

---

## Directory layout


```
/scratch/ks2922/mres/tda/          (all files directly here, no subdirectories)
```

**Diagram precompute (stage 1), single-ring:**
`precompute_tda_diagrams.py` (electrons), `precompute_tda_diagrams_mu.py` (muons), run via
`precompute_diagrams_e.job` / `precompute_diagrams_e_pmtlevel.job` (both call the same
script) and the `_mu` equivalents.

**Diagram precompute (stage 1), multi-ring:**
`precompute_tda_diagrams_multiring.py`, `precompute_tda_diagrams_multiring_oracle_perring.py`,
`precompute_tda_diagrams_multiring_wholeevent.py`, `precompute_merge_distance.py`, each with
a matching `.job`.

**Feature build (stage 2):**
`build_tda_features.py` / `build_tda_features_e_pmtlevel.py` / `build_tda_features_mu.py` /
`build_tda_features_mu_pmtlevel.py` / `build_tda_features_multiring.py` /
`build_tda_features_multiring_oracle_perring.py` / `build_tda_features_multiring_wholeevent.py`,
each with a matching `.job` (`build_features_*.job`).

**Legacy single-step scripts** (pre-dating the two-stage split, module-level, not PMT-level):
`precompute_tda_features.py`, `precompute_tda_features_full.py`, `precompute_tda_features_full_mu.py`
(jobs: `precompute_tda_full.job`, `precompute_tda_full_mu.job`). **[CHECK]** whether these are
still referenced anywhere or can be marked fully dead.

**Other scripts:**
`convert_perring_tda_to_npy.py`, `prepare_v8_warmstart_checkpoint.py`, `make_test_subset.py`
(small hit subset for smoke tests), `time_*.py` (benchmarking, not needed to reproduce
results).

**Smoke/cache (safe to ignore):** `cpu_smoke_test/`, `e_FC_tda_smoke.npz`, `diagram_cache*.npz`.

**Data outputs** live in this same flat folder too — see "Which files to use" below.

---

## Which files to use — the module-level trap

- `diagrams_e_pmtlevel.h5`, `diagrams_mu_pmtlevel.h5`: **correct**, PMT-level, 20 cm cutoff
  applied. **Use these.**
- `diagrams_e.h5`, `diagrams_mu.h5` (no suffix): **module-level** (97 module centroids, not
  PMT-level). Easy to mistake for the PMT-level files — the `_e.job` runs the *same current
  script* as `_e_pmtlevel.job` (an earlier version of the script is what actually produced
  the module-level file; the current script only writes the `_pmtlevel` output regardless of
  which job calls it). Do not use `diagrams_e.h5`/`diagrams_mu.h5` for results.
- Feature files exist as `.h5`, `.npz` and `.npy` per variant (e.g.
  `tda_features_full_e_pmtlevel.{h5,npz,npy}`). Use the **`.npy`** versions for training —
  they're memory-mapped so the DataLoader doesn't load the whole array per worker; loading
  the `.npz` in every worker caused out-of-memory crashes.

---

## Multi-ring TDA

Source data is a separate two-electron-gun simulation (`wcte_2e-_indep_vert_indep_energy_12M.h5`),
not the single-ring pgun files. Two feature representations:

1. **Oracle per-ring diagrams:** hits are separated by *truth-level* ring assignment before
   computing persistence, giving a ground-truth topological signature per ring.
   `precompute_diagrams_multiring_oracle_perring.job` → `diagrams_multiring_oracle_perring_e_pmtlevel.h5`
   → `build_features_multiring_oracle_perring.job` → `tda_features_multiring_oracle_perring_e_pmtlevel.h5`
   → `convert_perring_tda_to_npy.py` → `..._ring0.npy`, `..._ring1.npy`, `..._event_idxs.npy`.
2. **merge_distance:** the maximum H0 death value of the *whole-event* diagram — a cheap,
   truth-free proxy for how separable the rings are.
   `precompute_merge_distance.job`/`.py` → `merge_distance_multiring_e.h5`.
   Whole-event diagrams: `precompute_diagrams_multiring_wholeevent.job` →
   `diagrams_multiring_wholeevent_e_pmtlevel.h5`.

**Result (thesis finding):** TDA fusion consistently **degraded** multi-ring performance, even
though the network weighted the TDA features highly. However, this could be a consequence of BatchNorm instability in the PointNet++ backbone for multi-ring.

---

## Fusion into the network

Three fusion strategies, matched to job names in `~/mres/job_scripts/tda/`:

| Fusion type | Job name pattern |
|---|---|
| Base (concatenation) | `reg_{e,mu}_tda_full*` |
| Gated | `reg_{e,mu}_tda_gated_60ep` |
| Cross-attention | `reg_{e,mu}_tda_crossattn_60ep` |

Single-ring AMP (mixed precision) runs: `reg_{e,mu}_tda_60ep_amp`.
`reg_e_tda_mmap_smoke`: smoke test for the memory-mapped `.npy` loading fix — not a result run.

Base concatenation was best or equal-best for both particles with gated fusion. Cross-attention was by far the worst. 
Therefore only the basic concatenation and gated fusion were carried over to multi-ring.

---

## Running the simplest case (single-ring electrons)

```bash
cd /scratch/ks2922/mres/tda
sbatch precompute_diagrams_e_pmtlevel.job     # stage 1 -> diagrams_e_pmtlevel.h5
sbatch build_features_e_pmtlevel.job          # stage 2 -> tda_features_full_e_pmtlevel.{h5,npz,npy}
# stage 3: sbatch one of ~/mres/job_scripts/tda/reg_e_tda_*.job
```

Muons: same with `_mu_pmtlevel`. Wait for stage 1 to finish before submitting stage 2 (or use
`--dependency=afterok:<jobid>`).

---

## Environment on fir

- Container: `container_base_ml_v4.0.0_watchmal-pointnet2.sif` — used for training jobs; has
  `pointnet2_ops` already built in, so no manual build/patch is needed on fir's H100s.
- TDA precompute itself (`giotto-ph`/`giotto-tda`) runs via packages installed with
  `pip install --user` into `/scratch/ks2922/mres/pip_user_packages`
  (`PYTHONUSERBASE=/scratch/ks2922/mres/pip_user_packages`). Precompute
  jobs run directly on the login/compute node.
- `--cleanenv` is required on all Apptainer calls: without it, host TLS certificate settings
  leak in and break pip.
- Training jobs used an H100 MIG slice (`gpu:nvidia_h100_80gb_hbm3_2g.20gb:1`), 8 CPUs, 32G.

---

## Analysis notebook

`geometry_persistence_visualisation.ipynb` — for a given event index, shows the point cloud,
persistence diagram and persistence image from the precomputed files. This is located at
`/vols/hyperk/users/ks2922/ML_2025/scripts/analysis/tda/` on the HEP cluster, as is all other analysis work.

---

## Known caveats

- In multi-ring analysis, "Ring 0 / Ring 1" are raw matched output slots, not sorted into
  leading/subleading by true energy.
- `pointnet2_ops` hardcodes BatchNorm2d, so GroupNorm can only be applied to the FC heads.


---

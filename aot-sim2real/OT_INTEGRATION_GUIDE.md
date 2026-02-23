# Integration Guide: Adding OT-Sim2Real Algorithm to Robomimic v0.5.0

This guide describes how to integrate the Optimal Transport (OT) domain-alignment
algorithm from *"Generalizable Domain Adaptation for Sim-and-Real Policy
Co-Training"* ([GaTech-RL2/ot-sim2real](https://github.com/GaTech-RL2/ot-sim2real))
into your robomimic v0.5.0 at `/home/ubuntu/opt/robomimic/`.

> **Scope:** Algorithm only. Does NOT cover the paper's custom environments,
> assets, R2D2 dataset format, or experiment generation scripts. Covers only
> the OT alignment loss, paired dataset loader, dual-dataloader training loop,
> and config/registration plumbing.

---

## Prerequisites

### New Python Packages

```bash
pip install pot geomloss
```

- **`pot`** (Python Optimal Transport): provides `ot.unbalanced.sinkhorn_knopp_unbalanced`
- **`geomloss`**: provides `SamplesLoss("energy")` (only needed for the MMD baseline)

### Source Locations

| Label | Path |
|-------|------|
| **Your robomimic** | `/home/ubuntu/opt/robomimic/robomimic/` |
| **OT fork** | `/home/ubuntu/atz/aot-sim2real/robomimic/` |

---

## Step 1: Copy New Files (4 files)

These files do not exist in upstream robomimic.

### 1a. OT Diffusion Policy Algorithm

```
Source: aot-sim2real/robomimic/algo/diffusion_policy_ot.py  (776 lines)
Dest:   opt/robomimic/robomimic/algo/diffusion_policy_ot.py
```

Core contribution: `OTDiffusionPolicyUNet` class. Extends Diffusion Policy with
unbalanced OT alignment loss on observation encoder features. Includes an inlined
copy of `ConditionalUnet1D` (self-contained, does not import `diffusion_policy_nets.py`).

The OT class overrides two method signatures (self-contained, does not affect base classes):

- `process_batch_for_training(self, batch, B_ot)` -- extra `B_ot` arg
- `train_on_batch(self, batch, B_ot, ot_params, epoch, validate=False)` -- extra `B_ot`, `ot_params`

### 1b. OT Config

```
Source: aot-sim2real/robomimic/config/diffusion_policy_ot_config.py  (82 lines)
Dest:   opt/robomimic/robomimic/config/diffusion_policy_ot_config.py
```

`OTDiffusionPolicyConfig` with `ALGO_NAME = "diffusion_policy_ot"`. Adds:

- `train.ot.{pair_info_path, dataset, dataset_masks, batch_size}`
- `algo.ot.{sharpness, cutoff, window_size, emb_scale, cost_scale, reg, tau1, tau2, scale, heuristic, label}`

### 1c. OT Paired Dataset

```
Source: aot-sim2real/robomimic/utils/ot_dataset.py  (606 lines)
Dest:   opt/robomimic/robomimic/utils/ot_dataset.py
```

`ChunkSamplingDTWTrainDataset`: loads two HDF5 files (source + target domain),
returns paired `{src, tgt}` samples using precomputed DTW alignment info.

**Adaptation needed:** Imports `robomimic.utils.action_utils` which does not
exist in v0.5.0. Uses one function: `AcUtils.action_dict_to_vector()` (line 435).

- **Option A (recommended):** Copy `action_utils.py` too (see Step 3a).
- **Option B:** Replace the import/call with `np.concatenate(list(ac_dict.values()), axis=-1)`.

### 1d. OT Training Script

```
Source: aot-sim2real/robomimic/scripts/ot_sim2real/ot_train.py  (688 lines)
Dest:   opt/robomimic/robomimic/scripts/ot_train.py
```

Modified `train.py` that loads BC dataset + OT paired dataset, creates two
dataloaders, calls `TrainUtils.run_epoch_for_ot_policy()`, and visualizes
the OT transport plan as a heatmap.

**Adaptations needed:**

- Imports `deep_update` from non-existent `script_utils` -- see Step 3b.
- References `config.train.data_format` (not in v0.5.0) -- see Step 4e.
- Calls `TrainUtils.run_epoch_for_ot_policy()` -- added in Step 2c.

---

## Step 2: Modify Existing Files (3 files, all purely additive)

### 2a. Register OT Algorithm -- `algo/__init__.py`

**File:** `opt/robomimic/robomimic/algo/__init__.py`

Add at end of imports:

```python
from robomimic.algo.diffusion_policy_ot import OTDiffusionPolicyUNet
```

### 2b. Register OT Config -- `config/__init__.py`

**File:** `opt/robomimic/robomimic/config/__init__.py`

Add at end of imports:

```python
from robomimic.config.diffusion_policy_ot_config import OTDiffusionPolicyConfig
```

### 2c. Add OT Training Loop -- `utils/train_utils.py`

**File:** `opt/robomimic/robomimic/utils/train_utils.py`

Append these three functions at the end of the file (no existing code modified):

```python
def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def concat_two_batch(batch1: dict, batch2: dict):
    all_keys = list(set(list(batch1.keys()) + list(batch2.keys())))
    final_batch = {}
    for k in all_keys:
        if (k in batch1) and (k in batch2):
            if isinstance(batch1[k], dict):
                final_batch[k] = concat_two_batch(batch1[k], batch2[k])
            else:
                final_batch[k] = torch.cat([batch1[k], batch2[k]], dim=0)
        elif (k in batch1) and (k not in batch2):
            final_batch[k] = batch1[k]
        elif (k in batch2) and (k not in batch1):
            final_batch[k] = batch2[k]
        else:
            raise NotImplementedError
    return final_batch


def run_epoch_for_ot_policy(model, ot_params, bc_dataloader, ot_dataloader,
                            epoch, validate=False, num_steps=None,
                            obs_normalization_stats=None):
    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()

    assert num_steps is not None
    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[],
                        Log_Info=[])

    ot_dl = infinite_dataloader(ot_dataloader)
    bc_dl = infinite_dataloader(bc_dataloader)

    for _ in LogUtils.custom_tqdm(range(num_steps)):
        t = time.time()
        ot_batch = next(ot_dl)
        bc_batch = next(bc_dl)

        src_ot_batch = ot_batch["src"]
        tgt_ot_batch = ot_batch["tgt"]
        B_ot = src_ot_batch["actions"].shape[0]
        ot_batch = concat_two_batch(src_ot_batch, tgt_ot_batch)
        batch = concat_two_batch(ot_batch, bc_batch)
        timing_stats["Data_Loading"].append(time.time() - t)

        t = time.time()
        input_batch = model.process_batch_for_training(batch, B_ot)
        input_batch = model.postprocess_batch_for_training(
            input_batch, obs_normalization_stats=obs_normalization_stats
        )
        timing_stats["Process_Batch"].append(time.time() - t)

        t = time.time()
        info = model.train_on_batch(
            input_batch, B_ot, ot_params, epoch, validate=validate
        )
        timing_stats["Train_Batch"].append(time.time() - t)
        model.on_gradient_step()

        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    pi = step_log_dict.pop("pi", None)
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    for k in timing_stats:
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.
    if pi is not None:
        step_log_all["pi"] = pi[-1]

    return step_log_all
```

`time`, `numpy`, `torch` are already imported at the top of `train_utils.py`.

---

## Step 3: Add Small Utility Files

### 3a. `action_utils.py` (required by `ot_dataset.py`)

```
Source: aot-sim2real/robomimic/utils/action_utils.py  (36 lines)
Dest:   opt/robomimic/robomimic/utils/action_utils.py
```

Tiny file with `action_dict_to_vector` and `vector_to_action_dict`.

**Alternative:** Instead of copying, patch `ot_dataset.py` directly:

```python
# Remove:   import robomimic.utils.action_utils as AcUtils
# Replace line 435:
#   meta["actions"] = AcUtils.action_dict_to_vector(ac_dict)
# With:
#   meta["actions"] = np.concatenate(list(ac_dict.values()), axis=-1)
```

### 3b. `deep_update` utility (required by `ot_train.py`)

`ot_train.py` imports `deep_update` from `robomimic.utils.script_utils` (does
not exist in v0.5.0). Fix by defining it inline at the top of `ot_train.py`
and removing the import:

```python
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d
```

---

## Step 4: Compatibility Fixes (v0.3.0 fork vs. v0.5.0)

### 4a. `postprocess_batch_for_training` kwarg name

In `run_epoch_for_ot_policy` (Step 2c), the kwarg may need renaming.

- Fork uses: `obs_normalization_stats=`
- v0.5.0 uses: `normalization_stats=`

**Fix:** Check signature in `opt/robomimic/robomimic/algo/algo.py` and adjust
the kwarg name in `run_epoch_for_ot_policy` accordingly.

### 4b. Inlined UNet duplication

The OT file inlines `ConditionalUnet1D` rather than importing from
`diffusion_policy_nets.py`. No conflict, but two copies of the UNet now exist.
If you modify the UNet architecture, update both.

### 4c. `deserialize` method

The fork's `deserialize` does not accept `load_optimizers`. If v0.5.0 passes
this kwarg during checkpoint loading, add `**kwargs` to the OT class's
`deserialize` signature.

### 4d. `ot_dataset.py` leftover imports

```python
from robomimic.utils.dataset import action_stats_to_normalization_stats, _compute_traj_stats, _aggregate_traj_stats
```

These exist in v0.5.0 but are never called at runtime. Safe to remove if they
cause import errors.

### 4e. `ot_train.py` -- `config.train.data_format`

The script checks `config.train.data_format` for R2D2 support. Since you don't
use R2D2, either:

- Add `self.train.data_format = "robomimic"` to `BaseConfig.train_config()`, or
- Remove/comment out the `data_format` references in `ot_train.py`

---

## Step 5: Prepare Your Data

The OT algorithm requires three data inputs:

### 5a. Standard BC Dataset(s)

Your existing HDF5 demos. Config uses `train.data` as a list of dataset paths
with weights (same as your current `dp_stack_rs.json` setup).

### 5b. Paired OT Datasets

Two HDF5 files -- source domain (sim) and target domain (real) with demonstrations
that have been aligned via DTW:

```json
"train": {
    "ot": {
        "dataset": {
            "ot_src": "/path/to/source_domain.hdf5",
            "ot_tgt": "/path/to/target_domain.hdf5"
        },
        "batch_size": 32
    }
}
```

### 5c. DTW Pair Info JSON

Precomputed DTW alignments between source and target demonstrations:

```json
"train": {
    "ot": {
        "pair_info_path": "/path/to/pair_info.json"
    }
}
```

Expected structure:

```json
{
    "demo_0": {
        "paired_demos": [
            {"demo_name": "demo_42", "dtw_dist": 0.123},
            {"demo_name": "demo_17", "dtw_dist": 0.456}
        ],
        "pairing": {
            "demo_42": [[0, 1], [2, 3], [4, 5]],
            "demo_17": [[0], [1, 2], [3]]
        }
    }
}
```

Keys in `pairing` map target frame indices to lists of corresponding source
frame indices.

**Note:** The repo does not include a DTW computation script. You will need to
implement DTW alignment (e.g., using `tslearn` or `dtw-python`).

---

## Step 6: Create an Experiment Config

Based on your existing `dp_stack_rs.json`, create a new config:

```json
{
    "algo_name": "diffusion_policy_ot",

    "train": {
        "data": [
            {"path": "/path/to/bc_source.hdf5", "weight": 0.45},
            {"path": "/path/to/bc_target.hdf5", "weight": 0.1}
        ],
        "ot": {
            "pair_info_path": "/path/to/pair_info.json",
            "dataset": {
                "ot_src": "/path/to/paired_source.hdf5",
                "ot_tgt": "/path/to/paired_target.hdf5"
            },
            "dataset_masks": {},
            "batch_size": 32
        }
    },

    "algo": {
        "ot": {
            "sharpness": 1.0,
            "cutoff": 5.0,
            "window_size": 10,
            "emb_scale": 1.0,
            "cost_scale": 1.0,
            "reg": 0.01,
            "tau1": 1.0,
            "tau2": 1.0,
            "scale": 1.0,
            "heuristic": false,
            "label": "action"
        }
    }
}
```

### Key OT Hyperparameters

| Parameter | Description | Paper sweep |
|-----------|-------------|-------------|
| `reg` | Entropic regularization for Sinkhorn | `[1, 0.0001]` |
| `tau1`, `tau2` | Unbalanced OT marginal relaxation | swept together |
| `scale` | OT loss weight relative to BC loss | -- |
| `emb_scale` / `cost_scale` | Embedding vs label distance balance | -- |
| `label` | Label for structured cost | `"action"` or `"eef_pose"` |
| `window_size` | Temporal sampling window half-width | -- |
| `sharpness` | DTW distance-to-weight sigmoid steepness | -- |

---

## Step 7: Run Training

```bash
python robomimic/scripts/ot_train.py --config /path/to/your_ot_config.json
```

---

## File Change Summary

### New files to add:

| # | Destination (relative to `robomimic/`) | Lines | Purpose |
|---|----------------------------------------|-------|---------|
| 1 | `algo/diffusion_policy_ot.py` | 776 | OT diffusion policy algorithm |
| 2 | `config/diffusion_policy_ot_config.py` | 82 | OT config with hyperparams |
| 3 | `utils/ot_dataset.py` | 606 | DTW-paired dual-domain dataset |
| 4 | `scripts/ot_train.py` | 688 | OT training entry point |
| 5 | `utils/action_utils.py` *(optional)* | 36 | Action dict utilities |

### Existing files to modify (all purely additive):

| # | File | Change |
|---|------|--------|
| 1 | `algo/__init__.py` | Add 1 import line |
| 2 | `config/__init__.py` | Add 1 import line |
| 3 | `utils/train_utils.py` | Append 3 functions (~100 lines) |

**No files deleted or destructively modified.**

---

## Architecture Diagram

```
                      ot_train.py
                           |
            +--------------+--------------+
            |                             |
    BC DataLoader                  OT DataLoader
    (SequenceDataset)              (ot_dataset.py)
            |                             |
            |                      +------+------+
            |                      | {src, tgt}  |
            |                      +------+------+
            |                             |
            +-------------+---------------+
                          v
           run_epoch_for_ot_policy()        <-- train_utils.py
           batch = [OT_src | OT_tgt | BC]
           B_ot  = size of each OT sub-batch
                          |
                          v
           OTDiffusionPolicyUNet            <-- diffusion_policy_ot.py
           .train_on_batch()
                          |
                 obs_encoder (shared)
                    |             |
                    v             v
            features[:2*B_ot]   features[2*B_ot:]
            (OT src + tgt)      (BC samples)
                    |             |
                    v             v
            OT Alignment      Diffusion Loss
            Loss (Sinkhorn)   (MSE noise pred)
                    |             |
                    +------+------+
                           v
                 total = l2 + scale * ot_loss
```

# Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training

This repository provides the implementation for **Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training**.  
Our implementation builds on **MimicGen**, **Robosuite**, and **Robomimic**. To keep the codebase compact, we include a modified version of Robomimic in this repo. You must also install the official versions of Robosuite and MimicGen (following the installation instructions below).

## Installation
### Install MimicGen
```bash
conda create -n ot-sim2real python=3.8
conda activate ot-sim2real
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .
```
### Install Robosuite
```bash
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa
pip install -e .
```

### Install modified version of robomimic
```bash
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone
cd ot-sim2real
pip install -e .
```

### Install robosuite-model-zoo
```bash
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone https://github.com/touristCheng/model-zoo-mugs-only.git
cd model-zoo-mugs-only
pip install -e .
```


## Download Datasets
Specify the desired dataset directory in `robomimic/scripts/ot_sim2real/download_datasets.py`, and run the script

```bash
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
cd ot-sim2real
python robomimic/scripts/ot_sim2real/download_datasets.py
```

## Generate configs
Our proposed method is denoted as **`ot-sim2real`**.  
We also include four baselines:

- **`MMD`**
- **`cotrain`**
- **`source_only`**
- **`target_only`**

The experiment configs can be generated using `robomimic/scripts/ot_sim2real/generate_exp_configs.py`. Please specify the `DATA_ROOT`, `OUTPUT_DIR`, and `GENERATED_CONFIG_DIR`, and then run the script
```bash
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
cd ot-sim2real
python robomimic/scripts/ot_sim2real/generate_exp_configs.py
```
## Run training
For `ot-sim2real` and `MMD`, the training script is `robomimic/scripts/ot_sim2real/ot_train.py`.
```bash
# for "xxx_ot-sim2real.json" and "xxx_MMD.json"
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
cd ot-sim2real
python robomimic/scripts/ot_sim2real/ot_train.py --config path/to/config.json
```
For `cotrain`, `source_only` and `target_only`, the training script is `robomimic/scripts/train.py`.
```bash
# for "xxx_cotrain.json", "xxx_source_only.json" and "xxx_target_only.json"
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
cd ot-sim2real
python robomimic/scripts/train.py --config path/to/config.json
```



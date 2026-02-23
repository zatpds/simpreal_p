import json
import os
import copy

DATA_ROOT = None # Root directory containing the downloaded datasets and base config
OUTPUT_DIR = None # Output directory of training logs and models
GENERATED_CONFIG_DIR = None # Directory to save generated experiment configs

DTW_ROOT = os.path.join(DATA_ROOT, "dtw")
DATASET_ROOT = os.path.join(DATA_ROOT, "sim_demos")
BASE_CONFIG_PATH = os.path.join(DATA_ROOT, "base_config.json")

EXP_CONFIGS = [
        {
            "name": "box_bin",
            "paired": "box_bin_r2l_down",
            "unpaired": "box_bin_r2l_up",
            "DTW_name": "box_bin_pair_info"
        },
        {
            "name": "mug_hang",
            "paired": "mug_hang_r2l_down",
            "unpaired": "mug_hang_r2l_up",
            "DTW_name": "mug_hang_pair_info"
        },
        {
            "name": "mug_lift",
            "paired": "mug_lift_r",
            "unpaired": "mug_lift_l",
            "DTW_name": "mug_lift_pair_info"
        },
        {
            "name": "square",
            "paired": "square_r2l_down",
            "unpaired": "square_r2l_up",
            "DTW_name": "square_pair_info"
        },
        {
            "name": "stack",
            "paired": "stack_r2l_down",
            "unpaired": "stack_r2l_up",
            "DTW_name": "stack_pair_info"
        },
]

DOMAIN_SHIFTS = ["cam_pose", "table-wood"]
CAM_SUFFIX = "120x160"
    
def generate_all_robomimic_configs(log_wandb=True):
    assert OUTPUT_DIR is not None, "Please specify OUTPUT_DIR"
    assert DATASET_ROOT is not None, "Please specify DATASET_ROOT"
    assert GENERATED_CONFIG_DIR is not None, "Please specify GENERATED_CONFIG_DIR"

    config_root = GENERATED_CONFIG_DIR
    for exp in EXP_CONFIGS:
        for domain_shift in DOMAIN_SHIFTS:
            exp_name = f"{exp['name']}_{domain_shift}"
            config_dir = os.path.join(config_root, exp_name)
            os.makedirs(config_dir, exist_ok=True)

            # Create experiment configuration
            exp_config = {
                "exp_name": exp_name,
                "config_dir": config_dir,
                "paired_src_path": os.path.join(
                    DATASET_ROOT,
                    exp["paired"],
                    f"rgb_{CAM_SUFFIX}.hdf5",
                ),
                "unpaired_src_path": os.path.join(
                    DATASET_ROOT,
                    exp["unpaired"],
                    f"rgb_{CAM_SUFFIX}.hdf5",
                ),
                "paired_tgt_path": os.path.join(
                    DATASET_ROOT,
                    exp["paired"],
                    f"{domain_shift}_{CAM_SUFFIX}.hdf5",
                ),
                "unpaired_tgt_path": os.path.join(
                    DATASET_ROOT,
                    exp["unpaired"],
                    f"{domain_shift}_{CAM_SUFFIX}.hdf5",
                ),
                "dtw_path": os.path.join(
                    DTW_ROOT,
                    exp["DTW_name"]+".json",
                ),
            }

            generate_robomimic_configs_for_exp_config(exp_config, log_wandb=log_wandb)

def generate_robomimic_configs_for_exp_config(exp_config, log_wandb=False):
    config = json.load(open(BASE_CONFIG_PATH, "r"))
    config["train"]["output_dir"] = OUTPUT_DIR
    config["experiment"]["logging"]["log_wandb"] = log_wandb

    dataset_list = [
        {
            "path": exp_config["paired_src_path"],
            "eval": True,
            "key": "src_paired",
        },
        {
            "path": exp_config["unpaired_src_path"],
            "eval": True,
            "key": "src_unpaired",
        },
        {
            "path": exp_config["paired_tgt_path"],
            "eval": True,
            "key": "target_paired",
        },
        {
            "path": exp_config["unpaired_tgt_path"],
            "eval": True,
            "key": "target_unpaired",
        }
    ]

    # Modify the configurations for our method
    ot_sim2real_config = copy.deepcopy(config)
    ot_sim2real_config["experiment"]["name"] = exp_config["exp_name"] + "_ot-sim2real"
    ot_sim2real_config["train"]["data"] = list(dataset_list)
    for data_cfg in ot_sim2real_config["train"]["data"]:
        if data_cfg["key"] in ["src_paired", "src_unpaired"]:
            data_cfg["demo_limit"] = 500
            data_cfg["weight"] = 0.45
        elif data_cfg["key"] in ["target_paired"]:
            data_cfg["demo_limit"] = 10
            data_cfg["weight"] = 0.1
        elif data_cfg["key"] in ["target_unpaired"]:
            data_cfg["demo_limit"] = 0
            data_cfg["weight"] = 0.0
    
    ot_sim2real_config["train"]["ot"]["pair_info_path"] = exp_config["dtw_path"]
    ot_sim2real_config["train"]["ot"]["dataset"]["ot_src"] = exp_config["paired_src_path"]
    ot_sim2real_config["train"]["ot"]["dataset"]["ot_tgt"] = exp_config["paired_tgt_path"]
    json.dump(ot_sim2real_config, open(os.path.join(exp_config["config_dir"], exp_config["exp_name"] + "_ot-sim2real.json"), "w"), indent=4)

    # Modify the configurations for MMD
    mmd_config = copy.deepcopy(ot_sim2real_config)
    mmd_config["experiment"]["name"] = exp_config["exp_name"] + "_MMD"
    mmd_config["algo"]["ot"]["sharpness"] = 0.0
    mmd_config["algo"]["ot"]["window_size"] = 200
    mmd_config["algo"]["ot"]["no_window"] = True
    json.dump(mmd_config, open(os.path.join(exp_config["config_dir"], exp_config["exp_name"] + "_MMD.json"), "w"), indent=4)

    # Modify the configurations for co-training
    cotrain_config = copy.deepcopy(ot_sim2real_config)
    cotrain_config["experiment"]["name"] = exp_config["exp_name"] + "_cotrain"
    cotrain_config["algo_name"] = "diffusion_policy"
    cotrain_config["train"].pop("ot")
    cotrain_config["algo"].pop("ot")
    json.dump(cotrain_config, open(os.path.join(exp_config["config_dir"], exp_config["exp_name"] + "_cotrain.json"), "w"), indent=4)
    
    # Modify the configurations for source-only training
    source_only_config = copy.deepcopy(cotrain_config)
    source_only_config["experiment"]["name"] = exp_config["exp_name"] + "_source_only"
    source_only_config["train"]["data"] = list(dataset_list)
    for data_cfg in source_only_config["train"]["data"]:
        if data_cfg["key"] == "src_paired":
            data_cfg["weight"] = 0.5
        elif data_cfg["key"] == "src_unpaired":
            data_cfg["weight"] = 0.5
        elif data_cfg["key"] == "target_paired":
            data_cfg["demo_limit"] = 0
            data_cfg["weight"] = 0.0
        elif data_cfg["key"] == "target_unpaired":
            data_cfg["demo_limit"] = 0
            data_cfg["weight"] = 0.0
    json.dump(source_only_config, open(os.path.join(exp_config["config_dir"], exp_config["exp_name"] + "_source_only.json"), "w"), indent=4)

    # Modify the configurations for target-only training
    target_only_config = copy.deepcopy(cotrain_config)
    target_only_config["experiment"]["name"] = exp_config["exp_name"] + "_target_only"
    target_only_config["train"]["data"] = list([dataset_list[2]])
    for data_cfg in target_only_config["train"]["data"]:
        if data_cfg["key"] == "target_paired":
            data_cfg["demo_limit"] = 10
            data_cfg["weight"] = 1.0

    json.dump(target_only_config, open(os.path.join(exp_config["config_dir"], exp_config["exp_name"] + "_target_only.json"), "w"), indent=4)


if __name__ == "__main__":
    generate_all_robomimic_configs(log_wandb=False)
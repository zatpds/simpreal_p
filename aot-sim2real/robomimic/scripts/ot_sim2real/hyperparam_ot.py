"""
A useful script for generating json files and shell scripts for conducting parameter scans.
The script takes a path to a base json file as an argument and a shell file name.
It generates a set of new json files in the same folder as the base json file, and 
a shell file script that contains commands to run for each experiment.

Instructions:

(1) Start with a base json that specifies a complete set of parameters for a single 
    run. This only needs to include parameters you want to sweep over, and parameters
    that are different from the defaults. You can set this file path by either
    passing it as an argument (e.g. --config /path/to/base.json) or by directly
    setting the config file in @make_generator. The new experiment jsons will be put
    into the same directory as the base json.

(2) Decide on what json parameters you would like to sweep over, and fill those in as 
    keys in @make_generator below, taking note of the hierarchical key
    formatting using "/" or ".". Fill in corresponding values for each - these will
    be used in creating the experiment names, and for determining the range
    of values to sweep. Parameters that should be sweeped together should
    be assigned the same group number.

(3) Set the output script name by either passing it as an argument (e.g. --script /path/to/script.sh)
    or by directly setting the script file in @make_generator. The script to run all experiments
    will be created at the specified path.

Args:
    config (str): path to a base config json file that will be modified to generate config jsons.
        The jsons will be generated in the same folder as this file.

    script (str): path to output script that contains commands to run the generated training runs

Example usage:

    # assumes that /tmp/gen_configs/base.json has already been created (see quickstart section of docs for an example)
    python hyperparam_helper.py --config /tmp/gen_configs/base.json --script /tmp/gen_configs/out.sh
"""
import argparse
import os
import glob
import json

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils

from robomimic.scripts.pixel_sim2real.generate_exp_configs import generate_sbatch_scripts

def make_generator(config_file, script_file):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file
    )

    # ot label
    # generator.add_param(
    #     key="algo.ot.label",
    #     name="label", 
    #     group=1, 
    #     values=["action", "eef_pose"],
    # )

    # # regularization
    # generator.add_param(
    #     key="algo.ot.reg", 
    #     name="reg",
    #     group=2, 
    #     values=[0.02, 0.01, 0.005], 
    # )

    # generator.add_param(
    #     key="algo.ot.tau1", 
    #     name="tau1",
    #     group=2, 
    #     values=[0.01, 0.005, 0.0025], 
    # )

    # generator.add_param(
    #     key="algo.ot.tau2", 
    #     name="tau2",
    #     group=2, 
    #     values=[0.01, 0.005, 0.0025], 
    # )

    # # scale
    # generator.add_param(
    #     key="algo.ot.scale", 
    #     name="scale", 
    #     group=3, 
    #     values=[0.2, 0.1, 0.05], 
    # )

    # regularization
    # generator.add_param(
    #     key="algo.ot.reg", 
    #     name="reg",
    #     group=2, 
    #     values=[0.02, 0.05], 
    # )

    # generator.add_param(
    #     key="algo.ot.tau1", 
    #     name="tau1",
    #     group=2, 
    #     values=[0.01, 0.025], 
    # )

    # generator.add_param(
    #     key="algo.ot.tau2", 
    #     name="tau2",
    #     group=2, 
    #     values=[0.01, 0.025], 
    # )

    # # scale
    # generator.add_param(
    #     key="algo.ot.scale", 
    #     name="scale", 
    #     group=3, 
    #     values=[0.05, 0.02], 
    # )

    # box bin
    # generator.add_param(
    #     key="algo.ot.label",
    #     name="label", 
    #     group=1, 
    #     values=["action", "eef_pose"],
    # )

    # generator.add_param(
    #     key="algo.ot.cost_scale",
    #     name="label", 
    #     group=1, 
    #     values=[0.05, 5.0],
    # )

    # generator.add_param(
    #     key="algo.ot.reg", 
    #     name="reg",
    #     group=2, 
    #     values=[0.02, 0.01], 
    # )

    # generator.add_param(
    #     key="algo.ot.tau1", 
    #     name="tau1",
    #     group=2, 
    #     values=[0.01, 0.005], 
    # )

    # generator.add_param(
    #     key="algo.ot.tau2", 
    #     name="tau2",
    #     group=2, 
    #     values=[0.01, 0.005], 
    # )

    # # scale
    # generator.add_param(
    #     key="algo.ot.scale", 
    #     name="scale", 
    #     group=3, 
    #     values=[0.1, 0.05, 0.02], 
    # )

    # stack
    # generator.add_param(
    #     key="algo.ot.reg", 
    #     name="reg",
    #     group=2, 
    #     values=[0.02, 0.01], 
    # )

    # generator.add_param(
    #     key="algo.ot.tau1", 
    #     name="tau1",
    #     group=2, 
    #     values=[0.01, 0.005], 
    # )

    # generator.add_param(
    #     key="algo.ot.tau2", 
    #     name="tau2",
    #     group=2, 
    #     values=[0.01, 0.005], 
    # )

    # # scale
    # generator.add_param(
    #     key="algo.ot.scale", 
    #     name="scale", 
    #     group=3, 
    #     values=[0.1, 0.05], 
    # )

    # reg
    # generator.add_param(
    #     key="algo.ot.reg", 
    #     name="reg",
    #     group=1, 
    #     values=[0.1, 0.04, 0.02, 0.005, 0.001], 
    # )

    # tau
    # generator.add_param(
    #     key="algo.ot.tau1", 
    #     name="tau1",
    #     group=2, 
    #     values=[0.1, 0.04, 0.02, 0.01, 0.001], 
    # )

    # generator.add_param(
    #     key="algo.ot.tau2", 
    #     name="tau2",
    #     group=2, 
    #     values=[0.1, 0.04, 0.02, 0.01, 0.001], 
    # )

    # winsize
    # generator.add_param(
    #     key="algo.ot.window_size", 
    #     name="winsize",
    #     group=2, 
    #     values=[40, 20, 5], 
    # )

    # extreme
    # reg
    generator.add_param(
        key="algo.ot.reg", 
        name="reg",
        group=1, 
        values=[1, 0.0001], 
    )

    # tau
    # generator.add_param(
    #     key="algo.ot.tau1", 
    #     name="tau1",
    #     group=2, 
    #     values=[1.0, 0.0001], 
    # )

    # generator.add_param(
    #     key="algo.ot.tau2", 
    #     name="tau2",
    #     group=2, 
    #     values=[1.0, 0.0001], 
    # )

    # winsize
    # generator.add_param(
    #     key="algo.ot.window_size", 
    #     name="winsize",
    #     group=2, 
    #     values=[120, 1], 
    # )

    return generator


def main(args):

    # make config generator
    generator = make_generator(config_file=args.config, script_file=args.script)

    # generate jsons and script
    generator.generate()

    # generate sbatch script
    config_dir = os.path.dirname(args.config)
    sbatch_dir = os.path.join(config_dir, "slurm_scripts")
    os.makedirs(sbatch_dir, exist_ok=True)

    # Generate sbatch script for each config
    main_python_path = "robomimic/scripts/pixel_sim2real/cotrain.py"
    configs = glob.glob(os.path.join(config_dir, "*.json"))
    sbatch_paths = []
    for config_path in configs:
        config = json.load(open(config_path, "r"))
        exp_name = config["experiment"]["name"]
        sbatch_file_name = f"submit_{exp_name}"
        sbatch_path = generate_sbatch_scripts(main_python_path, exp_name, config_path, sbatch_dir, sbatch_file_name)
        sbatch_paths.append(sbatch_path)
    
    # Generate bash script to run all sbatch scripts
    bash_script_path = os.path.join(config_dir, "run_all_sbatch.sh")
    with open(bash_script_path, "w") as f:
        for sbatch_path in sbatch_paths:
            f.write(f"sbatch {sbatch_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to base json config - will override any defaults.
    parser.add_argument(
        "--config",
        type=str,
        help="path to base config json that will be modified to generate jsons. The jsons will\
            be generated in the same folder as this file.",
    )

    # Script name to generate - will override any defaults
    parser.add_argument(
        "--script",
        type=str,
        help="path to output script that contains commands to run the generated training runs",
        default="/tmp/train.sh",
    )

    args = parser.parse_args()
    main(args)

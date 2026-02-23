import os
import time
import json

from robomimic.utils.slurm_rollout_utils import evaluate_trained_agent

        
def evaluate_ckpt_for_all_envs(ckpt_path, dataset_cfgs, slurm_rollout_path, horizon=500):
    with open(os.path.join(slurm_rollout_path, "slurm_rollout.txt"), "a") as f:
        f.write(f"\nCkpt {os.path.basename(ckpt_path).split('.')[0]}:\n")
    for dataset_cfg in dataset_cfgs:
        do_eval = dataset_cfg.get("eval", True)
        if do_eval is not True:
            continue
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        env_key=os.path.splitext(os.path.basename(dataset_path))[0] if not dataset_cfg.get('key', None) else dataset_cfg['key']
        num_rollouts = dataset_cfg.get("num_rollouts", 50)
        start_time = time.time()
        success_rate, video_path = evaluate_trained_agent(
            agent_path=os.path.join(ckpt_path),
            env_key=env_key,
            env_meta_dataset_path=dataset_path,
            num_parallel_jobs=50,
            rollouts_per_job=1,
            total_rollouts=num_rollouts,
            epoch_log_dir=os.path.join(slurm_rollout_path, f"{os.path.basename(ckpt_path).split('.')[0]}"),
            rollout_horizon=horizon,
            num_videos=50,
            video_skip=1,
        )
        end_time = time.time()
        print(f"Slurm Rollout took {end_time - start_time} seconds")
        print(f"Ckpt {os.path.basename(ckpt_path).split('.')[0]} Env {env_key} Rollout Success Rate: {success_rate}")

        with open(os.path.join(slurm_rollout_path, "slurm_rollout.txt"), "a") as f:
            f.write(f" - Env {env_key} Rollout Success Rate: {success_rate}\n")


def evaluate_agents(ckpt_paths, dataset_cfgs, slurm_rollout_path, horizon):
    if isinstance(slurm_rollout_path, str):
        slurm_rollout_path = [slurm_rollout_path]
        slurm_rollout_path *= len(ckpt_paths)
    elif isinstance(slurm_rollout_path, list):
        assert len(slurm_rollout_path) == len(ckpt_paths), "slurm_rollout_path should be a single path or a list with the same length as ckpt_paths"
    for i in range(len(ckpt_paths)):
        ckpt_path = ckpt_paths[i]
        evaluate_ckpt_for_all_envs(ckpt_path, dataset_cfgs, slurm_rollout_path[i], horizon)

if __name__ == "__main__":
    ckpt_paths = [
        # "/nethome/lma326/sim2real_ws/data/logs/box_bin_cam_pose_MMD_no_window/20250514185133/models/model_epoch_50.pth",
        # "/nethome/lma326/sim2real_ws/data/logs/box_bin_cam_pose_target_only/20250512110908/models/model_epoch_250.pth",
        "/nethome/lma326/sim2real_ws/data/logs/box_bin_cam_pose_cotrain/20250512110910/models/model_epoch_250.pth",
        # "/nethome/lma326/sim2real_ws/data/logs/box_bin_cam_pose/20250512131633/models/model_epoch_300.pth",
    ]

    # ckpt_paths = [
    #     # "/nethome/lma326/sim2real_ws/data/logs/box_bin_table_wood/20250512131630/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/box_bin_table_wood_cotrain/20250512131630/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/box_bin_table_wood_MMD_no_window/20250514185437/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/box_bin_table_wood_target_only/20250512131633/models/model_epoch_250.pth"
    # ]

    ckpt_paths = [
        # "/nethome/lma326/sim2real_ws/data/logs/stack_ud_cam_pose/20250508151411/models/model_epoch_250.pth",
        # "/nethome/lma326/sim2real_ws/data/logs/stack_ud_cam_pose_MMD_no_window/20250514185842/models/model_epoch_250.pth",
        # "/nethome/lma326/sim2real_ws/data/logs/stack_ud_cam_pose_cotrain/20250526215324/models/model_epoch_250.pth",
        # "/nethome/lma326/sim2real_ws/data/logs/stack_ud_cam_pose_target_only/20250526215335/models/model_epoch_250.pth"
        "/nethome/lma326/sim2real_ws/data/logs/stack_ud_cam_pose_target_only/20250526215335/models/model_epoch_300.pth"
    ]

    # ckpt_paths = [
    #     "/nethome/lma326/sim2real_ws/data/logs/stack_ud_table_wood_reg_0.01/20250512153051/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/stack_ud_table_wood_target_only/20250510002736/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/stack_ud_table_wood_MMD_no_window/20250514185937/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/stack_ud_table_wood_cotrain/20250510002732/models/model_epoch_250.pth"
    # ]

    # ckpt_paths = [
    #     "/nethome/lma326/sim2real_ws/data/logs/square_ud_cam_pose/20250508233638/models/model_epoch_300.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/square_ud_cam_pose_cotrain/20250508233640/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/square_ud_cam_pose_MMD_no_window/20250514185437/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/square_ud_cam_pose_target_only/20250508233638/models/model_epoch_250.pth",
    # ]

    # ckpt_paths = [
    #     "/nethome/lma326/sim2real_ws/data/logs/square_ud_table_wood/20250510002835/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/square_ud_table_wood_cotrain/20250510002835/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/square_ud_table_wood_MMD_no_window/20250514185638/models/model_epoch_250.pth",
    #     "/nethome/lma326/sim2real_ws/data/logs/square_ud_table_wood_target_only/20250510002843/models/model_epoch_250.pth"
    # ]

    # config_path = "/nethome/lma326/sim2real_ws/data/logs/stack_ud_table_wood_reg_0.01/20250512153051/config.json"
    # config_path = "/nethome/lma326/sim2real_ws/data/logs/stack_ud_table_wood_cotrain/20250510002732/config.json"
    # config_path = "/nethome/lma326/sim2real_ws/data/logs/box_bin_cam_pose_MMD_no_window/20250514185133/config.json"
    # config_path = "/nethome/lma326/sim2real_ws/data/logs/box_bin_table_wood_MMD_no_window/20250514185437/config.json"
    config_path = "/nethome/lma326/sim2real_ws/data/logs/stack_ud_cam_pose/20250508151411/config.json"
    # config_path = "/nethome/lma326/sim2real_ws/data/logs/stack_ud_table_wood_cotrain/20250510002732/config.json"
    # config_path = "/nethome/lma326/sim2real_ws/data/logs/square_ud_cam_pose/20250508233638/config.json"
    # config_path = "/nethome/lma326/sim2real_ws/data/logs/square_ud_table_wood/20250510002835/config.json"
    config = json.load(open(config_path, "r"))

    dataset_cfgs = config["train"]["data"]
    selected_cfgs = []
    for data_cfg in dataset_cfgs:
        if data_cfg["key"] == "src_paired":
            continue
        elif data_cfg["key"] == "src_unpaired":
            continue
        else:
            selected_cfgs.append(data_cfg)
    dataset_cfgs = selected_cfgs


    slurm_rollout_paths = []
    for ckpt_path in ckpt_paths:
        slurm_rollout_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "offline_rollouts")
        os.makedirs(slurm_rollout_path, exist_ok=True)
        slurm_rollout_paths.append(slurm_rollout_path)

    # Evaluate the agents
    evaluate_agents(ckpt_paths, dataset_cfgs, slurm_rollout_paths, horizon=500)


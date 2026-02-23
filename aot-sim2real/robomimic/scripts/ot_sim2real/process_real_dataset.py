"""
Script to resize and crop iamges in the real dataset. Rename to robomimic

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

Example usage:
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
import cv2

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils

OBS_REAL_TO_ROBOMIMIC = {
    "agentview_depth": "agentview_depth",
    "agentview_image": "agentview_image",
    "eef_pos": "robot0_eef_pos",
    "eef_quat": "robot0_eef_quat",
    "gripper_position": "robot0_gripper_qpos",
    "joint_positions": "robot0_joint_pos",
}

# real obs keys 'agentview_depth', 'agentview_image', 'eef_axis_angle', 'eef_pos', 'eef_pose', 'eef_quat', 'gripper_position', 'joint_positions'
def resize_images(images, height, width):
    """
    Resize images to the specified height and width.
    Args:
        images (np.ndarray): Array of images to resize. Shape: (N, H, W, C) or (N, H, W)
        height (int): Desired height of the output images.
        width (int): Desired width of the output images.
    Returns:
        np.ndarray: Resized and cropped images.
    """
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        resized_images.append(resized_img)

    return np.array(resized_images)


def dataset_states_to_obs(args):
    # create environment to use for data processing

    if args.debug:
        os.makedirs(os.path.join(os.path.dirname(args.dataset), "debug_imgs"), exist_ok=True)


    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_name = args.output_name
    if output_name is None:
        output_name = os.path.basename(args.dataset)[:-5] + "_{}x{}.hdf5".format(args.camera_height, args.camera_width)

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    # copy over the environment metadata from the sim dataset
    sim_f = h5py.File(args.sim_dataset, "r")
    env_meta = json.loads(sim_f["data"].attrs["env_args"])
    env_meta["env_kwargs"]["camera_heights"] = args.camera_height
    env_meta["env_kwargs"]["camera_widths"] = args.camera_width
    data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4) # environment info

    total_samples = 0
    for ind in range(len(demos)):
        ep = demos[ind]

        if args.debug:
            print("keys in {}: {}".format(ep, f["data/{}".format(ep)].keys())) # 'actions', 'control_enabled', 'delta_actions', 'dones', 'obs', 'rewards'
        joint_configs = f["data/{}/actions".format(ep)][()]

        obs = f["data/{}/obs".format(ep)]
        rewards = f["data/{}/rewards".format(ep)][()]
        dones = f["data/{}/dones".format(ep)][()]
        if args.debug:
            print("obs keys in {}: {}".format(ep, obs.keys())) # 'agentview_depth', 'agentview_image', 'eef_axis_angle', 'eef_pos', 'eef_pose', 'eef_quat', 'gripper_position', 'joint_positions'
            print("obs shapes in {}: {}".format(ep, {k: obs[k].shape for k in obs.keys()})) 

        # store transitions

        ep_data_grp = data_grp.create_group(ep)

        # rename the keys to match the keys in robomimic
        ep_data_grp.create_dataset("rewards", data=rewards)
        ep_data_grp.create_dataset("dones", data=dones)

        ep_data_grp.create_dataset("joint_configs", data=joint_configs)

        if "data/{}/abs_eef_traj".format(ep) in f.keys():
            abs_eef_traj = f["data/{}/abs_eef_traj".format(ep)][()]
            ep_data_grp.create_dataset("abs_eef_traj", data=np.array(abs_eef_traj))

        offset = np.array([-0.56, 0, 0.912])

        for k in OBS_REAL_TO_ROBOMIMIC.keys():
            np_array = np.array(obs[k])
            if k == "joint_positions":
                np_array = np_array[:, :7] # only keep the first 7 joint positions
            if k == "gripper_position":
                np_array = np.stack([np_array*0.5, -np_array*0.5], axis=-1) # stack the gripper position and orientation
            if k == "eef_pos":
                np_array[:] += offset
                if args.debug:
                    print("eef_pos before offset: {}".format(obs[k][0]))
                    print("eef_pos after offset: {}".format(np_array[0]))
            if k == "agentview_image" or k == "agentview_depth":
                # if args.debug:
                #     # save the original images for debugging
                #     original_image_path = os.path.join(os.path.dirname(args.dataset), "original_{}_{}.png".format(ep, k))
                #     if np_array.shape[-1] == 3:
                #         img = cv2.cvtColor(np_array[0], cv2.COLOR_RGB2BGR)
                #         cv2.imwrite(original_image_path, img)

                # resize the images to the specified height and width
                np_array = resize_images(np_array, args.camera_height, args.camera_width)
                if args.debug and k == "agentview_image":
                    # save the resized images for debugging
                    resized_image_path = os.path.join(os.path.dirname(args.dataset), "debug_imgs/resized_{}_{}.png".format(ep, k))
                    img = cv2.cvtColor(np_array[0], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(resized_image_path, img)
                if len(np_array.shape) == 3:
                    # (N, H, W) -> (N, H, W, 1)
                    np_array = np.expand_dims(np_array, axis=-1)
                    print("np_array shape after expand dims: {}".format(np_array.shape))
            if args.compress:
                ep_data_grp.create_dataset("obs/{}".format(OBS_REAL_TO_ROBOMIMIC[k]), data=np_array, compression="gzip")
            else:
                ep_data_grp.create_dataset("obs/{}".format(OBS_REAL_TO_ROBOMIMIC[k]), data=np_array)

        # episode metadata
        ep_data_grp.attrs["num_samples"] = joint_configs.shape[0] # number of transitions in this episode
        total_samples += joint_configs.shape[0]
        print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))

    if args.debug:
        # print new obs shapes
        print("shapes after processing in {}: {}".format(ep, {k: ep_data_grp["obs"][k].shape for k in ep_data_grp["obs"].keys()}))

    # global metadata
    data_grp.attrs["total"] = total_samples
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--sim_dataset",
        type=str,
        required=True,
        help="path to input sim dataset for copying env_meta",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=120,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=160,
        help="(optional) width of image observations",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        action='store_true',
        help="(optional) compress observations with gzip option in hdf5",
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        help="(optional) print out information about each episode as it is processed",
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)

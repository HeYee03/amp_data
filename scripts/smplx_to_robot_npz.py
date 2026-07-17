import argparse
import pathlib
import os
import time

import numpy as np
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print


def _build_amp_motion_data(qpos_list, aligned_fps, kinematics_model, device="cuda:0"):
    qpos_list = np.asarray(qpos_list)
    root_pos = qpos_list[:, :3].copy()
    root_rot = qpos_list[:, 3:7].copy()
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:].copy()
    num_frames = root_pos.shape[0]

    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos,
        fk_root_rot,
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
    )

    body_pos, _ = kinematics_model.forward_kinematics(
        torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
        torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
    )
    lowest_height = torch.min(body_pos[..., 2]).item()
    root_pos[:, 2] = root_pos[:, 2] - lowest_height
    root_pos[:, :2] -= root_pos[0, :2]

    return {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": np.array(kinematics_model.body_names),
    }


if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "kuavo_s45",
                 "hightorque_hi"],
        default="unitree_g1",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )

    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--disable_hand_collision_avoidance",
        default=False,
        action="store_true",
        help="Disable hand collision avoidance to allow hand overlap.",
    )

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )

    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)

    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        enable_hand_collision_avoidance=not args.disable_hand_collision_avoidance,
    )

    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4", )

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break

        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        # Update task targets.
        smplx_data = smplx_data_frames[i]

        # retarget
        qpos = retarget.retarget(smplx_data)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )

        if args.save_path is not None:
            qpos_list.append(qpos)

    if args.save_path is not None:
        # 处理文件扩展名
        if not args.save_path.endswith('.npz'):
            args.save_path = os.path.splitext(args.save_path)[0] + ".npz"

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kinematics_model = KinematicsModel(retarget.xml_file, device=device)
        data_to_save = _build_amp_motion_data(qpos_list, aligned_fps, kinematics_model, device=device)

        print(f"Saving to {args.save_path}...")

        # 使用 **kwargs 解包字典进行保存
        np.savez_compressed(args.save_path, **data_to_save)

        print(f"Saved to {args.save_path}")



    # if args.save_path is not None:
    #     if not args.save_path.endswith('.npz'):
    #         args.save_path = os.path.splitext(args.save_path)[0] + ".npz"
    #
    #     root_pos = np.array([qpos[:3] for qpos in qpos_list])
    #     # save from wxyz to xyzw
    #     root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    #     dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    #
    #
    #     local_body_pos = np.array(None)
    #     body_names = np.array(None)
    #
    #     print(f"Saving to {args.save_path}...")
    #
    #
    #     np.savez_compressed(
    #         args.save_path,
    #         fps=aligned_fps,
    #         root_pos=root_pos,
    #         root_rot=root_rot,
    #         dof_pos=dof_pos,
    #         local_body_pos=local_body_pos,
    #         link_body_list=body_names,
    #     )
    #     print(f"Saved to {args.save_path}")
    #
    # robot_motion_viewer.close()

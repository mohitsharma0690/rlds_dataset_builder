import numpy as np
import os
from pathlib import Path

import pickle
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


data_paths = [
    (
        [
            '/home/mohit/experiment_results/object_centric/real_world/greenblock_peg_insert_May7/',
            '/home/mohit/experiment_results/object_centric/real_world/try_02_May_8',
            '/home/mohit/experiment_results/object_centric/real_world/try_03_May_10',
        ],
        'Insert greeen block.',
    ),
    (
        [
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/blue_block_00_May_14',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/blue_block_01_May_19',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/blue_block_02_May_20',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/blue_block_reach_robust_00_May_21'
        ],
        'Pick up blue block.',
    ),
    (
        [
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/blue_stick_01_May_19',
        ],
        'Pick up narrow blue stick.',
    ),
    (
        [
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/green_block_00_May_14',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/green_block_00_May_15',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/green_block_reach_robust_00_May_17'
        ],
        'Pick up green block.',
    ),
    (
        [
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/pink_flower_00_May_19',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/pink_holes_00_May_19',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/pink_holes_01_May_20/',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/pink_holes_reach_robust_00_May_21'
        ],
        'Pick up pink flower.',
    ),
    (
        [
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/yellow_block_00_May_14',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks/yellow_block_00_May_16',
        ],
        'Pick up yellow block.',
    ),
    # Different static camera view
    (
        [
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks_topdown/green_block_topdown_00_May_23',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks_topdown/green_block_topdown_reach_robust_00_May_24',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks_topdown/green_block_reach_robust_01_May_25',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks_topdown/green_block_reach_robust_02_Jun_4',
        ],
        'Pick up green block.',
    ),
    (
        [
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks_topdown/pink_block_topdown_00_May_23',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks_topdown/pink_holes_topdown_reach_robust_00_May_24',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks_topdown/pink_holes_reach_robust_01_May_25',
            '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks_topdown/pink_holes_reach_robust_02_Jun_4',
        ],
        'Pick up pink block.',
    ),
]


def convert_quaternion_to_euler(quat: np.ndarray):
    # Since we use pyquaternion, we need to convert to scipy rotation
    # r = Rotation.from_quat([quat[-1], quat[0], quat[1], quat[2]])
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return r.as_euler('zyx')


def load_image(image_path: Path, resize_ratio: float = 1.0):
    img = Image.open(str(image_path))
    if resize_ratio < 1.0:
        img = img.resize([int(resize_ratio * s) for s in img.size], Image.LANCZOS)

    return np.asarray(img)[:, :, :3]


def load_demo_data(demo_path: Path, lang_instruction: str, load_images: bool = False):
    info_pickle_path = demo_path / 'info.pkl'
    static_img_path = demo_path / 'static'
    hand_img_path = demo_path / 'hand'

    static_img_resize_ratio = 0.5
    hand_img_resize_ratio = 0.5

    with open(str(info_pickle_path), 'rb') as f:
        low_dim_data = pickle.load(f)
    
    images = sorted([p for p in static_img_path.iterdir() if p.name.endswith('.png')])
    hand_images = sorted([p for p in hand_img_path.iterdir() if p.name.endswith('.png')])

    def _get_action_at_step(step: int):
        O_T_EE = np.array(low_dim_data['O_T_EE'][step + 1]).reshape(4, 4).T

        ee_pos = O_T_EE[:3, -1]
        try:
            ee_quat = Quaternion(matrix=O_T_EE[:3, :3], rtol=1e-4, atol=1e-4).elements
        except ValueError:
            print("Rot matrix not in SO(3). Will lower precision and try again.")
            ee_quat = Quaternion(matrix=O_T_EE[:3, :3], rtol=1e-3, atol=1e-3).elements
        ee_quat = ee_quat / np.linalg.norm(ee_quat)

        # Gripper: 1 -> open, 0 -> close
        if 'Insert' in lang_instruction:
            ee_gripper = [0]
        else:
            ee_gripper = low_dim_data['gripper_open'][step + 1]

        return np.r_[ee_pos, ee_quat, ee_gripper].astype(np.float32)

    def _get_state_at_step(step: int):
        # Gripper status: 1 -> open, 0 -> close
        if 'Insert' in lang_instruction:
            gripper_status = [0]
        else:
            gripper_status = low_dim_data['gripper_open'][step]
        state = np.r_[low_dim_data['q'][step],
                      gripper_status,
                      low_dim_data['O_F_ext_hat_K'][step],
                      low_dim_data['K_F_ext_hat_K'][step],]
        return state.astype(np.float32)
    
    episode_len = len(low_dim_data['O_T_EE'])
    for t in range(episode_len - 1):
        data_t = {
            'observation': {
                'image': load_image(images[t], resize_ratio=static_img_resize_ratio) if load_images else images[t],
                'wrist_image': load_image(hand_images[t], resize_ratio=hand_img_resize_ratio) if load_images else hand_images[t],
                'state': _get_state_at_step(t),
            },
            'action': _get_action_at_step(t),
            'discount': 1.0,
            'reward': float(t == (episode_len - 1)),
            'is_first': t == 0,
            'is_last': t == (episode_len - 1),
            'is_terminal': t == (episode_len - 1),
            'language_instruction': lang_instruction 
            # 'language_embedding': language_embedding,
        }
        yield t, data_t

SKIP_DATA = [
    '/home/mohit/experiment_results/object_centric/real_world/pickup_blocks_topdown/pink_holes_reach_robust_02_Jun_4/demo_0007',
]


def main():
    step_count, episode_count = 0, 0
    for multi_task_paths, lang_instruction in data_paths:
        for data_path in multi_task_paths:
            data_dir = Path(data_path)
            assert data_dir.exists(), f'{data_dir} does not exist'
            print("Will convert data from ", data_dir)
            for demo_path in data_dir.iterdir():
                if demo_path.name.startswith('demo') and str(demo_path) not in SKIP_DATA:
                    for step, step_data in load_demo_data(Path(demo_path), lang_instruction):
                        step_count += 1
                    episode_count += 1
            print(f"Did convert data. Total episodes: {episode_count} steps: {step_count}")

    print(f"Total episodes: {episode_count} steps: {step_count}")


if __name__ == '__main__':
    main()

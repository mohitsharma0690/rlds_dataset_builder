from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from pathlib import Path


class IamlabCmuPickupInsertDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for IAM Lab CMU's multi-task Picup and insertion dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            # shape=(720, 1280, 3),
                            shape=(360, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            # shape=(480, 640, 3),
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        # state -- q (7), gripper status (1), joint-torques (6), ee-forces (6)
                        'state': tfds.features.Tensor(
                            shape=(20,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '1x gripper status, 6x joint torques, 6x end-effector force].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x end-effector position, '
                            '4x end-effector quaternion, 1x gripper open/close].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        from iamlab_cmu_pickup_insert.create_iamlab_data_helpers import load_demo_data, data_paths, SKIP_DATA
    
        def _parse_example(episode_path_instruction: Tuple[str, str]):
            episode_path, lang_instruction = Path(episode_path_instruction[0]), episode_path_instruction[1]

            episode = []
            steps = 0
            for step, step_data in load_demo_data(episode_path, lang_instruction, load_images=True):
                if step == 0:
                    language_embedding = self._embed([step_data['language_instruction']])[0].numpy()
                step_data['language_embedding'] = np.copy(language_embedding)
                episode.append(step_data)
                steps += 1

            # create output data sample
            relative_path_idx = episode_path.parts.index('object_centric')
            episode_str = '/'.join(episode_path.parts[relative_path_idx + 1:])
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_str
                }
            }
            return episode_str, sample

        debug = True
        episode_paths = []
        for multi_task_paths, lang_instruction in data_paths:
            for data_path in multi_task_paths:
                data_dir = Path(data_path)
                assert data_dir.exists(), f'{data_dir} does not exist'
                if debug:
                    print("Will convert data from ", data_dir)
                for demo_path in data_dir.iterdir():
                    if demo_path.name.startswith('demo') and str(demo_path) not in SKIP_DATA:
                        # episode_paths.append((str(demo_path), lang_instruction)) 
                        yield _parse_example((demo_path, lang_instruction))

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        beam = tfds.core.lazy_imports.apache_beam
        return (
                beam.Create(episode_paths)
                | beam.Map(_parse_example)
        )


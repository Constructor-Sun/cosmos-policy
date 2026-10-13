"""CPU regressions for failure identity, complete-trajectory P95 and partitions."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import collect_libero_plus_failures as collect
import select_libero_plus_success_p95 as select


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class ManifestToolsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline, self.repairs, self.demos = [self.root / name for name in ('baseline', 'repairs', 'demos')]
        self.group, self.base = 'sensor_noise', 'KITCHEN_test_task'
        self.protocol = dict(action_seed=42, environment_seed=10000, settle_steps=30,
                             max_steps=700, rng_mode='openwam', init_state_index=0, num_trials_per_task=1)
        base_path = self.root / 'base_tasks.json'
        save(base_path, {'tasks': [{'name': self.base, 'language': 'clean instruction'}]})
        self.tasks = [dict(task_id=i, source_id=i+1, task_name=f'{self.base}_noise_{i}',
                           group=self.group, category='Sensor Noise', difficulty_level=1) for i in range(4)]
        self.config = dict(base_tasks=str(base_path), fingerprint='same-policy-and-protocol', **self.protocol,
                           groups=[dict(key=self.group, category='Sensor Noise', tasks=self.tasks)])
        save(self.baseline / 'run_config.json', self.config)
        for task in self.tasks:
            directory = self.baseline / self.group / 'episodes' / f"task{task['task_id']:04d}"
            directory.mkdir(parents=True)
            for name in ('episode.hdf5', 'actions.npy', 'video.mp4'):
                (directory / name).touch()
            result = dict(task, **self.protocol, fingerprint=self.config['fingerprint'],
                          status='completed', success=task['task_id'] == 3,
                          instruction='actual perturbed instruction', videos=[str(directory / 'video.mp4')],
                          episode_hdf5=str(directory / 'episode.hdf5'), actions_path=str(directory / 'actions.npy'))
            save(directory / 'result.json', result)
        self.demos.mkdir()
        with h5py.File(self.demos / (self.base + '_demo.hdf5'), 'w') as data:
            for i, length in enumerate((10, 20)):
                data.create_dataset(f'data/demo_{i}/actions', data=np.zeros((length, 7)))
        for i, length in enumerate((19, 20, 20)):
            directory = self.repairs / self.group / f'task{i:04d}'
            directory.mkdir(parents=True)
            actions = np.zeros((length, 7), dtype=np.float32)
            np.save(directory / 'actions.npy', actions)
            (directory / 'video.mp4').touch()
            with h5py.File(directory / 'episode.hdf5', 'w') as data:
                data.attrs['success'] = i < 2
                data.create_dataset('actions', data=actions)
                data.create_dataset('proprio', data=np.zeros((length, 9)))
                data.create_dataset('primary_images_jpeg', data=np.zeros((length, 1), dtype=np.uint8))
                data.create_dataset('wrist_images_jpeg', data=np.zeros((length, 1), dtype=np.uint8))
            result = dict(task_id=i, case_id=f'task{i:04d}', task_name=self.tasks[i]['task_name'],
                          group=self.group, baseline_success=False, protocol={key:self.protocol[key] for key in collect.PROTOCOL_FIELDS[:5]},
                          status='completed', success=i < 2, action_steps=length, t_star=18,
                          video=str(directory / 'video.mp4'), actions=str(directory / 'actions.npy'), episode_hdf5=str(directory / 'episode.hdf5'))
            save(directory / 'result.json', result)
        self.failure_path = self.root / 'failed.json'
        self.refresh_failures()

    def baseline_result(self, i):
        return self.baseline / self.group / 'episodes' / f'task{i:04d}' / 'result.json'

    def repair_result(self, i):
        return self.repairs / self.group / f'task{i:04d}' / 'result.json'

    def refresh_failures(self):
        manifest, generation = collect.build_failure_manifest(self.baseline)
        save(self.failure_path, manifest)
        return manifest, generation

    def selection(self, **kwargs):
        return select.build_selection(self.failure_path, [self.repairs], self.demos, **kwargs)

    def test_failures_keep_exact_instruction_and_protocol(self):
        manifest, generation = self.refresh_failures()
        self.assertEqual([row['task_id'] for row in manifest['cases']], [0, 1, 2])
        self.assertEqual(manifest['protocol'], self.protocol)
        self.assertEqual(manifest['cases'][0]['instruction'], 'actual perturbed instruction')
        self.assertEqual(manifest['cases'][0]['clean_language'], 'clean instruction')
        self.assertEqual((generation['successes'], generation['failures']), (1, 3))

    def test_runtime_error_is_not_a_failure(self):
        path = self.baseline_result(2)
        row = collect.read_json(path)
        row.update(status='error', success=False)
        save(path, row)
        with self.assertRaisesRegex(ValueError, 'Baseline incomplete'):
            collect.build_failure_manifest(self.baseline)
        manifest, generation = collect.build_failure_manifest(self.baseline, allow_incomplete=True)
        self.assertEqual(manifest['num_cases'], 2)
        self.assertFalse(generation['complete'])

    def test_wrong_seed_rejected(self):
        path = self.baseline_result(0)
        row = collect.read_json(path)
        row['environment_seed'] = 99
        save(path, row)
        with self.assertRaisesRegex(ValueError, 'Protocol mismatch'):
            self.refresh_failures()

    def test_missing_source_artifact_rejected(self):
        self.baseline_result(0).with_name('episode.hdf5').unlink()
        with self.assertRaises(FileNotFoundError):
            self.refresh_failures()

    def test_duplicate_baseline_result_rejected(self):
        save(self.baseline / 'other_group/episodes/task0000/result.json', collect.read_json(self.baseline_result(0)))
        with self.assertRaisesRegex(ValueError, 'Unexpected/duplicate'):
            self.refresh_failures()

    def test_fractional_p95_uses_full_trajectory_and_partitions(self):
        selection = self.selection()
        self.assertEqual(selection['summary']['by_task'][self.base]['p95'], 19.5)
        self.assertEqual([row['task_id'] for row in selection['selected_manifest']['cases']], [0])
        self.assertEqual([row['task_id'] for row in selection['excluded_manifest']['cases']], [1])
        partition = selection['partitions'][self.group]
        self.assertEqual([row['task_id'] for row in partition['heldout']], [1, 2])
        self.assertEqual([row['task_id'] for row in partition['memory_failed']], [2])
        payloads = select.selection_payloads(selection, self.root / 'output')
        training = json.loads(payloads[self.root / 'output/by_category/sensor_noise/training_manifest.json'])
        self.assertEqual(training['evaluation_subset']['task_ids'], [0, 1, 2])
        self.assertEqual(training['num_action_steps'], 19)

    def test_duplicate_repairs_rejected(self):
        other = self.root / 'other_repair'
        save(other / self.group / 'task0000/result.json', collect.read_json(self.repair_result(0)))
        with self.assertRaisesRegex(ValueError, 'Unknown/duplicate repaired'):
            select.build_selection(self.failure_path, [self.repairs, other], self.demos)

    def test_unsaved_full_length_rejected(self):
        path = self.repair_result(0)
        row = collect.read_json(path)
        row['action_steps'] = 18
        save(path, row)
        with self.assertRaisesRegex(ValueError, 'shape/length mismatch'):
            self.selection()

    def test_saved_actions_mismatch_rejected(self):
        path = self.repair_result(0).with_name('actions.npy')
        np.save(path, np.ones((19, 7), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'actions.npy disagrees'):
            self.selection()

    def test_incomplete_repair_never_becomes_memory_failure(self):
        self.repair_result(2).unlink()
        with self.assertRaisesRegex(ValueError, 'Correction inputs incomplete'):
            self.selection()
        selection = self.selection(allow_incomplete=True)
        self.assertEqual(selection['generation']['memory_failures'], 0)
        self.assertFalse(selection['generation']['complete'])
        self.assertEqual(len(selection['partitions'][self.group]['heldout']), 2)

    def test_changed_baseline_config_rejected(self):
        config = dict(self.config, action_seed=7)
        save(self.baseline / 'run_config.json', config)
        with self.assertRaisesRegex(ValueError, 'run_config changed'):
            self.selection()

    def test_equivalent_output_keeps_hash_and_conflict_writes_nothing(self):
        existing = self.root / 'existing.json'
        existing.write_text('{"b": 2, "a": 1}')
        self.assertEqual(collect.json_payload(existing, {'a': 1, 'b': 2}), existing.read_bytes())
        new_path = self.root / 'not_created.json'
        with self.assertRaises(FileExistsError):
            collect.write_payloads({new_path: b'new', existing: b'different'})
        self.assertFalse(new_path.exists())


if __name__ == '__main__':
    unittest.main()

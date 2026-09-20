"""Storage and selection tests; no GPU or training process required."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import numpy.random


class FakeEval:
    def __init__(self, env, **kwargs):
        self.eval_env = env
        self.eval_freq = 1
        self.n_calls = 1
        self.n_eval_episodes = 10

    def _on_step(self):
        self._is_success_buffer, self.last_mean_reward = self.result
        return True


spec = importlib.util.spec_from_file_location(
    'checkpointing_under_test', Path(__file__).resolve().parents[1] / 'checkpointing.py'
)
module = importlib.util.module_from_spec(spec)
callbacks = types.ModuleType('stable_baselines3.common.callbacks')
callbacks.BaseCallback = object
callbacks.EvalCallback = FakeEval
with patch.dict(sys.modules, {'stable_baselines3.common.callbacks': callbacks}):
    spec.loader.exec_module(module)


class CheckpointTests(unittest.TestCase):
    def model(self):
        normalizer = types.SimpleNamespace(
            norm_obs=True, norm_reward=False,
            save=lambda path: Path(path).write_bytes(b'normalizer'),
        )
        return types.SimpleNamespace(
            num_timesteps=100, n_envs=8, gamma=.999, gradient_steps=4,
            observation_space=types.SimpleNamespace(shape=(17,)),
            action_space=types.SimpleNamespace(shape=(9,)),
            get_vec_normalize_env=lambda: normalizer,
            save=lambda path: Path(path).write_bytes(b'model'),
            save_replay_buffer=lambda path: Path(path).write_bytes(b'buffer'),
        )

    def test_complete_bundle_and_failed_write_preserves_pointer(self):
        with tempfile.TemporaryDirectory() as root:
            store = module.CheckpointStore(root, 'v2', 'v2_phase_1a', 42)
            model = self.model()
            bundle = store.save(model, 'periodic', replay=True)
            for name in ('model.zip', 'vec_normalize.pkl', 'replay_buffer.pkl', 'metadata.json'):
                self.assertTrue((bundle / name).is_file())
            pointer = (store.path / 'periodic.json').read_text()
            self.assertEqual(json.loads(pointer)['directory'], bundle.name)
            with patch.object(model, 'save_replay_buffer', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    store.save(model, 'periodic', replay=True)
            self.assertEqual((store.path / 'periodic.json').read_text(), pointer)
            self.assertTrue(bundle.is_dir())

    def test_success_precedes_reward_and_ties_use_reward(self):
        saves, seeds = [], []
        store = types.SimpleNamespace(save=lambda *args, **kwargs: saves.append(kwargs))
        callback = module.LandingEvalCallback(
            types.SimpleNamespace(seed=seeds.append), store, 100042
        )
        callback.model = self.model()
        for successes, reward in [(9, 10), (0, 1000), (9, 9), (9, 11), (10, -100)]:
            callback.result = ([True]*successes + [False]*(10-successes), reward)
            callback._on_step()
        self.assertEqual([entry['metrics']['success_rate'] for entry in saves], [.9, .9, 1.0])
        self.assertEqual([entry['metrics']['mean_reward'] for entry in saves], [10, 11, -100])
        self.assertEqual(seeds, [100042]*5)

    def test_sessions_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as root:
            a = module.CheckpointStore(root, 'v2', 'v2_phase_1a', 42)
            b = module.CheckpointStore(root, 'v2', 'v2_phase_1a', 42)
            self.assertNotEqual(a.path, b.path)

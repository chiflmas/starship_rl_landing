"""Resume action-selection tests, plus an optional real SB3 smoke test."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SB3_AVAILABLE = importlib.util.find_spec('stable_baselines3') is not None


class RecordingSAC:
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        self.sample_call = (learning_starts, action_noise, n_envs)
        return self.sample_result


spec = importlib.util.spec_from_file_location(
    'policy_warmup_under_test',
    Path(__file__).resolve().parents[1] / 'policy_warmup_sac.py',
)
module = importlib.util.module_from_spec(spec)
fake_sb3 = types.ModuleType('stable_baselines3')
fake_sb3.SAC = RecordingSAC
with patch.dict(sys.modules, {'stable_baselines3': fake_sb3}):
    spec.loader.exec_module(module)


class PolicyWarmupTests(unittest.TestCase):
    def test_random_sampling_threshold_is_disabled_without_changing_update_gate(self):
        for timestep in (0, 1_000_000, 1_050_008):
            with self.subTest(timestep=timestep):
                model = module.PolicyWarmupSAC()
                model.num_timesteps = timestep
                model.learning_starts = 1_050_000
                model.gradient_steps = 4
                model.sample_result = (object(), object())
                result = model._sample_action(model.learning_starts, n_envs=8)
                self.assertEqual(model.sample_call, (0, None, 8))
                self.assertIs(result, model.sample_result)
                self.assertEqual(model.learning_starts, 1_050_000)
                self.assertEqual(model.num_timesteps, timestep)
                self.assertEqual(model.gradient_steps, 4)

    def test_noise_and_vectorization_are_delegated_to_sb3(self):
        model = module.PolicyWarmupSAC()
        model.sample_result = (object(), object())
        noise = object()
        model._sample_action(50_000, action_noise=noise, n_envs=6)
        self.assertEqual(model.sample_call, (0, noise, 6))

    def test_default_single_environment_is_preserved(self):
        model = module.PolicyWarmupSAC()
        model.sample_result = (object(), object())
        model._sample_action(50_000)
        self.assertEqual(model.sample_call, (0, None, 1))

    @unittest.skipUnless(SB3_AVAILABLE, 'Requires installed Gymnasium/PyTorch/SB3')
    def test_real_resume_collects_with_policy_before_updates_and_can_be_reloaded(self):
        import gymnasium as gym
        import numpy as np
        from stable_baselines3 import SAC
        from policy_warmup_sac import PolicyWarmupSAC

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)
            action_space = gym.spaces.Box(
                np.asarray([0.0]*3 + [-1.0]*6, dtype=np.float32),
                np.ones(9, dtype=np.float32),
            )

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(17, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(17, dtype=np.float32), -float(np.square(action).sum()), False, False, {}

        with tempfile.TemporaryDirectory() as folder:
            source = SAC(
                'MlpPolicy', TinyEnv(), policy_kwargs=dict(net_arch=[8]),
                buffer_size=64, batch_size=2, gradient_steps=1,
                learning_starts=0, device='cpu', seed=42,
            )
            source.num_timesteps = 100
            source.save(str(Path(folder) / 'source'))
            resumed = PolicyWarmupSAC.load(str(Path(folder) / 'source'), env=TinyEnv(), device='cpu')
            resumed.learning_starts = resumed.num_timesteps + 6
            updates_before = resumed._n_updates
            with patch.object(resumed.action_space, 'sample', side_effect=AssertionError('Uniform random action')):
                with patch.object(resumed, 'predict', wraps=resumed.predict) as predict:
                    resumed.learn(total_timesteps=6, reset_num_timesteps=False)
                    self.assertEqual(predict.call_count, 6)
                    self.assertTrue(all(
                        call.kwargs['deterministic'] is False for call in predict.call_args_list
                    ))
                self.assertEqual(resumed.replay_buffer.size(), 6)
                self.assertEqual(resumed._n_updates, updates_before)
                resumed.learn(total_timesteps=2, reset_num_timesteps=False)
            self.assertGreater(resumed._n_updates, updates_before)
            resumed.save(str(Path(folder) / 'resumed'))
            standard = SAC.load(str(Path(folder) / 'resumed'), env=TinyEnv(), device='cpu')
            self.assertEqual(standard.observation_space.shape, (17,))
            self.assertEqual(standard.action_space.shape, (9,))
            source.get_env().close()
            resumed.get_env().close()
            standard.get_env().close()


if __name__ == '__main__':
    unittest.main()

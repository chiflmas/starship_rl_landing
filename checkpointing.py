"""Matched, immutable SAC snapshots and success-first evaluation."""
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class CheckpointStore:
    def __init__(self, root, version, phase, seed, source=None):
        run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
        self.path = Path(root) / run
        self.path.mkdir(parents=True, exist_ok=False)
        self.context = dict(version=version, phase=phase, seed=seed, source=source)
        print(f'[Checkpoints] New session: {self.path}')

    def save(self, model, kind, replay=False, metrics=None):
        """Publish the directory only after every file has been written.

        An interrupted write leaves a .pending directory, never a valid bundle.
        The small manifest points to a complete snapshot and is replaced last.
        """
        name = f'{kind}_{model.num_timesteps:012d}_{uuid4().hex[:8]}'
        pending = self.path / ('.pending_' + name)
        pending.mkdir()
        normalizer = model.get_vec_normalize_env()
        if normalizer is None:
            raise ValueError('A matched VecNormalize is required for checkpointing')
        model.save(str(pending / 'model.zip'))
        normalizer.save(str(pending / 'vec_normalize.pkl'))
        if replay:
            model.save_replay_buffer(str(pending / 'replay_buffer.pkl'))
        metadata = dict(
            **self.context, kind=kind, timesteps=int(model.num_timesteps),
            n_envs=int(model.n_envs), observation_shape=list(model.observation_space.shape),
            action_shape=list(model.action_space.shape), gamma=float(model.gamma),
            gradient_steps=int(model.gradient_steps), norm_obs=bool(normalizer.norm_obs),
            norm_reward=bool(normalizer.norm_reward), replay_buffer=replay,
            metrics=metrics, created_utc=datetime.now(timezone.utc).isoformat(),
        )
        (pending / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        pending.rename(self.path / name)
        pointer = self.path / f'.{kind}.json.tmp'
        pointer.write_text(json.dumps(dict(directory=name, **metadata), indent=2), encoding='utf-8')
        os.replace(pointer, self.path / f'{kind}.json')
        print(f'[Checkpoints] {kind}: {self.path / name}')
        return self.path / name


class LandingEvalCallback(EvalCallback):
    """Keep SB3 eval logs, but select snapshots by (success rate, reward)."""
    def __init__(self, eval_env, store, eval_seed, **kwargs):
        super().__init__(eval_env, best_model_save_path=None, **kwargs)
        self.store = store
        self.eval_seed = eval_seed
        self.best_score = (-math.inf, -math.inf)
        # Disable the parent's reward-only best-model branch.
        self.best_mean_reward = math.inf

    def _on_step(self):
        due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if not due:
            return True
        # Reuse the same scenario sequence, even when videos share eval_env.
        py_rng, np_rng = random.getstate(), np.random.get_state()
        self.eval_env.seed(self.eval_seed)
        try:
            keep_training = super()._on_step()
        finally:
            random.setstate(py_rng)
            np.random.set_state(np_rng)
        if len(self._is_success_buffer) != self.n_eval_episodes:
            raise ValueError('Every evaluation episode must report is_success')
        success_rate = float(np.mean(self._is_success_buffer))
        score = (success_rate, float(self.last_mean_reward))
        if all(math.isfinite(value) for value in score) and score > self.best_score:
            self.store.save(self.model, 'best', metrics=dict(
                success_rate=score[0], mean_reward=score[1],
                episodes=self.n_eval_episodes, eval_seed=self.eval_seed,
                selection='success_rate_then_mean_reward',
            ))
            self.best_score = score
        return keep_training


class PeriodicCheckpointCallback(BaseCallback):
    def __init__(self, store, interval=100_000):
        super().__init__()
        self.store, self.interval = store, interval

    def _on_training_start(self):
        self.last_saved = self.model.num_timesteps

    def _on_rollout_end(self):
        # Unlike _on_step, this runs after the transition enters replay memory.
        if self.model.num_timesteps - self.last_saved >= self.interval:
            self.store.save(self.model, 'periodic', replay=True)
            self.last_saved = self.model.num_timesteps

    def _on_step(self):
        return True

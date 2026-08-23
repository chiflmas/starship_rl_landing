"""Rewards for V1.1's 10-observation landing task.

Unlike V1, this reward does not calculate a phase-dependent attitude target or
a model-based braking margin. It uses only quantities represented by V1.1's
raw observation: position, velocity, attitude, angular rate, actuator state,
and elapsed time. The curriculum and terminal success criteria remain shared
with V1.
"""

import math

import numpy as np


class V11SingleEngineRewardSystem:
    """Progress-based rewards without engineered guidance variables."""

    def __init__(self, task='landing', world_bounds=None):
        self.task = task
        self.world_bounds = world_bounds or {}
        self.rocket_height = 50.0
        self.previous_metrics = None

    def reset_episode_memory(self):
        self.previous_metrics = None

    @staticmethod
    def _clipped_delta(previous, current, limit):
        """Return positive progress when an error-like raw metric decreases."""
        return float(np.clip(previous - current, -limit, limit))

    def _metrics(self, state):
        theta = float(state['theta'])
        downward_speed = max(0.0, -float(state['vy']))
        return {
            'altitude': max(
                0.0,
                float(state['y']) - 0.5 * self.rocket_height,
            ),
            'abs_x': abs(float(state['x'])),
            'abs_vx': abs(float(state['vx'])),
            # Computed directly from the observed cos(theta). It measures only
            # uprightness and contains no altitude-dependent target attitude.
            'upright_cost': 1.0 - math.cos(theta),
            'downward_speed': downward_speed,
        }

    def calculate_reward(self, state, action=None, engines_thrust=None,
                         step_id=None, max_steps=None, **kwargs):
        if self.task != 'landing':
            return 0.0, {}

        metrics = self._metrics(state)

        # Start rewarding upright control when the aerodynamic belly-flop hold
        # begins to disappear (450 -> 350 m in rocket.py), and reach full
        # weight by 250 m.  This aligns the learning signal with the physical
        # release without prescribing an intermediate target angle.
        upright_proximity = float(np.clip(
            (450.0 - metrics['altitude']) / 200.0,
            0.0,
            1.0,
        ))
        touchdown_proximity = float(np.clip(
            (100.0 - metrics['altitude']) / 75.0,
            0.0,
            1.0,
        ))

        # The safe-speed thresholds come from the terminal contact rules. They
        # do not estimate stopping distance or available braking authority.
        vertical_speed_excess = max(0.0, metrics['downward_speed'] - 5.0)

        if self.previous_metrics is None:
            pad_progress = 0.0
            lateral_velocity_progress = 0.0
            upright_progress = 0.0
            vertical_speed_progress = 0.0
        else:
            pad_progress = 0.080 * self._clipped_delta(
                self.previous_metrics['abs_x'], metrics['abs_x'], 3.0
            )
            lateral_velocity_progress = 0.050 * self._clipped_delta(
                self.previous_metrics['abs_vx'], metrics['abs_vx'], 5.0
            )
            upright_progress = 0.75 * upright_proximity * self._clipped_delta(
                self.previous_metrics['upright_cost'],
                metrics['upright_cost'],
                0.20,
            )
            previous_speed_excess = max(
                0.0,
                self.previous_metrics['downward_speed'] - 5.0,
            )
            vertical_speed_progress = (
                0.040
                * touchdown_proximity
                * self._clipped_delta(
                    previous_speed_excess,
                    vertical_speed_excess,
                    5.0,
                )
            )

        self.previous_metrics = metrics

        time_cost = -0.020
        ascent_penalty = -0.75 * float(
            np.clip(float(state['vy']) / 10.0, 0.0, 1.0)
        )
        angular_rate_penalty = -0.10 * abs(float(state['vtheta']))

        # Preserve a continuous distinction between a moderate tilt and a
        # nose-first attitude.  The previous 20-degree clipping assigned the
        # same penalty to every attitude from 20 to 180 degrees.
        upright_penalty = (
            -1.5
            * upright_proximity
            * metrics['upright_cost']
        )
        vertical_speed_penalty = -1.0 * touchdown_proximity * float(
            np.clip(vertical_speed_excess / 20.0, 0.0, 1.0)
        )
        lateral_speed_penalty = -0.50 * touchdown_proximity * float(
            np.clip(metrics['abs_vx'] / 10.0, 0.0, 1.0)
        )
        outside_pad_penalty = -0.75 * touchdown_proximity * float(
            np.clip((metrics['abs_x'] - 15.0) / 15.0, 0.0, 1.0)
        )

        components = {
            'pad_progress': pad_progress,
            'lateral_velocity_progress': lateral_velocity_progress,
            'upright_progress': upright_progress,
            'vertical_speed_progress': vertical_speed_progress,
            'upright': upright_penalty,
            'vertical_speed': vertical_speed_penalty,
            'lateral_speed': lateral_speed_penalty,
            'outside_pad': outside_pad_penalty,
            'ascent': ascent_penalty,
            'angular_rate': angular_rate_penalty,
            'time_cost': time_cost,
        }
        return sum(components.values()), components

    def get_terminal_rewards(self):
        return {
            'successful_landing': 300.0,
            'crash': -250.0,
            'timeout': -200.0,
        }

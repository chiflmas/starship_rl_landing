"""Rewards for V1: progress-based terminal landing with one virtual actuator."""

import math

import numpy as np


class V1SingleEngineRewardSystem:
    """Reward shaping for the V1 landing task.

    V1 deliberately does not prescribe a vertical-speed profile or a burn
    altitude. The agent is free to discover its own braking strategy. Dense
    rewards are based on *progress* instead of remaining in a desirable state,
    so a centred upright hover cannot accumulate reward indefinitely.
    """

    def __init__(self, task='landing', world_bounds=None):
        self.task = task
        self.world_bounds = world_bounds or {}
        self.rocket_height = 50.0
        self.max_total_thrust = 6e6

        # This is an attitude reference, not a prescribed velocity trajectory.
        # The environment retains belly-flop aerodynamics above this window and
        # releases them before terminal landing.
        self.flip_start_altitude = 450.0
        self.landing_burn_start_altitude = 250.0

        # Safety-envelope parameters. These do not prescribe when to burn: they
        # only estimate whether the current downward speed can still be reduced
        # to a safe touchdown with the actuator authority available in V1.
        self.touchdown_speed = 5.5
        # Conservative, learnable estimate of usable braking authority.
        self.effective_braking_deceleration = 20.0
        self.required_braking_margin = 40.0

        self.previous_metrics = None

    def reset_episode_memory(self):
        self.previous_metrics = None

    def _target_abs_theta(self, altitude):
        """Altitude-dependent attitude objective for belly flop and flip."""
        if altitude >= self.flip_start_altitude:
            return math.radians(80.0)
        if altitude > self.landing_burn_start_altitude:
            blend = (
                (altitude - self.landing_burn_start_altitude)
                / (self.flip_start_altitude - self.landing_burn_start_altitude)
            )
            return math.radians(80.0) * blend
        return 0.0

    def _metrics(self, state):
        altitude = max(0.0, float(state['y']) - 0.5 * self.rocket_height)
        target_abs_theta = self._target_abs_theta(altitude)
        downward_speed = max(0.0, -float(state['vy']))
        stopping_distance = max(
            0.0,
            (downward_speed ** 2 - self.touchdown_speed ** 2)
            / (2.0 * self.effective_braking_deceleration),
        )
        braking_margin = altitude - stopping_distance
        # Once the vehicle is physically on the pad, touchdown logic rather
        # than the in-flight braking envelope decides success or failure.
        braking_deficit = (
            0.0
            if altitude <= 2.0
            else max(0.0, self.required_braking_margin - braking_margin)
        )
        return {
            'altitude': altitude,
            'abs_x': abs(float(state['x'])),
            'attitude_error': abs(abs(float(state['theta'])) - target_abs_theta),
            'abs_vx': abs(float(state['vx'])),
            'braking_deficit': braking_deficit,
        }

    @staticmethod
    def _clipped_delta(previous, current, limit):
        """Positive when an error-like quantity is reduced."""
        return float(np.clip(previous - current, -limit, limit))

    def calculate_reward(self, state, action=None, engines_thrust=None,
                         step_id=None, max_steps=None, **kwargs):
        if self.task != 'landing':
            return 0.0, {}

        metrics = self._metrics(state)

        # All positive shaping terms are deltas. Losing altitude by gravity is
        # deliberately not rewarded: time pressure already encourages progress.
        if self.previous_metrics is None:
            pad_progress = 0.0
            attitude_progress = 0.0
            lateral_velocity_progress = 0.0
        else:
            pad_progress = 0.080 * self._clipped_delta(
                self.previous_metrics['abs_x'], metrics['abs_x'], 3.0
            )
            attitude_progress = 1.50 * self._clipped_delta(
                self.previous_metrics['attitude_error'],
                metrics['attitude_error'],
                math.radians(5.0),
            )
            lateral_velocity_progress = 0.050 * self._clipped_delta(
                self.previous_metrics['abs_vx'], metrics['abs_vx'], 5.0
            )
            braking_viability_progress = 0.050 * self._clipped_delta(
                self.previous_metrics['braking_deficit'],
                metrics['braking_deficit'],
                20.0,
            )
        if self.previous_metrics is None:
            braking_viability_progress = 0.0
        self.previous_metrics = metrics

        # The agent is not told how fast to descend. It may choose a late or
        # conservative burn. V1 has no fuel model, so thrust itself is not
        # penalised; time pressure prevents indefinite hovering.
        time_cost = -0.020
        ascent_penalty = -0.75 * float(
            np.clip(float(state['vy']) / 10.0, 0.0, 1.0)
        )
        angular_rate_penalty = -0.10 * abs(float(state['vtheta']))
        braking_viability_penalty = -1.0 * float(
            np.clip(metrics['braking_deficit'] / 60.0, 0.0, 1.0)
        )

        # Progress alone gives no signal when a large attitude error remains
        # constant. This non-positive term activates smoothly through the flip
        # window and cannot be farmed by remaining upright.
        vertical_phase = float(np.clip(
            (self.flip_start_altitude - metrics['altitude'])
            / (self.flip_start_altitude - self.landing_burn_start_altitude),
            0.0,
            1.0,
        ))
        attitude_error_penalty = -0.50 * vertical_phase * float(
            np.clip(metrics['attitude_error'] / math.radians(30.0), 0.0, 1.0)
        )

        # Terminal-control shaping. Near the ground, attitude and angular-rate
        # control remain active everywhere: leaving the pad must not switch off
        # the penalties and become an easier alternative to stabilising.
        ground_proximity = float(np.clip(
            (100.0 - metrics['altitude']) / 75.0,
            0.0,
            1.0,
        ))
        touchdown_proximity = ground_proximity

        touchdown_attitude_penalty = -1.0 * touchdown_proximity * float(
            np.clip(
                metrics['attitude_error'] / math.radians(10.0),
                0.0,
                1.0,
            )
        )
        touchdown_spin_penalty = -0.75 * touchdown_proximity * float(
            np.clip(
                abs(float(state['vtheta'])) / math.radians(10.0),
                0.0,
                1.0,
            )
        )

        # The valid touchdown corridor is hard at +/-15 m. Outside it, apply a
        # smooth non-positive cost that saturates at 30 m. Unlike progress-only
        # shaping, remaining far from the pad keeps carrying a cost.
        outside_pad_penalty = -0.75 * ground_proximity * float(
            np.clip((metrics['abs_x'] - 15.0) / 15.0, 0.0, 1.0)
        )

        components = {
            'pad_progress': pad_progress,
            'attitude_progress': attitude_progress,
            'attitude_error': attitude_error_penalty,
            'touchdown_attitude': touchdown_attitude_penalty,
            'touchdown_spin': touchdown_spin_penalty,
            'outside_pad': outside_pad_penalty,
            'lateral_velocity_progress': lateral_velocity_progress,
            'braking_viability_progress': braking_viability_progress,
            'braking_viability': braking_viability_penalty,
            'ascent': ascent_penalty,
            'angular_rate': angular_rate_penalty,
            'time_cost': time_cost,
        }
        return sum(components.values()), components

    def get_terminal_rewards(self):
        # Terminal rewards dominate progress shaping: crashing or waiting for a
        # timeout cannot beat a valid touchdown.
        return {
            'successful_landing': 300.0,
            'crash': -250.0,
            'timeout': -200.0,
        }

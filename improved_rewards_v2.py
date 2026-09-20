"""Progress-based rewards for the isolated V2 three-engine environment.

V2 deliberately keeps the raw-state philosophy introduced by V1.1. The
reward does not prescribe an engine count, exact ignition/shutdown altitude,
throttle value, target descent profile, or gimbal allocation. This copy is
independent of V1.1 so either version can evolve without silently changing the
other.
"""

import math

import numpy as np


class V2ThreeEngineRewardSystem:
    """Symmetric landing rewards without engineered guidance variables."""

    def __init__(self, task='landing', world_bounds=None):
        self.task = task
        self.world_bounds = world_bounds or {}
        self.rocket_height = 50.0
        self.previous_metrics = None
        self.ground_progress_scale = 1.00
        self.terminal_ground_progress_scale = 3.00
        self.terminal_progress_transition_start = 25.0
        self.terminal_progress_full_below = 10.0
        self.previous_engine_capacity = None

    def reset_episode_memory(self):
        self.previous_metrics = None
        self.previous_engine_capacity = None

    @staticmethod
    def _clipped_delta(previous, current, limit):
        return float(np.clip(previous - current, -limit, limit))

    def _altitude_cost(self, altitude):
        """Integrate the descent weights into a continuous height cost.

        Its slope is 3 below 10 m, fades to 1 between 10 and 25 m, and
        remains 1 above 25 m with the default settings. Taking the signed
        difference of this cost makes a closed altitude cycle sum to zero
        before discounting, regardless of step size or threshold crossings.
        This is transition shaping, not a claim of discounted policy invariance.
        """
        altitude = max(0.0, float(altitude))
        lower = self.terminal_progress_full_below
        upper = self.terminal_progress_transition_start
        width = upper - lower
        transition_height = float(np.clip(altitude - lower, 0.0, width))
        return (
            self.terminal_ground_progress_scale * min(altitude, lower)
            + self.terminal_ground_progress_scale * transition_height
            - 0.5 * (
                self.terminal_ground_progress_scale - self.ground_progress_scale
            ) * transition_height ** 2 / width
            + self.ground_progress_scale * max(0.0, altitude - upper)
        )

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
            # Linear angular error keeps the difference between 0 and 5
            # degrees visible.  1-cos(theta) was effectively flat precisely
            # where the policy needs its final attitude correction.
            'upright_cost': abs(theta) / math.radians(10.0),
            'downward_speed': downward_speed,
        }

    def calculate_reward(self, state, action=None, engines_thrust=None,
                         step_id=None, max_steps=None, **kwargs):
        if self.task != 'landing':
            return 0.0, {}

        metrics = self._metrics(state)
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
        approach_proximity = float(np.clip(
            (300.0 - metrics['altitude']) / 250.0,
            0.0,
            1.0,
        ))
        # Ablation: disable vertical_speed_progress to evaluate early braking.
        # Keep the original calculation commented for easy restoration.
        # braking_proximity = float(np.clip(
        #     (150.0 - metrics['altitude']) / 125.0,
        #     0.0,
        #     1.0,
        # ))
        vertical_speed_excess = max(0.0, metrics['downward_speed'] - 5.0)
        if self.previous_metrics is None:
            pad_progress = 0.0
            lateral_velocity_progress = 0.0
            upright_progress = 0.0
            vertical_speed_progress = 0.0
            ground_progress = 0.0
            ascent = 0.0
        else:
            pad_progress = 1.50 * self._clipped_delta(
                self.previous_metrics['abs_x'], metrics['abs_x'], 3.0
            )
            lateral_velocity_progress = 0.20 * self._clipped_delta(
                self.previous_metrics['abs_vx'], metrics['abs_vx'], 5.0
            )
            upright_progress = 0.75 * upright_proximity * self._clipped_delta(
                self.previous_metrics['upright_cost'],
                metrics['upright_cost'],
                0.50,
            )
            # previous_speed_excess = max(
            #     0.0,
            #     self.previous_metrics['downward_speed'] - 5.0,
            # )
            # vertical_speed_progress = (
            #     0.15
            #     * braking_proximity
            #     * self._clipped_delta(
            #         previous_speed_excess,
            #         vertical_speed_excess,
            #         5.0,
            #     )
            # )
            vertical_speed_progress = 0.0
            # Use only this transition, not an unobserved all-episode minimum.
            # Ascent gives back the same progress that descent earns. Integrate
            # the weight instead of multiplying by the destination weight or
            # clipping each delta: both would make cycles/path length matter.
            ground_progress = (
                self._altitude_cost(self.previous_metrics['altitude'])
                - self._altitude_cost(metrics['altitude'])
            )
            # Charge ascent by metres climbed rather than by time spent with a
            # positive velocity. Long recoverable episodes should not become
            # intrinsically worse than an immediate crash.
            altitude_gain = float(np.clip(
                metrics['altitude'] - self.previous_metrics['altitude'],
                0.0,
                2.0,
            ))
            ascent = -1.50 * altitude_gain

        self.previous_metrics = metrics

        # Engine-management penalties use only raw observable state. There is
        # deliberately no stopping-distance model, target burn altitude, or
        # prescribed engine sequence hidden in the reward.
        deadstick_shutdown = 0.0
        shutdown_option_cost = 0.0
        unpowered_descent_risk = 0.0
        safe_touchdown_contact = 0.0
        touchdown_quality = 0.0
        # Disabled for this reward ablation; keep the logging component stable.
        terminal_time_pressure = 0.0

        # Bridge the sparse 20-step stability objective with a one-shot bonus
        # for entering the same safe touchdown envelope. `just_touched` is true
        # only on the impact transition, so this reward cannot be farmed.
        if bool(state.get('just_touched', False)):
            impact_speed = abs(float(state.get(
                'impact_velocity',
                state.get('vy', float('inf')),
            )))
            impact_angle = abs(float(state.get(
                'impact_theta',
                state['theta'],
            )))
            # A near miss must be measurably better than a nose-first or
            # off-pad impact.  These four terms are the same raw quantities
            # used by the success test, not a burn schedule or control hint.
            position_quality = float(np.clip(
                1.0 - metrics['abs_x'] / 45.0, 0.0, 1.0
            ))
            speed_quality = float(np.clip(
                1.0 - impact_speed / 15.0, 0.0, 1.0
            ))
            attitude_quality = float(np.clip(
                1.0 - impact_angle / math.radians(20.0), 0.0, 1.0
            ))
            spin_quality = float(np.clip(
                1.0 - abs(float(state['vtheta'])) / math.radians(10.0),
                0.0,
                1.0,
            ))
            touchdown_quality = 1500.0 * (
                position_quality
                * speed_quality
                * attitude_quality
                * spin_quality
            )
            safe_touchdown = (
                impact_speed < 5.0
                and metrics['abs_x'] < 15.0
                and impact_angle < math.radians(5.0)
                and abs(float(state['vtheta'])) < math.radians(3.0)
            )
            if safe_touchdown:
                safe_touchdown_contact = 1000.0

        # Keep a smooth target-distance cost throughout the flight.
        # Contact ends this cost while stability is being verified.
        target_distance_cost = 0.0
        if not bool(state.get('touchdown_contact', False)):
            # State cost tied only to the task objective. It does not prescribe
            # an ignition altitude, velocity curve, engine count, or sequence;
            # it makes indefinite flight far from the pad less attractive.
            normalized_target_distance = math.hypot(
                metrics['abs_x'] / 100.0,
                metrics['altitude'] / 100.0,
            )
            target_distance_cost = -0.40 * float(np.clip(
                normalized_target_distance,
                0.0,
                5.0,
            ))

        # The normal belly-flop spawn remains unpenalised through 90 degrees.
        # Beyond that, a smooth barrier discourages rotation towards a
        # nose-first attitude before the existing 120-degree crash terminal.
        over_rotation = -3.0 * float(np.clip(
            (abs(float(state['theta'])) - math.radians(90.0))
            / math.radians(30.0),
            0.0,
            1.0,
        ))
        curriculum_mask = list(state.get('curriculum_engine_mask', []))
        engine_operational = list(state.get('engine_operational', []))
        engine_locked_out = list(state.get('engine_locked_out', []))
        engines_on = list(state.get('engines_on', []))
        engine_state_lengths_match = (
            len(curriculum_mask)
            == len(engine_operational)
            == len(engine_locked_out)
            == len(engines_on)
            and len(curriculum_mask) > 0
        )
        if engine_state_lengths_match:
            # Capacity matches the compact observation's semantics: both an
            # ON engine and a healthy, unignited OFF engine remain usable.
            # Masked, failed and permanently locked engines do not.
            remaining_engines = sum(
                bool(enabled) and bool(operational) and not bool(locked)
                for enabled, operational, locked in zip(
                    curriculum_mask, engine_operational, engine_locked_out
                )
            )
            touchdown_contact = bool(state.get('touchdown_contact', False))
            airborne = not touchdown_contact and metrics['altitude'] > 1.0

            # Staging one or two engines is expected in Phase 1C, where three
            # engines at minimum throttle have T/W > 1. Keep those decisions
            # almost neutral and reserve the strong penalty for discarding the
            # final enabled engine and entering an unrecoverable deadstick.
            if (
                self.previous_engine_capacity is not None
                and not touchdown_contact
            ):
                lost_capacity = max(
                    0, self.previous_engine_capacity - remaining_engines
                )
                if lost_capacity > 0:
                    shutdown_option_cost = (
                        -1.0 * lost_capacity
                        if remaining_engines > 0
                        else -25.0
                    )

                    # Charge the transition that discards the final usable
                    # engine, even if still ascending. No once-per-episode
                    # hidden flag, prescribed burn altitude or thrust demand.
                    if remaining_engines == 0 and airborne:
                        deadstick_shutdown = -(
                            200.0 + 10.0 * min(metrics['downward_speed'], 20.0)
                        )
            self.previous_engine_capacity = remaining_engines

            # Coasting with a reserve is not irreversible deadstick. Remove
            # the old <300 m ignition incentive; only an unpowered descent
            # with NO usable engines incurs this continuing state cost.
            if airborne and remaining_engines == 0 and metrics['downward_speed'] > 0.5:
                unpowered_descent_risk = -float(np.clip(
                    metrics['downward_speed'] / 30.0, 0.0, 1.0
                ))
        else:
            # Do not compare capacities across missing/incomplete telemetry.
            self.previous_engine_capacity = None

        components = {
            'pad_progress': pad_progress,
            'lateral_velocity_progress': lateral_velocity_progress,
            'upright_progress': upright_progress,
            'vertical_speed_progress': vertical_speed_progress,
            'ground_progress': ground_progress,
            'upright': -0.60 * upright_proximity * float(np.clip(
                metrics['upright_cost'], 0.0, 5.0
            )),
            'vertical_speed': -0.35 * touchdown_proximity * float(
                np.clip(vertical_speed_excess / 20.0, 0.0, 1.0)
            ),
            'lateral_speed': -0.50 * approach_proximity * float(
                np.clip(metrics['abs_vx'] / 5.0, 0.0, 2.0)
            ),
            'pad_offset': -0.40 * approach_proximity * float(
                np.clip(metrics['abs_x'] / 15.0, 0.0, 3.0)
            ),
            'outside_pad': -0.75 * touchdown_proximity * float(
                np.clip((metrics['abs_x'] - 15.0) / 15.0, 0.0, 1.0)
            ),
            'ascent': ascent,
            'angular_rate': -0.4 * abs(float(state['vtheta'])),
            'over_rotation': over_rotation,
            'shutdown_option_cost': shutdown_option_cost,
            'unpowered_descent_risk': unpowered_descent_risk,
            'deadstick_shutdown': deadstick_shutdown,
            'safe_touchdown_contact': safe_touchdown_contact,
            'touchdown_quality': touchdown_quality,
            'terminal_time_pressure': terminal_time_pressure,
            'target_distance_cost': target_distance_cost,
            'time_cost': -0.20,
        }
        return sum(components.values()), components

    def get_terminal_rewards(self):
        return {
            'successful_landing': 4000.0,
            'crash': -2500.0,
            'timeout': -2000.0,
        }

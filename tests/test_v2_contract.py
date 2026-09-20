import math
import unittest

import numpy as np

from improved_rewards_v2 import V2ThreeEngineRewardSystem
from rocket import Rocket


def make_state(**overrides):
    state = {
        "x": 0.0,
        "y": 425.0,
        "vx": 0.0,
        "vy": -35.0,
        "theta": 0.0,
        "vtheta": 0.0,
        "phi": 0.0,
        "f": 0.0,
        "t": 0.0,
        "a_": 0.0,
        "engine_gimbals": [0.0, 0.0, 0.0],
    }
    state.update(overrides)
    return state


def make_v2(state=None, phase="phase_1"):
    rocket = Rocket(
        max_steps=750,
        task="landing",
        rocket_type="starship",
        engine_mode="three_v2",
        observation_mode="v2_raw",
        curriculum_phase=phase,
    )
    rocket.reward_system = V2ThreeEngineRewardSystem()
    rocket.reset(state or make_state())
    return rocket


def v2_action(throttles, gimbals=(0.0, 0.0, 0.0), switches=(1.0, 1.0, 1.0)):
    return np.asarray([*throttles, *gimbals, *switches], dtype=np.float32)


class V2ContractTests(unittest.TestCase):
    def test_ground_progress_rewards_descent_but_not_hover(self):
        reward_system = V2ThreeEngineRewardSystem()
        state = make_state(y=125.0, engines_on=[False, False, False])

        _, initial = reward_system.calculate_reward(state)
        self.assertEqual(initial["ground_progress"], 0.0)

        state["y"] = 124.0
        _, descent = reward_system.calculate_reward(state)
        self.assertAlmostEqual(descent["ground_progress"], 1.00)

        _, hover = reward_system.calculate_reward(state)
        self.assertEqual(hover["ground_progress"], 0.0)

        state["y"] = 125.0
        _, ascent = reward_system.calculate_reward(state)
        self.assertAlmostEqual(ascent["ground_progress"], -1.00)
        self.assertAlmostEqual(ascent["ascent"], -1.50)

        # Returning earns the progress repaid during ascent; the closed
        # excursion adds no progress and still pays the ascent/time costs.
        state["y"] = 124.0
        _, repeated_descent = reward_system.calculate_reward(state)
        self.assertAlmostEqual(repeated_descent["ground_progress"], 1.00)
        self.assertAlmostEqual(
            ascent["ground_progress"] + repeated_descent["ground_progress"],
            0.0,
        )

        self.assertEqual(
            reward_system.get_terminal_rewards()["successful_landing"],
            4000.0,
        )

    def test_ground_progress_is_stronger_below_ten_metres(self):
        reward_system = V2ThreeEngineRewardSystem()
        state = make_state(y=30.0, vy=-1.0)

        reward_system.calculate_reward(state)
        state["y"] -= 0.05
        _, descent = reward_system.calculate_reward(state)
        self.assertAlmostEqual(descent["ground_progress"], 0.15)

        state["y"] += 0.05
        state["vy"] = 1.0
        _, ascent = reward_system.calculate_reward(state)
        self.assertAlmostEqual(ascent["ground_progress"], -0.15)
        self.assertAlmostEqual(ascent["ascent"], -0.075)

    def test_ascent_and_terminal_penalties_are_unambiguous(self):
        reward_system = V2ThreeEngineRewardSystem()
        state = make_state(y=125.0, vy=5.0)
        reward_system.calculate_reward(state)
        state["y"] += 0.25
        _, components = reward_system.calculate_reward(state)
        self.assertAlmostEqual(components["ascent"], -0.375)
        self.assertNotIn("pwm_switching", components)

        terminal = reward_system.get_terminal_rewards()
        self.assertEqual(terminal["successful_landing"], 4000.0)
        self.assertEqual(terminal["crash"], -2500.0)
        self.assertEqual(terminal["timeout"], -2000.0)

    def test_timeout_reward_is_applied_only_at_the_actual_limit(self):
        rocket = make_v2(make_state(y=125.0, vy=0.0))

        _, before = rocket.calculate_reward(
            state=rocket.state,
            action=v2_action([0.0, 0.0, 0.0]),
            step_id=749,
            max_steps=750,
        )
        _, at_limit = rocket.calculate_reward(
            state=rocket.state,
            action=v2_action([0.0, 0.0, 0.0]),
            step_id=750,
            max_steps=750,
        )

        self.assertNotIn("terminal_timeout", before)
        self.assertEqual(at_limit["terminal_timeout"], -2000.0)

    def test_terminal_time_pressure_remains_disabled(self):
        reward_system = V2ThreeEngineRewardSystem()

        _, above_terminal_band = reward_system.calculate_reward(
            make_state(y=50.0, vy=0.0)
        )
        self.assertEqual(
            above_terminal_band["terminal_time_pressure"], 0.0
        )

        _, low_hover = reward_system.calculate_reward(
            make_state(y=35.0, vy=0.0)
        )
        self.assertAlmostEqual(
            low_hover["terminal_time_pressure"], 0.0
        )

    def test_target_distance_cost_penalizes_airborne_hover_only(self):
        reward_system = V2ThreeEngineRewardSystem()

        _, airborne = reward_system.calculate_reward(
            make_state(x=0.0, y=125.0, vy=0.0)
        )
        self.assertAlmostEqual(airborne["target_distance_cost"], -0.40)

        contact_state = make_state(x=0.0, y=25.0, vy=0.0)
        contact_state["touchdown_contact"] = True
        _, contact = reward_system.calculate_reward(contact_state)
        self.assertEqual(contact["target_distance_cost"], 0.0)

        _, touchdown = reward_system.calculate_reward(
            make_state(y=25.0, vy=0.0, touchdown_contact=True)
        )
        self.assertEqual(touchdown["terminal_time_pressure"], 0.0)

    def test_deadstick_penalty_only_applies_once_to_irreversible_shutdown(self):
        reward_system = V2ThreeEngineRewardSystem()
        state = make_state(
            y=150.0,
            vy=-10.0,
            curriculum_engine_mask=[False, True, False],
            engine_operational=[True, True, True],
            engine_locked_out=[False, False, False],
            engines_on=[False, False, False],
            touchdown_contact=False,
        )

        # All OFF before first ignition is recoverable and must not be punished.
        _, recoverable = reward_system.calculate_reward(state)
        self.assertEqual(recoverable["deadstick_shutdown"], 0.0)

        # Only the centre engine is enabled in Phase 1A. Once it is locked out,
        # the two curriculum-disabled engines do not hide the deadstick state.
        state["engine_locked_out"][1] = True
        _, deadstick = reward_system.calculate_reward(state)
        self.assertEqual(deadstick["deadstick_shutdown"], -300.0)

        _, repeated = reward_system.calculate_reward(state)
        self.assertEqual(repeated["deadstick_shutdown"], 0.0)

        reward_system.reset_episode_memory()
        state["touchdown_contact"] = True
        _, touchdown = reward_system.calculate_reward(state)
        self.assertEqual(touchdown["deadstick_shutdown"], 0.0)

    def test_unignited_reserve_is_not_irreversible_deadstick(self):
        reward_system = V2ThreeEngineRewardSystem()
        state = make_state(
            y=150.0,
            vy=-10.0,
            curriculum_engine_mask=[True, True, True],
            engine_operational=[True, True, True],
            engine_locked_out=[True, True, False],
            engines_on=[False, False, False],
            touchdown_contact=False,
        )

        _, coasting = reward_system.calculate_reward(state)
        self.assertEqual(coasting["deadstick_shutdown"], 0.0)
        self.assertEqual(coasting["unpowered_descent_risk"], 0.0)

        reward_system.reset_episode_memory()
        state["engines_on"][2] = True
        _, reserve_ignited = reward_system.calculate_reward(state)
        self.assertEqual(reserve_ignited["deadstick_shutdown"], 0.0)

    def test_shutdown_option_cost_allows_staging_but_protects_last_engine(self):
        reward_system = V2ThreeEngineRewardSystem()
        state = make_state(
            y=550.0,
            vy=-90.0,
            curriculum_engine_mask=[True, True, True],
            engine_operational=[True, True, True],
            engine_locked_out=[False, False, False],
            engines_on=[True, True, True],
            touchdown_contact=False,
        )
        reward_system.calculate_reward(state)

        state["engine_locked_out"] = [True, False, False]
        state["engines_on"] = [False, True, True]
        _, first_shutdown = reward_system.calculate_reward(state)
        self.assertEqual(first_shutdown["shutdown_option_cost"], -1.0)
        self.assertNotIn("shutdown_kinetic_cost", first_shutdown)
        self.assertEqual(first_shutdown["deadstick_shutdown"], 0.0)

        state["engine_locked_out"] = [True, True, False]
        state["engines_on"] = [False, False, True]
        _, second_shutdown = reward_system.calculate_reward(state)
        self.assertEqual(second_shutdown["shutdown_option_cost"], -1.0)
        self.assertEqual(second_shutdown["deadstick_shutdown"], 0.0)

        state["engine_locked_out"] = [True, True, True]
        state["engines_on"] = [False, False, False]
        _, third_shutdown = reward_system.calculate_reward(state)
        self.assertEqual(third_shutdown["shutdown_option_cost"], -25.0)

        _, repeated = reward_system.calculate_reward(state)
        self.assertEqual(repeated["shutdown_option_cost"], 0.0)

    def test_unpowered_fall_risk_requires_no_available_engines(self):
        reward_system = V2ThreeEngineRewardSystem()
        state = make_state(
            y=425.0,
            vy=-30.0,
            curriculum_engine_mask=[True, True, True],
            engine_operational=[True, True, True],
            engine_locked_out=[False, False, False],
            engines_on=[False, False, False],
            touchdown_contact=False,
        )
        _, high_coast = reward_system.calculate_reward(state)
        self.assertEqual(high_coast["unpowered_descent_risk"], 0.0)

        state["y"] = 225.0  # No altitude-based ignition incentive.
        _, low_coast = reward_system.calculate_reward(state)
        self.assertEqual(low_coast["unpowered_descent_risk"], 0.0)

        state["y"] = 223.0
        _, coast_progress = reward_system.calculate_reward(state)
        self.assertAlmostEqual(coast_progress["ground_progress"], 2.0)
        self.assertEqual(coast_progress["unpowered_descent_risk"], 0.0)

        state["engine_locked_out"] = [True, True, True]
        state["y"] = 221.0
        _, deadstick = reward_system.calculate_reward(state)
        self.assertEqual(deadstick["deadstick_shutdown"], -400.0)
        self.assertEqual(deadstick["unpowered_descent_risk"], -1.0)
        self.assertAlmostEqual(deadstick["ground_progress"], 2.0)

    def test_over_rotation_barrier_starts_after_ninety_degrees(self):
        reward_system = V2ThreeEngineRewardSystem()
        _, belly_flop = reward_system.calculate_reward(
            make_state(theta=math.radians(85.0))
        )
        self.assertEqual(belly_flop["over_rotation"], 0.0)

        _, nose_first = reward_system.calculate_reward(
            make_state(theta=math.radians(105.0))
        )
        self.assertAlmostEqual(nose_first["over_rotation"], -1.5)

    def test_safe_touchdown_contact_bonus_is_one_shot(self):
        reward_system = V2ThreeEngineRewardSystem()
        state = make_state(
            x=2.0,
            y=25.0,
            vy=0.0,
            theta=math.radians(2.0),
            vtheta=math.radians(1.0),
            just_touched=True,
            impact_velocity=3.0,
            impact_theta=math.radians(2.0),
        )
        _, contact = reward_system.calculate_reward(state)
        self.assertEqual(contact["safe_touchdown_contact"], 1000.0)
        self.assertGreater(contact["touchdown_quality"], 0.0)

        state["just_touched"] = False
        _, after_contact = reward_system.calculate_reward(state)
        self.assertEqual(after_contact["safe_touchdown_contact"], 0.0)
        self.assertEqual(after_contact["touchdown_quality"], 0.0)

    def test_v2_compact_physical_scales(self):
        rocket = make_v2(make_state(
            x=50.0,
            y=525.0,
            vx=20.0,
            vy=-50.0,
            theta=math.radians(45.0),
            vtheta=math.radians(15.0),
        ))
        observation = rocket.flatten(rocket.state)

        np.testing.assert_allclose(
            observation[[0, 1, 2, 3, 4, 5]],
            [0.125, 2.0 * (525.0 - rocket.world_y_min) /
             (rocket.world_y_max - rocket.world_y_min) - 1.0,
             0.4, -0.5, 0.25, math.radians(15.0)],
            atol=1e-6,
        )
        self.assertEqual(observation.shape, (17,))

    def test_approach_costs_distinguish_centered_upright_state(self):
        centered = V2ThreeEngineRewardSystem()
        displaced = V2ThreeEngineRewardSystem()

        _, centered_components = centered.calculate_reward(
            make_state(x=0.0, y=125.0, vx=0.0, theta=0.0)
        )
        _, displaced_components = displaced.calculate_reward(
            make_state(
                x=15.0,
                y=125.0,
                vx=5.0,
                theta=math.radians(5.0),
            )
        )

        self.assertLess(displaced_components["pad_offset"], 0.0)
        self.assertLess(displaced_components["lateral_speed"], 0.0)
        self.assertLess(displaced_components["upright"], 0.0)
        self.assertEqual(centered_components["pad_offset"], 0.0)
        self.assertEqual(centered_components["upright"], 0.0)

    def test_touchdown_quality_rewards_near_miss_more_than_bad_impact(self):
        near = V2ThreeEngineRewardSystem()
        bad = V2ThreeEngineRewardSystem()
        _, near_components = near.calculate_reward(make_state(
            x=16.0,
            y=25.0,
            vy=-5.5,
            theta=math.radians(6.0),
            vtheta=math.radians(3.5),
            just_touched=True,
            impact_velocity=5.5,
            impact_theta=math.radians(6.0),
        ))
        _, bad_components = bad.calculate_reward(make_state(
            x=60.0,
            y=25.0,
            vy=-20.0,
            theta=math.radians(45.0),
            vtheta=math.radians(15.0),
            just_touched=True,
            impact_velocity=20.0,
            impact_theta=math.radians(45.0),
        ))

        self.assertGreater(
            near_components["touchdown_quality"],
            bad_components["touchdown_quality"],
        )

    def test_v2_phase_1_engine_curriculum_masks_actuators(self):
        expected = {
            "v2_phase_1a": [False, True, False],
            "v2_phase_1b": [True, True, True],
            "v2_phase_1c": [True, True, True],
            "v2_phase_1": [True, True, True],
        }
        expected_initial_on = {
            "v2_phase_1a": [False, False, False],
            "v2_phase_1b": [False, False, False],
            "v2_phase_1c": [False, False, False],
            "v2_phase_1": [False, False, False],
        }
        for phase, available in expected.items():
            with self.subTest(phase=phase):
                rocket = make_v2(phase=phase)
                self.assertEqual(rocket.curriculum_engine_mask, available)
                self.assertEqual(rocket.engine_operational, [True, True, True])
                self.assertEqual(rocket.engine_failed, [False, False, False])
                self.assertEqual(rocket.engine_locked_out, [False, False, False])
                observation = rocket.flatten(rocket.state)
                np.testing.assert_array_equal(
                    observation[14:17],
                    [(-1.0 if not enabled else float(on))
                     for enabled, on in zip(available, expected_initial_on[phase])],
                )
                np.testing.assert_allclose(
                    observation[11:14],
                    np.asarray(expected_initial_on[phase]) * 0.30,
                    atol=1e-7,
                )

        phase_1a = make_v2(phase="v2_phase_1a")
        np.testing.assert_allclose(
            np.asarray(phase_1a.engines_thrust) / phase_1a.engine_thrust_sl,
            [0.0, 0.0, 0.0],
            atol=1e-7,
        )
        # OFF before ignition leaves the centre engine available.
        phase_1a.step(v2_action(
            [1.0, 1.0, 1.0], switches=[-1.0, -1.0, -1.0]
        ))
        np.testing.assert_allclose(
            np.asarray(phase_1a.engines_thrust) / phase_1a.engine_thrust_sl,
            [0.0, 0.0, 0.0],
            atol=1e-7,
        )
        self.assertEqual(phase_1a.engine_locked_out, [False, False, False])

        phase_1a.step(v2_action([1.0, 1.0, 1.0]))
        self.assertEqual(phase_1a.engines_on, [False, True, False])
        self.assertAlmostEqual(phase_1a.engine_throttles[1], 0.30)
        np.testing.assert_array_equal(
            phase_1a.flatten(phase_1a.state)[14:17], [-1.0, 1.0, -1.0]
        )
        phase_1a.step(v2_action([1.0, 1.0, 1.0], switches=[-1.0]*3))
        self.assertEqual(phase_1a.engine_locked_out, [False, True, False])
        phase_1a.step(v2_action([1.0, 1.0, 1.0]))
        self.assertEqual(phase_1a.engines_on, [False, False, False])
        np.testing.assert_array_equal(
            phase_1a.flatten(phase_1a.state)[14:17], [-1.0, -1.0, -1.0]
        )

        for phase in ("v2_phase_1b", "v2_phase_1c"):
            with self.subTest(lifecycle=phase):
                rocket = make_v2(phase=phase)
                rocket.step(v2_action([1.0]*3, switches=[-1.0]*3))
                self.assertEqual(rocket.engines_on, [False]*3)
                self.assertEqual(rocket.engine_locked_out, [False]*3)
                # First ignition must start at 30%, regardless of command.
                rocket.step(v2_action([1.0]*3))
                self.assertEqual(rocket.engines_on, expected[phase])
                np.testing.assert_allclose(
                    rocket.engine_throttles, np.array(expected[phase])*0.30
                )
                # Shut down only the left engine; the others remain usable.
                rocket.step(v2_action([0.5]*3, switches=[-1.0, 1.0, 1.0]))
                self.assertEqual(rocket.engine_locked_out, [True, False, False])
                rocket.step(v2_action([1.0]*3))
                self.assertEqual(rocket.engines_on,
                                 [False, expected[phase][1], True])
                self.assertEqual(rocket.engine_throttles[0], 0.0)
                self.assertEqual(rocket.flatten(rocket.state)[14], -1.0)
                rocket.reset(make_state())
                self.assertEqual(rocket.engines_on, [False]*3)
                self.assertEqual(rocket.engine_locked_out, [False]*3)

    def test_v2_has_seventeen_observations_and_nine_actions(self):
        rocket = make_v2()

        self.assertEqual(rocket.num_engines, 3)
        self.assertEqual(rocket.state_dims, 17)
        self.assertEqual(rocket.action_dims, 9)
        self.assertEqual(rocket.engine_x_offsets, (-1.15, 0.0, 1.15))
        self.assertAlmostEqual(rocket.min_engine_throttle, 0.30)
        self.assertAlmostEqual(rocket.engine_switch_on_threshold, 0.20)
        self.assertAlmostEqual(rocket.engine_switch_off_threshold, 0.0)
        self.assertFalse(hasattr(rocket, "engine_throttle_time_constant"))
        self.assertAlmostEqual(rocket.engine_thrust_sl, 2.45e6)
        observation = rocket.flatten(rocket.state)
        self.assertEqual(observation.shape, (17,))
        np.testing.assert_array_equal(observation[11:14], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(observation[14:17], [0.0, 0.0, 0.0])

    def test_ignition_starts_at_minimum_before_applying_linear_throttle(self):
        rocket = make_v2()

        rocket.step(v2_action([0.2, 0.5, 1.0]))
        ignition_fractions = (
            np.asarray(rocket.engines_thrust) / rocket.engine_thrust_sl
        )
        np.testing.assert_allclose(
            ignition_fractions, [0.30, 0.30, 0.30], atol=1e-7
        )

        rocket.step(v2_action([0.2, 0.5, 1.0]))
        fractions = np.asarray(rocket.engines_thrust) / rocket.engine_thrust_sl
        expected = np.asarray([0.44, 0.65, 1.0])

        np.testing.assert_allclose(fractions, expected, atol=1e-7)
        observation = rocket.flatten(rocket.state)
        np.testing.assert_allclose(observation[11:14], expected, atol=1e-7)

    def test_v2_normalized_gimbal_maps_linearly_to_thirty_degrees(self):
        rocket = make_v2(make_state(y=550.0, vy=-20.0))
        action = v2_action(
            [0.0, 0.2, 0.0],
            gimbals=[0.0, 0.1, 0.0],
            switches=[-1.0, 1.0, -1.0],
        )

        # +0.1 of the normalized range means +3 degrees. The existing slew
        # limiter reaches that target after three 1-degree physics steps.
        for _ in range(5):
            rocket.step(action)

        self.assertAlmostEqual(
            rocket.engine_gimbals[1],
            math.radians(3.0),
            places=7,
        )

    def test_strong_off_command_locks_an_ignited_engine(self):
        rocket = make_v2()

        rocket.step(v2_action([0.0, 0.5, 0.0], switches=[-1.0, 1.0, -1.0]))
        self.assertEqual(rocket.engines_on, [False, True, False])
        self.assertAlmostEqual(
            rocket.engines_thrust[1] / rocket.engine_thrust_sl, 0.30
        )
        rocket.step(v2_action([0.0, 0.5, 0.0], switches=[-1.0, 1.0, -1.0]))
        expected_center = 0.65
        self.assertAlmostEqual(
            rocket.engines_thrust[1] / rocket.engine_thrust_sl,
            expected_center,
        )
        observation = rocket.flatten(rocket.state)
        np.testing.assert_allclose(
            observation[11:14], [0.0, expected_center, 0.0], atol=1e-7
        )
        np.testing.assert_array_equal(observation[14:17], [0.0, 1.0, 0.0])

        rocket.step(v2_action([0.0, 0.5, 0.0], switches=[-1.0, -1.0, -1.0]))
        self.assertEqual(rocket.engines_on, [False, False, False])
        self.assertEqual(rocket.engine_locked_out, [False, True, False])
        observation = rocket.flatten(rocket.state)
        np.testing.assert_array_equal(observation[11:14], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(observation[14:17], [0.0, -1.0, 0.0])

        rocket.step(v2_action([0.0, 1.0, 0.0], switches=[-1.0, 1.0, -1.0]))
        self.assertEqual(rocket.engines_on, [False, False, False])
        self.assertEqual(rocket.engines_thrust[1], 0.0)
        rocket.step(v2_action([0.0, 1.0, 0.0], switches=[-1.0, 1.0, -1.0]))
        self.assertEqual(rocket.engines_thrust[1], 0.0)

    def test_off_engine_stays_available_until_its_first_ignition(self):
        rocket = make_v2()

        rocket.step(v2_action([0.0, 0.0, 0.0], switches=[-1.0, -1.0, -1.0]))
        self.assertEqual(rocket.engine_locked_out, [False, False, False])

        # A command inside the deadband preserves the initial OFF state.
        rocket.step(v2_action([0.0, 0.0, 0.0], switches=[-1.0, 0.01, -1.0]))
        self.assertFalse(rocket.engines_on[1])
        self.assertFalse(rocket.engine_locked_out[1])

        rocket.step(v2_action([0.0, 0.0, 0.0], switches=[-1.0, 0.20, -1.0]))
        self.assertTrue(rocket.engines_on[1])
        self.assertAlmostEqual(
            rocket.engines_thrust[1] / rocket.engine_thrust_sl, 0.30
        )

        # Once ON, an ambiguous command holds ON instead of causing an
        # accidental irreversible shutdown.
        rocket.step(v2_action([0.0, 1.0, 0.0], switches=[-1.0, 0.10, -1.0]))
        self.assertTrue(rocket.engines_on[1])
        self.assertFalse(rocket.engine_locked_out[1])

        rocket.step(v2_action([0.0, 0.5, 0.0], switches=[-1.0, 0.0, -1.0]))
        self.assertFalse(rocket.engines_on[1])
        self.assertTrue(rocket.engine_locked_out[1])
        self.assertEqual(rocket.engines_thrust[1], 0.0)
        rocket.step(v2_action([0.0, 0.5, 0.0], switches=[-1.0, 1.0, -1.0]))
        self.assertEqual(rocket.engines_thrust[1], 0.0)

    def test_mirrored_state_and_actions_produce_mirrored_dynamics(self):
        theta = math.radians(40.0)
        omega = math.radians(-2.0)
        left = make_v2(
            make_state(x=80.0, vx=-8.0, theta=theta, vtheta=omega)
        )
        right = make_v2(
            make_state(x=-80.0, vx=8.0, theta=-theta, vtheta=-omega)
        )

        action_left = v2_action(
            [0.45, 0.70, 0.20],
            [math.radians(5), math.radians(-3), math.radians(2)],
            [1.0, 1.0, -1.0],
        )
        action_right = v2_action(
            [0.20, 0.70, 0.45],
            [math.radians(-2), math.radians(3), math.radians(-5)],
            [-1.0, 1.0, 1.0],
        )

        _, reward_left, done_left, _ = left.step(action_left)
        _, reward_right, done_right, _ = right.step(action_right)
        state_left = left.state
        state_right = right.state

        self.assertAlmostEqual(state_left["x"], -state_right["x"], places=7)
        self.assertAlmostEqual(state_left["vx"], -state_right["vx"], places=7)
        self.assertAlmostEqual(state_left["theta"], -state_right["theta"], places=7)
        self.assertAlmostEqual(state_left["vtheta"], -state_right["vtheta"], places=7)
        self.assertAlmostEqual(state_left["y"], state_right["y"], places=7)
        self.assertAlmostEqual(state_left["vy"], state_right["vy"], places=7)
        self.assertAlmostEqual(reward_left, reward_right, places=7)
        self.assertEqual(done_left, done_right)

        np.testing.assert_allclose(
            left.engines_thrust,
            np.asarray(right.engines_thrust)[::-1],
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            left.engine_gimbals,
            -np.asarray(right.engine_gimbals)[::-1],
            rtol=0.0,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()

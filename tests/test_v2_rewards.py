"""Reward-only regressions; no renderer, Gymnasium or PyTorch required."""

import math
import unittest

from improved_rewards_v2 import V2ThreeEngineRewardSystem


def state(altitude=100.0, **overrides):
    values = dict(
        x=0.0, y=25.0 + altitude, vx=0.0, vy=-10.0,
        theta=0.0, vtheta=0.0,
        curriculum_engine_mask=[True, True, True],
        engine_operational=[True, True, True],
        engine_locked_out=[False, False, False],
        engines_on=[False, False, False],
        touchdown_contact=False,
    )
    values.update(overrides)
    return values


class V2RewardRegressionTests(unittest.TestCase):
    def test_same_transition_does_not_depend_on_historical_low(self):
        rewards = [V2ThreeEngineRewardSystem(), V2ThreeEngineRewardSystem()]
        rewards[0].calculate_reward(state(50.0))
        rewards[1].calculate_reward(state(100.0))
        for reward in rewards:
            reward.calculate_reward(state(60.0))
        results = [reward.calculate_reward(state(59.0)) for reward in rewards]
        self.assertEqual(results[0], results[1])
        self.assertAlmostEqual(results[0][1]['ground_progress'], 1.0)

    def test_closed_altitude_cycles_cannot_accumulate_progress(self):
        for heights in (
            [30.0, 9.0, 15.0, 24.0, 30.0],
            [30.0, 24.0, 15.0, 9.0, 30.0],
            [12.0, 11.0, 9.0, 12.0],
        ):
            with self.subTest(heights=heights):
                reward = V2ThreeEngineRewardSystem()
                components = [reward.calculate_reward(state(h))[1] for h in heights]
                self.assertAlmostEqual(sum(c['ground_progress'] for c in components), 0.0)
                self.assertLess(sum(sum(c.values()) for c in components), 0.0)

    def test_progress_is_independent_of_transition_partition(self):
        progress = []
        for heights in ([100.0, 0.0], [100.0, 25.0, 18.0, 10.0, 5.0, 0.0]):
            reward = V2ThreeEngineRewardSystem()
            progress.append(sum(
                reward.calculate_reward(state(h))[1]['ground_progress']
                for h in heights
            ))
        self.assertAlmostEqual(progress[0], 135.0)
        self.assertAlmostEqual(progress[0], progress[1])

    def test_hover_does_not_earn_progress_at_any_altitude(self):
        for altitude in (0.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 500.0):
            with self.subTest(altitude=altitude):
                reward = V2ThreeEngineRewardSystem()
                reward.calculate_reward(state(altitude, vy=0.0))
                total, components = reward.calculate_reward(state(altitude, vy=0.0))
                self.assertEqual(components['ground_progress'], 0.0)
                self.assertEqual(components['terminal_time_pressure'], 0.0)
                self.assertEqual(components['vertical_speed_progress'], 0.0)
                self.assertLess(total, 0.0)

    def test_coasting_and_powered_descent_have_same_progress(self):
        coast = V2ThreeEngineRewardSystem()
        powered = V2ThreeEngineRewardSystem()
        for altitude in (200.0, 198.0, 80.0, 79.0, 20.0, 19.0, 6.0, 5.0):
            a = coast.calculate_reward(state(altitude))
            b = powered.calculate_reward(state(altitude, engines_on=[False, True, False]))
            self.assertEqual(a, b)

    def test_staging_to_unignited_reserve_does_not_trigger_deadstick(self):
        reward = V2ThreeEngineRewardSystem()
        reward.calculate_reward(state(100.0, engines_on=[True, False, True]))
        coast = state(99.0, engine_locked_out=[True, False, True])
        _, components = reward.calculate_reward(coast)
        self.assertEqual(components['shutdown_option_cost'], -2.0)
        self.assertEqual(components['deadstick_shutdown'], 0.0)
        self.assertEqual(components['unpowered_descent_risk'], 0.0)
        _, reserve_ignition = reward.calculate_reward(dict(coast, engines_on=[False, True, False]))
        self.assertEqual(reserve_ignition['deadstick_shutdown'], 0.0)

    def test_losing_last_engine_is_charged_on_transition_even_ascending(self):
        reward = V2ThreeEngineRewardSystem()
        initial = state(100.0, vy=2.0, engine_locked_out=[True, False, True],
                        engines_on=[False, True, False])
        reward.calculate_reward(initial)
        lost = dict(initial, engine_locked_out=[True, True, True], engines_on=[False]*3)
        _, event = reward.calculate_reward(lost)
        self.assertEqual(event['deadstick_shutdown'], -200.0)
        self.assertEqual(event['shutdown_option_cost'], -25.0)
        _, descending = reward.calculate_reward(dict(lost, vy=-15.0))
        self.assertEqual(descending['deadstick_shutdown'], 0.0)
        self.assertEqual(descending['unpowered_descent_risk'], -0.5)

    def test_no_engine_management_penalties_on_contact(self):
        reward = V2ThreeEngineRewardSystem()
        reward.calculate_reward(state(2.0, engines_on=[True]*3))
        _, components = reward.calculate_reward(state(
            0.0, engine_locked_out=[True]*3, touchdown_contact=True,
        ))
        for key in ('shutdown_option_cost', 'deadstick_shutdown', 'unpowered_descent_risk'):
            self.assertEqual(components[key], 0.0)

    def test_masked_or_failed_engines_do_not_count_as_reserves(self):
        for overrides in (
            dict(curriculum_engine_mask=[False, True, False]),
            dict(engine_operational=[False, True, False]),
        ):
            with self.subTest(overrides=overrides):
                reward = V2ThreeEngineRewardSystem()
                reward.calculate_reward(state(**overrides))
                _, components = reward.calculate_reward(state(
                    **overrides, engine_locked_out=[False, True, False],
                ))
                self.assertEqual(components['deadstick_shutdown'], -300.0)
                self.assertAlmostEqual(components['unpowered_descent_risk'], -1.0/3.0)

    def test_reset_removes_transition_memory(self):
        reward = V2ThreeEngineRewardSystem()
        reward.calculate_reward(state(100.0))
        reward.reset_episode_memory()
        _, components = reward.calculate_reward(state(50.0, engine_locked_out=[True]*3))
        self.assertEqual(components['ground_progress'], 0.0)
        self.assertEqual(components['deadstick_shutdown'], 0.0)
        self.assertEqual(components['shutdown_option_cost'], 0.0)

    def test_rewards_are_mirror_symmetric(self):
        left, right = V2ThreeEngineRewardSystem(), V2ThreeEngineRewardSystem()
        for altitude, locks in ((150.0, [False]*3), (149.0, [True, False, False]),
                                (148.0, [True, True, True])):
            a = state(altitude, x=-20.0, vx=2.0, theta=-0.1, vtheta=0.05,
                      engine_locked_out=locks)
            b = dict(a, x=-a['x'], vx=-a['vx'], theta=-a['theta'], vtheta=-a['vtheta'],
                     engine_locked_out=locks[::-1])
            self.assertEqual(left.calculate_reward(a), right.calculate_reward(b))

    def test_existing_terminal_rewards_and_speed_cost_are_preserved(self):
        reward = V2ThreeEngineRewardSystem()
        _, components = reward.calculate_reward(state(10.0, vy=-15.0))
        self.assertAlmostEqual(components['vertical_speed'], -0.175)
        self.assertEqual(components['time_cost'], -0.20)
        self.assertTrue(all(math.isfinite(value) for value in components.values()))
        self.assertEqual(reward.get_terminal_rewards(), dict(
            successful_landing=4000.0, crash=-2500.0, timeout=-2000.0,
        ))


if __name__ == '__main__':
    unittest.main()

"""Configuration regressions independent of the renderer and RL stack."""

import unittest

from curriculum_phases import get_curriculum_phase


class CurriculumPhaseTests(unittest.TestCase):
    def test_phase_1b_only_changes_description_and_engine_availability(self):
        phase_a = get_curriculum_phase('v2_phase_1a')
        phase_b = get_curriculum_phase('v2_phase_1b')
        differences = {
            key for key in phase_a.keys() | phase_b.keys()
            if phase_a.get(key) != phase_b.get(key)
        }
        self.assertEqual(differences, {'description', 'engine_available_mask'})
        self.assertEqual(phase_a['engine_available_mask'], (False, True, False))
        self.assertEqual(phase_b['engine_available_mask'], (True, True, True))
        self.assertEqual(phase_b['initial_engine_on_mask'], (False, False, False))
        self.assertFalse(phase_b['force_initial_engines_on_in_flight'])

    def test_loading_phase_1b_returns_an_independent_copy(self):
        phase = get_curriculum_phase('v2_phase_1b')
        phase['altitude_agl_m'] = (1.0, 2.0)
        phase['engine_available_mask'] = (False, False, False)
        self.assertEqual(get_curriculum_phase('v2_phase_1b')['altitude_agl_m'], (80.0, 120.0))
        self.assertEqual(get_curriculum_phase('v2_phase_1b')['engine_available_mask'], (True, True, True))


if __name__ == '__main__':
    unittest.main()

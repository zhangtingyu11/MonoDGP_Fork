import unittest

import numpy as np

from lib.datasets.utils import clipped_box_bounds, paint_clipped_box


class NegativeCoordinateClippingTest(unittest.TestCase):
    def test_visible_box_part_is_clipped_to_canvas(self):
        mask = np.zeros((6, 8), dtype=bool)
        paint_clipped_box(mask, np.asarray((-2.7, 1.2, 3.9, 5.8)))

        expected = np.zeros_like(mask)
        expected[1:5, 0:3] = True
        np.testing.assert_array_equal(mask, expected)

    def test_fully_outside_negative_box_does_not_wrap(self):
        mask = np.zeros((6, 8), dtype=bool)
        paint_clipped_box(mask, np.asarray((-7.5, 1.0, -1.2, 5.0)))
        self.assertFalse(mask.any())

    def test_in_bounds_integer_conversion_is_unchanged(self):
        self.assertEqual(
            clipped_box_bounds(
                np.asarray((1.8, 2.7, 6.9, 5.9)), height=6, width=8),
            (2, 5, 1, 6))


if __name__ == '__main__':
    unittest.main()

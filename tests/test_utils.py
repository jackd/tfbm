import operator
import unittest

from tfbm import utils


class UtilsTest(unittest.TestCase):
    def test_group_by(self):
        x = ("a", "x", 2)
        y = ("a", "y", 3)
        z = ("b", "x", 4)
        groups = utils.group_by((x, y, z), operator.itemgetter(0))
        self.assertEqual(groups, {"a": [x, y], "b": [z]})

        groups = utils.group_by((x, y, z), operator.itemgetter(1))
        self.assertEqual(groups, {"x": [x, z], "y": [y]})


if __name__ == "__main__":
    unittest.main()

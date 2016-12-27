import unittest

from ex3 import count_values, information_gain, choose_attribute, split_examples_by_value

EXAMPLES = [['crew', 'adult', 'male', 'no'], ['crew', 'adult', 'male', 'yes'], ['crew', 'adult', 'male', 'no'],
            ['3rd', 'adult', 'male', 'no'], ['2nd', 'adult', 'male', 'no'], ['crew', 'adult', 'male', 'no'],
            ['2nd', 'adult', 'male', 'no'], ['crew', 'adult', 'male', 'yes'], ['crew', 'adult', 'male', 'no'],
            ['crew', 'adult', 'male', 'no']]
ATTRIBUTES = ['pclass', 'age', 'sex', 'survived']


class TestStringMethods(unittest.TestCase):
    def test_count_values(self):
        counter = count_values(ATTRIBUTES, EXAMPLES)
        self.assertEqual(counter, {})

    def test_information_gain(self):
        ig = information_gain(ATTRIBUTES, EXAMPLES, gain_measure)
        self.assertEqual(ig, {})

    def test_choose_attribute(self):
        self.assertEqual(choose_attribute(ATTRIBUTES, EXAMPLES, gain_measure), 'pclass')

    def test_split_examples_by_value(self):
        self.assertEqual(split_examples_by_value(EXAMPLES, 0), {})


if __name__ == '__main__':
    unittest.main()

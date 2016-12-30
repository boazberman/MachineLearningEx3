import sys
from collections import Counter
from math import log

import numpy as np

GAIN_MEASURE_VALUES = ["info-gain", "err"]


class AttributeNotFoundError(Exception):
    def __str__(self):
        return "Not found in tree"


class InvalidValueGiven(Exception):
    def __init__(self, value, legal):
        self.value = value
        self.legal = legal

    def __str__(self):
        message = "Invalid value given for %s" % self.value
        if self.legal is not None:
            message += ", legal values: %s" % self.legal
        return message


class TreeNode:
    def __init__(self, attribute=None, value=None):
        self.children = []
        self.attribute = attribute
        self.value = value

    def __str__(self):
        if self.value == "ROOT":
            return "'%s'" % (self.attribute) if not self.has_children() else "%s(%s)" % (
                self.attribute, ', '.join(str(child) for child in self.children))
        if self.has_children():
            return "%s -> %s(%s)" % (self.value, self.attribute, ', '.join(str(child) for child in self.children))
        return "%s: '%s'" % (self.value, self.attribute)

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def has_children(self):
        return len(self.children) > 0


def is_empty(iterable):
    return all(False for _ in iterable)


def same_classification(examples):
    cls = classification(examples)
    return all(cls == example[-1] for example in examples[1:])


def classification(examples):
    return examples[0][-1]


def majority_value(examples):
    return Counter(examples[:, -1]).most_common(1)[0][0]


def choose_attribute(attributes, examples, gain_measure):
    ig = information_gain(attributes, examples, gain_measure)
    max_ig = max(ig, key=lambda id: ig[id])
    return max_ig


def without_tag(examples):
    return examples[:, :-1]


def count_values(attributes, examples):
    # Each key: (tag, value, attribute)
    # counter = Counter()
    # Fucking dictionary of attributes where each attribute contains Counter of (value, tag)
    values = dict.fromkeys(attributes[:-1])
    for example in examples:
        for i, value in enumerate(example[:-1]):
            attribute = attributes[i]
            if values[attribute] is None:
                values[attribute] = {}
            if value not in values[attribute]:
                values[attribute][value] = {'count': 0, 'tags': Counter()}
            values[attribute][value]['count'] += 1
            values[attribute][value]['tags'][example[-1]] += 1
    return values


def calculate_c(taggings_counter, total_size, gain_measure):
    probs = [float(count) / total_size for tag, count in taggings_counter.iteritems()]
    return entropy(probs) if gain_measure == "info-gain" else min(probs)


def entropy(probs):
    return -1 * sum(prob * log(prob, 2) for prob in probs)


def information_gain(attributes, examples, gain_measure):
    '''
    :param gain_measure:
    :param examples:
    :param attributes:
    :return: list of each attribute's information gain
    '''
    s_entropy = calculate_c(Counter(examples[:, -1]), len(examples), gain_measure)
    values_count = count_values(attributes, examples)
    all_ig = {}
    for attribute in values_count:
        all_ig[attribute] = s_entropy
        values = values_count[attribute]
        for value in values:
            value_probability = float(values[value]['count']) / len(examples)
            value_appearances = values[value]['count']
            all_ig[attribute] -= value_probability * calculate_c(values[value]['tags'], value_appearances,
                                                                 gain_measure)
            # all_ig[attribute] += value_probability * sum(
            #     list((float(c) / value_appearances) * log(float(c) / value_appearances, 2) for tag, c in
            #          values[value]['tags'].iteritems()))
    return all_ig


def list_without(a_list, i):
    return np.delete(a_list, i)


def split_examples_by_value(examples, attribute_index, attribute_values):
    examples_by_value = {}
    for value in attribute_values:
        examples_by_value[value] = []
    for example in examples:
        value = example[attribute_index]
        examples_by_value[value].append(list_without(example, attribute_index))
    for value in examples_by_value:
        examples_by_value[value] = np.array(examples_by_value[value])
    return examples_by_value


def id3_algorithm(examples, attributes, attributes_values, gain_measure, default=None, value="ROOT"):
    '''
    :param attributes_values:
    :param gain_measure:
    :param value:
    :param examples: dictionary containing classification and a list of attributes
    :param attributes: list of all attributes
    :param default:
    :return:
    '''
    if is_empty(examples):
        return TreeNode(default, value)
    elif same_classification(examples):
        return TreeNode(classification(examples), value)
    elif is_empty(attributes[:-1]):
        return TreeNode(majority_value(examples), value)
    else:
        best_attribute = choose_attribute(attributes, examples, gain_measure)
        tree = TreeNode(best_attribute, value)
        examples_by_value = split_examples_by_value(examples, attributes.index(best_attribute),
                                                    attributes_values[best_attribute])
        for value, example_i in examples_by_value.iteritems():
            subtree = id3_algorithm(example_i, [a for a in attributes if a != best_attribute], attributes_values,
                                    gain_measure,
                                    majority_value(examples), value)
            tree.add_child(subtree)
        return tree


def extract_attributes_values(attributes, examples):
    attributes_values = {}
    for attribute in attributes:
        attributes_values[attribute] = set()
    for example in examples:
        for i, value in enumerate(example):
            attribute = attributes[i]
            attributes_values[attribute].add(value)
    return attributes_values


def train(train_file_path, gain_measure):
    text = np.loadtxt(train_file_path, dtype=str)
    attributes = text[0].tolist()
    examples = text[1:]
    attributes_values = extract_attributes_values(attributes, examples)
    return id3_algorithm(examples, attributes, attributes_values, gain_measure)


def predict(tree, example, attributes):
    while tree.has_children():
        value = example[attributes.index(tree.attribute)]
        tree = list(child for child in tree.get_children() if child.value == value)[0]
        if not tree.has_children():
            return tree.attribute
    raise AttributeNotFoundError()


def parse_and_predict(tree, validation_file_path):
    text = np.loadtxt(validation_file_path, dtype=str)
    attributes = text[0].tolist()
    examples = text[1:]
    correct = 0
    for i, example in enumerate(examples):
        prediction = predict(tree, example, attributes)
        correct += 1 if prediction == example[-1] else 0
        print '%s: %s' % (i, prediction)
    return float(correct) / len(examples)


def extract_gain_measure():
    measure_gain = sys.argv[3] if len(sys.argv) >= 4 else GAIN_MEASURE_VALUES[0]
    if measure_gain not in GAIN_MEASURE_VALUES:
        raise InvalidValueGiven("gain_measure", GAIN_MEASURE_VALUES)
    return measure_gain


if __name__ == '__main__':
    train_file_path = sys.argv[1] if len(sys.argv) >= 2 else "dataset/titanic_train.txt"
    validation_file_path = sys.argv[2] if len(sys.argv) >= 3 else "dataset/titanic_val.txt"
    gain_measure = extract_gain_measure()
    tree = train(train_file_path, gain_measure)
    accuracy = parse_and_predict(tree, validation_file_path)
    with open("output_tree.txt", 'w') as output_tree:
        output_tree.write(str(tree))
    with open("output_acc.txt", 'w') as output_acc:
        output_acc.write(str(accuracy))

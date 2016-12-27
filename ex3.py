import sys
from collections import Counter
from math import log

import numpy as np


class AttributeNotFoundError(Exception):
    def __str__(self):
        return "Not found in tree"


class TreeNode:
    def __init__(self, attribute=None, value=None):
        self.children = []
        self.attribute = attribute
        self.value = value

    def __str__(self):
        if self.has_children():
            return "%s -> %s(%s)" % (self.value, self.attribute, ', '.join(str(child) for child in self.children))
        return "%s -> '%s'" % (self.value, self.attribute)

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
    fucker = dict.fromkeys(attributes[:-1])
    for example in examples:
        for i, value in enumerate(example[:-1]):
            attribute = attributes[i]
            if fucker[attribute] is None:
                fucker[attribute] = {}
            if value not in fucker[attribute]:
                fucker[attribute][value] = {'count': 0, 'tags': Counter()}
            fucker[attribute][value]['count'] += 1
            fucker[attribute][value]['tags'][example[-1]] += 1
    return fucker


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


def split_examples_by_value(examples, attribute_index):
    examples_by_value = {}
    for example in examples:
        value = example[attribute_index]
        if value not in examples_by_value:
            examples_by_value[value] = []
        examples_by_value[value].append(list_without(example, attribute_index))
    for value in examples_by_value:
        examples_by_value[value] = np.array(examples_by_value[value])
    return examples_by_value


def id3_algorithm(examples, attributes, gain_measure, default=None, value="ROOT"):
    '''
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
        examples_by_value = split_examples_by_value(examples, attributes.index(best_attribute))
        for value, example_i in examples_by_value.iteritems():
            subtree = id3_algorithm(example_i, [a for a in attributes if a != best_attribute], gain_measure,
                                    majority_value(examples), value)
            tree.add_child(subtree)
        return tree


def train(train_file_path, gain_measure):
    text = np.loadtxt(train_file_path, dtype=str)
    attributes = text[0].tolist()
    examples = text[1:]
    return id3_algorithm(examples, attributes, gain_measure)


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
    print "Accuracy: %s" % (float(correct) / len(examples))


if __name__ == '__main__':
    train_file_path = sys.argv[1] if len(sys.argv) >= 2 else "dataset/titanic_train.txt"
    validation_file_path = sys.argv[2] if len(sys.argv) >= 3 else "dataset/titanic_val.txt"
    gain_measure = sys.argv[3] if len(sys.argv) >= 4 else "info-gain"
    tree = train(train_file_path, gain_measure)
    print tree
    parse_and_predict(tree, validation_file_path)

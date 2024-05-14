import numpy as np
from components.dataset.dataset import Dataset


class TreeNode(object):
    def __init__(self, ids=None, children=None, entropy=0, depth=0):
        if children is None:
            children = []
        self.ids = ids  # index of data in this node
        self.entropy = entropy  # entropy
        self.depth = depth  # distance to root node
        self.split_attribute = None  # which attribute is chosen, it non-leaf
        self.children = children  # list of its child nodes
        self.order = None  # order of values of split_attribute in children
        self.label = None  # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    # remove prob 0 
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0 / float(freq_0.sum())
    return -np.sum(prob_0 * np.log(prob_0))


class DecisionTreeID3(object):
    def __init__(self, input, labels, args):
        self.root = None
        self.args = args

        self.input = input
        self.labels = labels.unique()
        self.attributes = list(input)
        self.Ntrain = input.count()[0]

    def fit(self):
        ids = range(self.Ntrain)
        self.root = TreeNode(ids=ids, entropy=self.entropy(ids), depth=0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.args.max_depth or node.entropy < self.args.threshold:
                node.children = self.split(node)
                if not node.children:  # leaf node
                    self.set_label(node)
                queue += node.children
            else:
                self.set_label(node)

    def entropy(self, ids):
        # calculate entropy of a node with index ids
        if len(ids) == 0:
            return 0

        ids = [i + 1 for i in ids]  # panda series index starts from 1
        freq = np.array(self.labels[ids].value_counts())
        return entropy(freq)

    def set_label(self, node):
        # find label for a node if it is a leaf
        # simply chose by major voting 
        target_ids = [i + 1 for i in node.ids]  # target is a series variable
        node.set_label(self.labels[target_ids].mode()[0])  # most frequent label

    def split(self, node):
        ids = node.ids
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.input.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.input.iloc[ids, i].unique().tolist()
            if len(values) == 1: continue  # entropy = 0
            splits = []
            for val in values:
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id - 1 for sub_id in sub_ids])
            # don't split if a node has too small number of points
            if min(map(len, splits)) < self.args.min_samples_split: continue
            # information gain
            HxS = 0
            for split in splits:
                HxS += len(split) * self.entropy(split) / len(ids)
            gain = node.entropy - HxS
            if gain < self.args.threshold:
                continue  # stop if small gain
            if gain > best_gain:
                best_gain = gain
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids=split,
                                entropy=self.entropy(split), depth=node.depth + 1) for split in best_splits]
        return child_nodes

    def predict(self, data):
        npoints = data.count()[0]
        labels = [None] * npoints
        for n in range(npoints):
            x = data.iloc[n, :]  # one point
            # start from root and recursively travel if not meet a leaf 
            node = self.root
            while node.children:
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[n] = node.label

        return labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=200)
    parser.add_argument('--min_samples_split', type=int, default=2)

    args = parser.parse_args()

    dataset = Dataset(file_path='data/eval/loan_data_test.csv')
    dataset.load_data()

    tree = DecisionTreeID3(dataset.input, dataset.labels, args)
    tree.fit()

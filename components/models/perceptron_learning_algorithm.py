import numpy as np

np.random.seed(23)


def sign(w, x):
    return np.sign(np.dot(w.T, x))


def has_converged(x, y, w):
    return np.array_equal(sign(w, x.T), y)


class PLA:
    def __init__(self, input, labels, args):
        self.args = args
        self.weight = None
        self.input = input
        self.labels = labels

    def train(self):
        N = self.input.shape[0]
        dim = self.input.shape[1]
        w_init = np.random.rand(dim, 1)
        w = [w_init]

        count = 0

        while count <= self.args.epoch:
            mix_id = np.random.permutation(N)
            for id in mix_id:
                xi = self.input[id, :].reshape(dim, 1)
                yi = self.labels[id]
                if sign(w[-1], xi) != yi:
                    w_new = w[-1] + yi * xi
                    w.append(w_new)

            if has_converged(self.input, self.labels, w[-1]):
                break

            count += 1

        self.weight = w[-1]
        return

    def test(self, x):
        output = sign(self.weight, x)

        return max(output, 0)


if __name__ == '__main__':
    import argparse
    from components.dataset.dataset import Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)

    args = parser.parse_args()

    dataset = Dataset(file_path='data/train/loan_data.csv')
    dataset.load_data()

    dataset.labels = [-1 if x == 0 else 1 for x in dataset.labels]

    pla = PLA(dataset.input, dataset.labels, args)
    pla.train()

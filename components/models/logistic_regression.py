import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Logistic_Regression:
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
        while count < self.args.max_count:
            mix_id = np.random.permutation(N)

            for id in mix_id:
                xi = self.input[id, :].reshape(dim, 1)
                yi = self.labels[id]
                zi = sigmoid(np.dot(w[-1].T, xi))
                w_new = w[-1] + self.args.eta * (yi - zi) * xi
                count += 1

                # Điều kiện dừng
                if count % self.args.logging_step == 0:
                    if np.linalg.norm(w_new - w[-self.args.logging_step]) < self.args.threshold:
                        self.weight = w[-1]
                        return

                w.append(w_new)

        self.weight = w[-1]

        return

    def test(self, x):
        return sigmoid(np.dot(self.weight.T, x))


if __name__ == '__main__':
    import argparse
    from components.dataset.dataset import Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--max_count', type=int, default=10000)
    parser.add_argument('--logging_step', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=1e-5)

    args = parser.parse_args()

    dataset = Dataset(file_path='data/train/loan_data.csv')
    dataset.load_data()

    logistic = Logistic_Regression(dataset.input, dataset.labels, args)
    logistic.train()

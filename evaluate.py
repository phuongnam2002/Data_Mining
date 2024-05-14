import argparse
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, LogisticRegression

from components.dataset.dataset import Dataset
from components.models.decision_tree import DecisionTreeID3
from components.models.perceptron_learning_algorithm import PLA
from components.models.logistic_regression import Logistic_Regression

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--max_depth', type=int, default=200)
    parser.add_argument('--max_count', type=int, default=10000)
    parser.add_argument('--logging_step', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=1e-5)
    parser.add_argument('--min_samples_split', type=int, default=2)

    args = parser.parse_args()

    dataset = Dataset(file_path='data/train/loan_data_train.csv')

    dataset.load_data()

    logistic = Logistic_Regression(dataset.input, dataset.labels, args)
    logistic.train()

    logistic_model = LogisticRegression(random_state=0, max_iter=10000)
    logistic_model.fit(dataset.input, dataset.labels)

    tree = DecisionTreeID3(dataset.input, dataset.labels, args)
    tree.fit()

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(dataset.input, dataset.labels)

    dataset.labels = [-1 if x == 0 else 1 for x in dataset.labels]

    pla = PLA(dataset.input, dataset.labels, args)
    pla.train()

    perceptron_model = Perceptron(random_state=0, max_iter=10000, tol=1e-6)
    perceptron_model.fit(dataset.input, dataset.labels)

    test_data = Dataset(file_path='data/test/load_data_test.csv')
    test_data.load_data()

    predict_logistic = logistic.predict(test_data.input)
    accuracy_logistic = accuracy_score(predict_logistic, test_data.labels)

    predict_tree = tree.predict(test_data.input)
    accuracy_tree = accuracy_score(predict_tree, test_data.labels)

    test_data.labels = [-1 if x == 0 else 1 for x in test_data.labels]
    predict_pla = pla.predict(test_data.input)
    accuracy_pla = accuracy_score(predict_pla, test_data.labels)

    print('__________Accuracy Trên Tập Kiểm Thử Của Nhóm__________')
    print(f'Logistic Regression: {accuracy_logistic}')
    print(f'Perceptron Learning Algorithm: {accuracy_pla}')
    print(f'Decision Tree: {accuracy_tree}')

    print()

    print('__________Accuracy Trên Tập Kiểm Thử Của Thư Viện__________')
    predict_logistic = logistic_model.predict(test_data.input)
    accuracy_logistic = accuracy_score(predict_logistic, test_data.labels)

    predict_tree = tree.predict(test_data.input)
    accuracy_tree = accuracy_score(predict_tree, test_data.labels)

    test_data.labels = [-1 if x == 0 else 1 for x in test_data.labels]
    predict_pla = pla.predict(test_data.input)
    accuracy_pla = accuracy_score(predict_pla, test_data.labels)

    print(f'Logistic Regression: {accuracy_logistic}')
    print(f'Perceptron Learning Algorithm: {accuracy_pla}')
    print(f'Decision Tree: {accuracy_tree}')

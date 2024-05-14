import os
import sys
import argparse
import numpy as np
from django.views import View
from django.shortcuts import render
from django.http import JsonResponse

from components.dataset.dataset import Dataset
from components.models.decision_tree import DecisionTreeID3
from components.models.perceptron_learning_algorithm import PLA
from components.models.logistic_regression import Logistic_Regression

np.random.seed(23)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

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

tree = DecisionTreeID3(dataset.input, dataset.labels, args)
tree.fit()

dataset.labels = [-1 if x == 0 else 1 for x in dataset.labels]

pla = PLA(dataset.input, dataset.labels, args)
pla.train()


# Create your models here.
class IndexView(View):
    template_name = 'index.html'

    def __init__(self, **kwargs):
        super(IndexView).__init__(**kwargs)

    def get(self, request):
        context = {
            'image_url': os.path.join('/static', 'diamond.png')
        }
        return render(request, self.template_name, context=context)

    def post(self, request):
        credit_policy = float(request.POST['credit_policy'])
        int_rate = float(request.POST['int_rate'])
        installment = float(request.POST['installment'])
        log_annual_inc = float(request.POST['log_annual_inc'])
        dti = float(request.POST['dti'])
        fico = float(request.POST['fico'])
        revol_bal = float(request.POST['revol_bal'])
        inq_last_6mths = float(request.POST['inq_last_6mths'])
        delinq_2yrs = float(request.POST['delinq_2yrs'])
        algo = request.POST['algo']

        input = np.array(
            [[credit_policy, int_rate, installment, log_annual_inc, dti, fico, revol_bal, inq_last_6mths, delinq_2yrs]]
        )

        if algo == "Logistic Regression":
            answer = logistic.predict(input)[0]
        elif algo == "Perceptron Learning Algorithm":
            answer = pla.predict(input)[0]
        else:
            answer = tree.predict(input)[0]

        if answer == 0:
            answer = "Người này không đủ khả năng chi trả khoản nợ"
        else:
            answer = "Người này có đủ khả năng chi trả khoản nợ"

        data = {
            'answer': answer
        }
        return JsonResponse(data)

import sys
import os
import numpy as np
from django.views import View
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, LogisticRegression

from components.dataset.dataset import Dataset

np.random.seed(23)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

dataset = Dataset(file_path='data/train/loan_data.csv')
dataset.load_data()

logistic_model = LogisticRegression(random_state=0, max_iter=10000)
logistic_model.fit(dataset.input, dataset.labels)

perceptron_model = Perceptron(random_state=0, max_iter=10000, tol=1e-6)
perceptron_model.fit(dataset.input, dataset.labels)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(dataset.input, dataset.labels)


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
            answer = logistic_model.predict(input)[0]
        elif algo == "Perceptron Learning Algorithm":
            answer = perceptron_model.predict(input)[0]
        else:
            answer = decision_tree.predict(input)[0]

        if answer == 0:
            answer = "Người này không đủ khả năng chi trả khoản nợ"
        else:
            answer = "Người này có đủ khả năng chi trả khoản nợ"

        data = {
            'answer': answer
        }
        return JsonResponse(data)

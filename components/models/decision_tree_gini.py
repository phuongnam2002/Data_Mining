import numpy as np
import pandas as pd
from scipy import stats


# Kiểm tra xem 1 cột là số hay chữ
def is_number(column):
    for i in range(0, len(column)):
        if not pd.isnull(column[i]):
            try:
                float(column[i])
                return True
            except ValueError:
                return False


# Kiểm tra xem 1 giá trị này là số hay chữ
def is_number_value(value):
    if not pd.isnull(value):
        try:
            float(value)
            return True
        except ValueError:
            return False


def chi_squared_test(label_df, feature_df):
    label_df.reset_index(drop=True, inplace=True)
    feature_df.reset_index(drop=True, inplace=True)

    data = pd.concat([pd.DataFrame(label_df.values.reshape((label_df.shape[0], 1))), feature_df], axis=1)
    data.columns = ["label", "feature"]
    contigency_table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1], margins=False)

    m = contigency_table.values.sum()
    if m <= 10000 and contigency_table.shape == (2, 2):
        p_value = stats.fisher_exact(contigency_table)
    else:
        p_value = stats.chi2_contingency(contigency_table, correction=False)

    return p_value[1]


def gini_index(features, uniques, targets):
    gini = 0
    weigthed_gini = 0
    denominator_init = features.count()

    data = pd.concat([pd.DataFrame(targets.values.reshape((targets.shape[0], 1))), features], axis=1)

    for word in range(0, len(uniques)):
        denominator = features[(features == uniques[word])].count()
        if denominator[0] > 0:
            for lbl in range(0, len(np.unique(targets))):
                numerator = data.iloc[:, 0][
                    (data.iloc[:, 0] == np.unique(targets)[lbl]) & (data.iloc[:, 1] == uniques[word])].count()

                if numerator > 0:
                    gini = gini + (numerator / denominator) ** 2

        gini = 1 - gini
        weigthed_gini = weigthed_gini + gini * (denominator / denominator_init)
        gini = 0

    return weigthed_gini


def split_numeric_data(feature, split):
    result = pd.DataFrame(feature.values.reshape((feature.shape[0], 1)))
    for fill in range(0, len(feature)):
        result.iloc[fill, 0] = feature.iloc[fill]
    lower = "<=" + str(split)
    upper = ">" + str(split)
    for convert in range(0, len(feature)):
        if float(feature.iloc[convert]) <= float(split):
            result.iloc[convert, 0] = lower
        else:
            result.iloc[convert, 0] = upper
    binary_split = [lower, upper]
    return result, binary_split


def decision_tree_gini(Xdata, ydata, pre_pruning="none", chi_lim=0.1, min_lim=5):
    ################     Phần 1 - Tiền Xử Lý    #############################
    # Tạo Dataframe
    name = ydata.name
    ydata = pd.DataFrame(ydata.values.reshape((ydata.shape[0], 1)))
    for j in range(0, ydata.shape[1]):
        if ydata.iloc[:, j].dropna().value_counts().index.isin([0, 1]).all():
            for i in range(0, ydata.shape[0]):
                if ydata.iloc[i, j] == 0:
                    ydata.iloc[i, j] = "zero"
                else:
                    ydata.iloc[i, j] = "one"
    dataset = pd.concat([ydata, Xdata], axis=1)

    # Xử lý Boolean
    for j in range(0, dataset.shape[1]):
        if dataset.iloc[:, j].dtype == "bool":
            dataset.iloc[:, j] = dataset.iloc[:, j].astype(str)

    # Preprocessing - One Hot Encode
    count = 0
    end_count = dataset.shape[1]
    while (count < end_count - 1):
        count = count + 1
        if is_number(dataset.iloc[:, 1]) == False:
            col_name = dataset.iloc[:, 1].name
            new_col = dataset.iloc[:, 1].unique()
            for k in range(0, len(new_col)):
                one_hot_data = pd.DataFrame({str(col_name) + "[" + str(new_col[k]) + "]": dataset.iloc[:, 1]})
                for L in range(0, one_hot_data.shape[0]):
                    if one_hot_data.iloc[L, 0] == new_col[k]:
                        one_hot_data.iloc[L, 0] = " 1 "
                    else:
                        one_hot_data.iloc[L, 0] = " 0 "
                dataset = pd.concat([dataset, one_hot_data.astype(np.int32)], axis=1)
            dataset.drop(col_name, axis=1, inplace=True)
            end_count = dataset.shape[1]
        else:
            col_name = dataset.iloc[:, 1].name
            one_hot_data = dataset.iloc[:, 1]
            dataset.drop(col_name, axis=1, inplace=True)
            dataset = pd.concat([dataset, one_hot_data], axis=1)

    bin_names = list(dataset)

    # Giá trị nhị phân
    for i in range(0, dataset.shape[0]):
        for j in range(1, dataset.shape[1]):
            if dataset.iloc[:, j].dropna().value_counts().index.isin([0, 1]).all():
                bin_names[j] = "binary"
                if dataset.iloc[i, j] == 0:
                    dataset.iloc[i, j] = str(0)
                else:
                    dataset.iloc[i, j] = str(1)

    # Unique Words List
    unique = []
    uniqueWords = []
    for j in range(0, dataset.shape[1]):
        for i in range(0, dataset.shape[0]):
            token = dataset.iloc[i, j]
            if not token in unique:
                unique.append(token)
        uniqueWords.append(unique)
        unique = []

    # Label Matrix
    label = np.array(uniqueWords[0])
    label = label.reshape(1, len(uniqueWords[0]))

    ################    Phần 2 - Khởi tạo các tham số    #############################
    i = 0
    branch = [None] * 1
    branch[0] = dataset
    gini_vector = np.empty([1, branch[i].shape[1]])
    rule = [None] * 1
    rule[0] = "IF "
    skip_update = False
    stop = 2

    ################     Phần 3 - Thuật toán    #############################
    while (i < stop):
        gini_vector.fill(1)
        for element in range(1, branch[i].shape[1]):
            if len(branch[i]) == 0:
                skip_update = True
                break
            if len(np.unique(branch[i][0])) == 1 or len(branch[i]) == 1:
                if ";" not in rule[i]:
                    rule[i] = rule[i] + " THEN " + name + " = " + branch[i].iloc[0, 0] + ""
                    rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                    if i == 1 and (rule[i].find("{0}") != -1 or rule[i].find("{1}") != -1):
                        rule[i] = rule[i].replace(";", "")
                skip_update = True
                break
            if i > 0 and is_number(dataset.iloc[:, element]) == False and pre_pruning == "chi_2" and chi_squared_test(
                    branch[i].iloc[:, 0], branch[i].iloc[:, element]) > chi_lim:
                if ";" not in rule[i]:
                    rule[i] = rule[i] + " THEN " + name + " = " + branch[i].agg(lambda x: x.value_counts().index[0])[
                        0] + ";"
                    rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                skip_update = True
                continue
            if is_number(dataset.iloc[:, element]) == True and bin_names[element] != "binary":
                gini_vector[0, element] = 1.0
                value = branch[i].iloc[:, element].unique()
                skip_update = False
                for bin_split in range(0, len(value)):
                    bin_sample = split_numeric_data(feature=branch[i].iloc[:, element], split=value[bin_split])
                    if i > 0 and pre_pruning == "chi_2" and chi_squared_test(branch[i].iloc[:, 0],
                                                                             bin_sample[0]) > chi_lim:
                        if ";" not in rule[i]:
                            rule[i] = rule[i] + " THEN " + name + " = " + \
                                      branch[i].agg(lambda x: x.value_counts().index[0])[0] + ";"
                            rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                        skip_update = True
                        continue
                    g_index = gini_index(targets=branch[i].iloc[:, 0], features=bin_sample[0], uniques=bin_sample[1])
                    if g_index < float(gini_vector[0, element]):
                        gini_vector[0, element] = g_index
                        uniqueWords[element] = bin_sample[1]
            if (is_number(dataset.iloc[:, element]) == False or bin_names[element] == "binary"):
                gini_vector[0, element] = 1.0
                skip_update = False
                g_index = gini_index(targets=branch[i].iloc[:, 0], features=pd.DataFrame(
                    branch[i].iloc[:, element].values.reshape((branch[i].iloc[:, element].shape[0], 1))),
                                     uniques=uniqueWords[element])
                gini_vector[0, element] = g_index
            if i > 0 and pre_pruning == "min" and len(branch[i]) <= min_lim:
                if ";" not in rule[i]:
                    rule[i] = rule[i] + " THEN " + name + " = " + branch[i].agg(lambda x: x.value_counts().index[0])[
                        0] + ";"
                    rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                skip_update = True
                continue

        if skip_update == False:
            root_index = np.argmin(gini_vector)
            rule[i] = rule[i] + list(branch[i])[root_index]
            for word in range(0, len(uniqueWords[root_index])):
                uw = str(uniqueWords[root_index][word]).replace("<=", "")
                uw = uw.replace(">", "")
                lower = "<=" + uw
                upper = ">" + uw
                if uniqueWords[root_index][word] == lower and bin_names[root_index] != "binary":
                    branch.append(branch[i][branch[i].iloc[:, root_index] <= float(uw)])
                elif uniqueWords[root_index][word] == upper and bin_names[root_index] != "binary":
                    branch.append(branch[i][branch[i].iloc[:, root_index] > float(uw)])
                else:
                    branch.append(branch[i][branch[i].iloc[:, root_index] == uniqueWords[root_index][word]])
                node = uniqueWords[root_index][word]
                rule.append(rule[i] + " = " + "{" + str(node) + "}")
            for logic_connection in range(1, len(rule)):
                if len(np.unique(branch[i][0])) != 1 and rule[logic_connection].endswith(" AND ") == False and rule[
                    logic_connection].endswith("}") == True:
                    rule[logic_connection] = rule[logic_connection] + " AND "

        skip_update = False
        i = i + 1
        print("iteration: ", i)
        stop = len(rule)

    for i in range(len(rule) - 1, -1, -1):
        if rule[i].endswith(";") == False:
            del rule[i]

    rule.append("Total Number of Rules: " + str(len(rule)))
    rule.append(dataset.agg(lambda x: x.value_counts().index[0])[0])

    return rule


if __name__ == '__main__':
    df = pd.read_csv('data/train/loan_data.csv')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    decision_tree = decision_tree_gini(X, y)

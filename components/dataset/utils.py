def mean(data):
    return sum(data) / len(data)


def mode(data):
    freq = {}
    for x in data:
        freq[x] = freq.get(x, 0) + 1
    max_freq = max(freq.values())
    return [k for k, v in freq.items() if v == max_freq]


def data_range(data):
    return max(data) - min(data)


def symmetry_and_skewness(data):
    meanf = mean(data)
    medianf = median(data)
    modef = mode(data)[0] if mode(data) else None
    if meanf == medianf == modef:
        return "The distribution is symmetric and not skewed."
    elif meanf > medianf:
        return "The distribution is right and positively skewed."
    elif meanf < medianf:
        return "The distribution is left and negatively skewed."


def five_number_summary(data):
    data = sorted(data)
    n = len(data)
    Q1 = median(data[:n // 2])
    Q2 = median(data)
    Q3 = median(data[-(n // 2):])
    return min(data), Q1, Q2, Q3, max(data)


def median(data):
    data = sorted(data)

    n = len(data)

    return (data[n // 2 - 1] + data[n // 2]) / 2 if n % 2 == 0 else data[n // 2]


def iqr_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    threshold = 1.5

    df = df[(df[column] >= Q1 - threshold * IQR) & (df[column] <= Q3 + threshold * IQR)]

    return df

import numpy as np
from tqdm import tqdm
from components.dataset.io import read_csv


class Dataset:
    def __init__(self, file_path):
        self.df = None
        self.input = []
        self.labels = None
        self.file_path = file_path

    def __len__(self):
        return len(self.input)

    def load_data(self):
        self.df = read_csv(self.file_path)

        columns = self.df.columns.values.tolist()

        # Chuyển giá trị categorical về numerical
        for column in tqdm(columns):
            if self.df[column].dtype == 'object':
                seen = set()
                seen_add = seen.add
                numeric_column = [x for x in self.df[column].values.copy().tolist() if not (x in seen or seen_add(x))]

                self.df[column] = self.df[column].apply(lambda x: numeric_column.index(x) + 1)

        # Loại bỏ các hàng bị lặp (nếu có)
        self.df.drop_duplicates(inplace=True)

        # Tách label ra khỏi Dataframe
        columns.remove('not.fully.paid')
        self.labels = self.df['not.fully.paid'].values.copy().tolist()
        self.df.drop(columns='not.fully.paid', inplace=True)

        # Chuyển Dataframe sang trainable dataset
        for id, row in self.df.iterrows():
            datapoint = [1]

            for column in columns:
                datapoint.append(row[column])

            self.input.append(datapoint)

        self.input = np.array(self.input)
        self.labels = np.array(self.labels)

        return

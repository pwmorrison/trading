import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob
from pathlib import Path
import os
from datetime import datetime
import random


class RankingDataset(Dataset):

    def __init__(self, filename, start=0.0, end=1.0, skip_equal_target=True, max_n_examples=None):
        print(filename)
        self.skip_equal_target = skip_equal_target
        self.df = pd.read_csv(filename)

        num_rows = self.df.shape[0]
        self.df = self.df.iloc[int(start * num_rows): int(end * num_rows)]

        if max_n_examples:
            self.df = self.df.iloc[:max_n_examples]

        self.id_col = 'id'
        self.feature_cols = [c for c in self.df if c.startswith("feature")]
        self.target_col = 'target_kazutsugi'

        print(f'Created dataset with {self.df.shape[0]} examples.')

    def __len__(self):
        return self.df.shape[0]

    def read_example(self, index):
        # TODO: extact the features and target separately.
        # TODO: Normalise features.
        # TODO: Make target binary 0, 1.
        row = self.df.iloc[index]
        id = row[self.id_col]
        features = np.array(row[self.feature_cols]).astype(float) - 0.5
        target = row[self.target_col]
        return id, features, target

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        while True:
            indices = random.sample(range(self.df.shape[0]), 2)
            id_1, features_1, target_1 = self.read_example(indices[0])
            id_2, features_2, target_2 = self.read_example(indices[1])
            if not self.skip_equal_target or target_1 != target_2:
                break
            #print(f'Skipping pair with targets {target_1}, {target_2}.')

        target_class = 1 if target_1 >= target_2 else 0

        sample = {
            'features_1': torch.from_numpy(features_1),
            'features_2': torch.from_numpy(features_2),
            'target_1': target_1,
            'target_2': target_2,
            'target_class': target_class,
            'id_1': id_1,
            'id_2': id_2,
        }
        return sample


def read_csv(filename, datetime_format):
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'], format=datetime_format)
    df.set_index('date', inplace=True)
    return df


def create_ranking_dataset(training_filename, start, end, max_n_examples=None):
    dataset = RankingDataset(training_filename, start, end, max_n_examples=max_n_examples)
    return dataset


def test_ranking_dataset():
    training_filename = r'E:\data\numerai\20200714\numerai_datasets\numerai_training_data_small.csv'
    start = 0.0
    end = 0.7
    max_n_examples = 100

    dataset = create_ranking_dataset(training_filename, start, end, max_n_examples=max_n_examples)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for b, batch in enumerate(dataloader):
        features_1 = batch['features_1'].numpy()
        features_2 = batch['features_2'].numpy()
        target_1 = batch['target_1'].numpy()
        target_2 = batch['target_2'].numpy()
        target_class = batch['target_class'].numpy()
        id_1 = batch['id_1']
        id_2 = batch['id_2']
        print(b, features_1.shape, features_1)
        print(b, target_1.shape, target_1)
        print(b, target_class.shape, target_class)
        print(b, len(id_1), id_1)

        break


if __name__ == '__main__':
    # test_ticker_dataset()
    test_ranking_dataset()
    # main()

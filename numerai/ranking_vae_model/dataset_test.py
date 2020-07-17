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


"""
Dataset for testing a model with known synthetic data.
"""

def generate_gradient_data(dim):
    xs = np.linspace(0, 1, dim)
    ys = np.linspace(0, 1, dim)

    gradient = np.zeros((ys.shape[0], xs.shape[0]), dtype=float)
    for y_ind in range(ys.shape[0]):
        y = ys[y_ind]
        for x_ind in range(xs.shape[0]):
            x = xs[x_ind]
            g = (x + y) / 2
            gradient[y_ind, x_ind] = g

    return ys, xs, gradient


class RankingDataset(Dataset):

    def __init__(self, skip_equal_target=True, n_examples=100):
        self.skip_equal_target = skip_equal_target

        # Generate n_examples samples from a gradient image.
        self.ys, self.xs, self.gradient = generate_gradient_data(100)

        y_inds = random.choices(range(self.gradient.shape[0]), k=n_examples)
        x_inds = random.choices(range(self.gradient.shape[1]), k=n_examples)

        self.examples = [[(self.ys[y_ind], self.xs[x_ind]), self.gradient[y_ind, x_ind]]
            for y_ind, x_ind in zip(y_inds, x_inds)]

        print(self.examples)

        print(f'Created dataset with {len(self.examples)} examples.')

    def __len__(self):
        return len(self.examples)

    def read_example(self, index):
        example = self.examples[index]
        features = np.array(example[0]).astype(float) - 0.5
        target = example[1]
        return features, target

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        while True:
            indices = random.sample(range(len(self.examples)), 2)
            features_1, target_1 = self.read_example(indices[0])
            features_2, target_2 = self.read_example(indices[1])
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
            'id_1': indices[0],
            'id_2': indices[1],
        }
        return sample


def create_ranking_dataset(n_examples=None):
    dataset = RankingDataset(n_examples=n_examples)
    return dataset


def test_ranking_dataset_test():
    gradient = generate_gradient_data(100)
    print(gradient)

    dataset = create_ranking_dataset(n_examples=10)
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
    # test_ranking_dataset()
    test_ranking_dataset_test()

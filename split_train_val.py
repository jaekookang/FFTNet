'''
Split dataset into train and validation set

2018-08-30 Jaekoo
'''

import ipdb as pdb
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def read_feature_list(ftxt):
    with open(ftxt, 'r') as f:
        lines = f.readlines()
        X, y = [], []
        for line in lines:
            splitted = line.strip().split('|')
            X.append('|'.join(splitted))
            y.append(splitted[-1])
        X = np.array(X)
        y = np.array(y)
    return X, y


def split_train_val(X, y, test_ratio=0.2):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio)
    sss.get_n_splits(X, y)
    for train_idx, test_idx in sss.split(X, y):
        pass
    train = X[train_idx]
    val = X[test_idx]
    return train, val


if __name__ == '__main__':
    feature_txt = 'features/train.txt'
    X, y = read_feature_list(feature_txt)
    train_data, val_data = split_train_val(X, y)

    # Write to txt
    np.savetxt('features/my_train.txt', train_data, fmt='%s')
    np.savetxt('features/my_val.txt', val_data, fmt='%s')
    print('Done')

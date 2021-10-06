import os
import sys

from scipy.signal.signaltools import resample

sys.path.append('../')

from adles.adles import ADLES
from adles.datasets import Dataset, DataLoader

if __name__ == "__main__":
    data_path = f'../data'
    train_data_path, test_data_path = os.path.join(data_path, 'training_dataset'), os.path.join(data_path, 'testing_dataset')
    
    trainset = Dataset(data_path, 'train', resample=8000)
    trainloader = DataLoader(trainset)

    for (sig, samp_r), label in trainloader:
        ad = ADLES(sig, samp_r, window_size=50, frame_shift=10)
        ad.train(verbose=True, patience=6, n_iters=100)

        print(f"Label: {label}, Alpha: {ad.alpha}, Beta: {ad.beta}, Delta: {ad.delta}")
        break

    
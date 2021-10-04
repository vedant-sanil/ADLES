import os
import sys

sys.path.append('../')

from adles.adles import ADLES
from adles.datasets import Dataset, DataLoader

if __name__ == "__main__":
    data_path = f'../data'
    train_data_path, test_data_path = os.path.join(data_path, 'training_dataset'), os.path.join(data_path, 'testing_dataset')
    
    trainset = Dataset(data_path, 'train')
    trainloader = DataLoader(trainset)

    for (sig, samp_r), label in trainloader:
        ad = ADLES(sig[:10000], samp_r, window_size=300, frame_shift=10)
        ad.integrate()
        ad.plot_phase_portrait()
        ad.solve()
        break

    
import os
import sys

sys.path.append('../')

from adles.adles import ADLES
from adles.datasets import Dataset, DataLoader
 
if __name__ == "__main__":
    window, frame = 0.05, 0.025
    data_path = f'../data'
    train_data_path, test_data_path = os.path.join(data_path, 'training_dataset'), os.path.join(data_path, 'testing_dataset')
    
    trainset = Dataset(data_path, 'train', resample=8000)
    trainloader = DataLoader(trainset)

    for (sig, samp_r), label in trainloader:
        window_size, frame_shift = int(window*samp_r), int(frame*samp_r)
        ad = ADLES(sig, samp_r, window_size=window_size, frame_shift=frame_shift)
        ad.train(verbose=True, patience=6, n_iters=100)

        print(f"Label: {label}, Alpha: {ad.alpha}, Beta: {ad.beta}, Delta: {ad.delta}")
        break

    
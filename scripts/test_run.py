import os
import sys
import csv

sys.path.append('../')

from adles.adles import ADLES
from adles.datasets import Dataset, DataLoader
 
if __name__ == "__main__":
    window, frame = 0.05, 0.025
    init_left_distend, init_right_distend = 0.1, 0.15
    data_path = f'../data'
    train_data_path, test_data_path = os.path.join(data_path, 'training_dataset'), os.path.join(data_path, 'testing_dataset')
    
    trainset = Dataset(data_path, 'train', resample=8000)
    trainloader = DataLoader(trainset, shuffle_seed=40)

    res_csv = open(f'../results/results.csv', 'w')
    err_csv = open(f'../results/errors.csv', 'w')
    writer = csv.writer(res_csv)
    err_writer = csv.writer(err_csv)

    writer.writerow(['Label', 'Alpha', 'Beta', 'Delta', 'Cr', 'Cl', 'Window Size', 'Frame Shift'])
    err_writer.writerow(['Label', 'Cr', 'Cl', 'Window Size', 'Frame Shift'])

    for (sig, samp_r), label in trainloader:
        print(label)
        try:
            window_size, frame_shift = int(window*samp_r), int(frame*samp_r)
            ad = ADLES(sig, samp_r, window_size=window_size, frame_shift=frame_shift,
                        init_left_distend=init_left_distend, init_right_distend=init_right_distend)

            # Train the ADLES model
            ad.train(verbose=True, patience=6, n_iters=35, step_size=100)

            #print(f"Label: {label}, Alpha: {ad.alpha}, Beta: {ad.beta}, Delta: {ad.delta}")
            writer.writerow([label, ad.alpha, ad.beta, ad.delta, init_right_distend, init_left_distend, window_size, frame_shift])
            res_csv.flush()
        except:
            err_writer.writerow([label, init_right_distend, init_left_distend, window_size, frame_shift])
            err_csv.flush()

    res_csv.close()
    err_csv.close()

    
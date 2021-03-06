import os
import sys
import random

import librosa
from librosa.core.audio import resample

def load_single_path(path, resample=None):
    y ,sr = librosa.load(path)
    if resample and isinstance(resample, int):
        y = librosa.resample(y, sr, resample)
    return y, resample

class Dataset():
    def __init__(self, data_path, data_type, resample=None):
        self.resample = resample
        self.files = []
        self.labels = []
        if data_type == 'train':
            types = ['Normal', 'Pathological']
            for tp in types:
                if tp == 'Normal':
                    #self.train_files.append(os.path.join(data_path, 'training_dataset', tp))
                    train_path = os.path.join(data_path, 'training_dataset', tp)
                    for f in os.listdir(train_path):
                        if f.endswith('wav'):
                            self.files.append(os.path.join(train_path, f))
                            self.labels.append('Normal')
                elif tp == 'Pathological':
                    patho_dir = os.path.join(data_path, 'training_dataset', tp)
                    for patho in os.listdir(patho_dir):
                        if patho in ['Neoplasm', 'Phonotrauma', 'Vocal palsy']: 
                            sp_patho_dir = os.path.join(patho_dir, patho)
                            for f in os.listdir(sp_patho_dir):
                                if f.endswith('wav'):
                                    self.files.append(os.path.join(sp_patho_dir, f))
                                    self.labels.append(patho)

        # TODO: Implement shuffle here
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load_single_path(self.files[idx], self.resample), self.labels[idx]

def DataLoader(dataset, shuffle_seed=None):
    idxs = list(range(len(dataset)))
    if shuffle_seed:
        random.Random(shuffle_seed).shuffle(idxs)
    for i in idxs:
        yield dataset[i]        
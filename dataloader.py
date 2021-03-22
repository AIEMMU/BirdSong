import librosa
import numpy as np
import torchaudio, torch
import librosa
import os.path as osp
from torch.utils.data import DataLoader

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def normalize(image, mean=None, std=None):
    image = image / 255.0
    if mean is not None and std is not None:
        image = (image - mean) / std
    return image.astype(np.float32)

class Dataset:
    def __init__(self, tp, config=None,
                 mode='test'):
        #tp is the csv file that is loaded in

        self.tp = tp
        #config contains the data such as duration, window lengths, sample rate and other stuff
        self.config = config
        self.sr = self.config.sr
        self.duration = self.config.duration
        self.mode = mode
        #configurations for generating mel spectrograms
        self.nmels = self.config.nmels
        self.fmin, self.fmax = 84, self.sr//2
        self.sliding_window = self.config.sliding_window
        self.mode = mode
        #resamples the audio to a sample rate set in the config
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=44_000, new_freq=self.sr)
        #num classes
        self.num_classes = self.config.num_classes
        #number of splits in the audio files


    def __len__(self):
        return len(self.tp.recording_id.unique())

    def __getitem__(self, idx):
        #get the audio file and possible labels associated with the file
        recording_id = self.tp.loc[idx, 'recording_id']
        df = self.tp.loc[self.tp.recording_id == recording_id]
        #load the audio file and create the path
        audio_fn = osp.join(self.config.data_root, f"{recording_id}.wav")
        y, sr = librosa.load(audio_fn, sr=None,
                            duration=self.config.total_duration)
        segments = len(y) / (self.config.sliding_window*sr)
        segments = int(np.ceil(segments))
        y_stacked = []
        length = int(self.config.duration * sr)

        for i in range(0,segments):
            if (i + 1) * length > len(y):
                y_ = y[len(y) - length:len(y)]
            else:
                y_ = y[i * length:(i + 1) * length]
            y_stacked.append(y_)
        #create mel spectrograms
        y_stacked  = np.array(y_stacked)
        melspec_stacked = []
        for y in y_stacked:
            y = self.resampler(torch.from_numpy(y).float()).numpy()
            melspec = librosa.feature.melspectrogram(
                y, sr=self.sr, n_mels=self.nmels, fmin=self.fmin, fmax=self.fmax,
            )
            #Convert a dB-scale spectrogram to a power spectrogram.
            melspec = librosa.power_to_db(melspec)
            #change to 3 channel image and prepare to be used as a tensor
            melspec = mono_to_color(melspec)
            melspec = normalize(melspec, mean=None, std=None)
            melspec = np.moveaxis(melspec, 2, 0)
            melspec_stacked.append(melspec)
        
        melspec_stacked = np.stack(melspec_stacked)
        #if there are labels, the melspecs and labels are returned as 1 hot encoded vectors
        if self.mode == 'val':
            species = df.loc[:, 'species_id'].unique()
            species = species[species != -1];
            labels = np.zeros((self.num_classes,))
            np.put(labels, species, 1)

            return melspec_stacked, labels
        else:
            return melspec_stacked

def get_dataloader(tp, config, mode):
    dataset = Dataset(tp, config=config, mode=mode)
    data_loader = DataLoader(dataset, batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             shuffle=False, drop_last=False)
    return data_loader


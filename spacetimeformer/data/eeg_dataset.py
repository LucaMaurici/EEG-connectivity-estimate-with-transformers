import random
from typing import List
import os
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

import pickle as pkl

import spacetimeformer as stf

import sys
#sys.path.insert(0,"J:/Il mio Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer")
sys.path.insert(0,"C:/Users/lucam/Google Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer")
import config_file as cf
#cf.reset()


class EEGTimeSeries:
    def __init__(
        self,
        data_path: str = None,
        raw_data: List = None,
        val_split: float = 0.30,  # mod 0.15,
        test_split: float = 0.0, #0.04,  # mod 0.15,
        time_features_dim: int = cf.read("x_dim")  #4  # mod 6
    ):

        assert data_path is not None or raw_data is not None

        if raw_data is None:
            self.data_path = data_path
            assert os.path.exists(self.data_path)
            with open(self.data_path, 'rb') as handle:
                raw_data = np.array(pkl.load(handle))

        samples = np.shape(raw_data)[1]
        #self.num_samples = samples
        self.time_features = stf.data.eeg_timefeatures.eeg_time_features(samples, time_features_dim)
        #print(f"\n\n---------------self.time_cols {self.time_cols}---------------------\n\n")

        # Train/Val/Test Split using holdout approach #
        trials = np.shape(raw_data)[0]
        train_cutoff = round(trials*(1-val_split-test_split))
        val_cutoff = round(trials*(1-test_split))

        trials_idxs = [*range(trials)]
        random.seed(0)
        random.shuffle(trials_idxs)

        train_idxs = trials_idxs[:train_cutoff]
        val_idxs = trials_idxs[train_cutoff:val_cutoff]
        #test_idxs = trials_idxs[val_cutoff:]
        test_idxs = trials_idxs

        #BATCH_SIZE = 128
        #print(f"len(train_idxs): {len(train_idxs)}")
        #train_idxs = train_idxs[:int(len(train_idxs)/BATCH_SIZE)]
        #val_idxs = val_idxs[:int(len(val_idxs)/BATCH_SIZE)]
        #test_idxs = test_idxs[:int(len(test_idxs)/BATCH_SIZE)]

        self._train_data_unscaled = raw_data[train_idxs]
        #print(f"self._train_data_unscaled.shape: {self._train_data_unscaled.shape}")
        self._val_data_unscaled = raw_data[val_idxs]
        #print(f"self._val_data_unscaled.shape: {self._val_data_unscaled.shape}")
        #self._test_data_unscaled = raw_data[test_idxs]
        self._test_data_unscaled = raw_data[test_idxs]  # raw_data[:] # mod
        #print(f"self._test_data_unscaled.shape: {self._test_data_unscaled.shape}")


        # Plotting dataset sample before scaling
        print(raw_data[test_idxs].shape)
        yc = raw_data[test_idxs]
        #Statistical properties mod
        print(f"yc.shape: {yc.shape}")
        mean_yc = np.mean(yc, axis=(0,1))
        std_dev_yc = np.std(yc, axis=(0,1))
        print("\n\n-----------Statistical properties before scaling------------")
        print(f"mean yc: {mean_yc}")
        print(f"std_dev yc: {std_dev_yc}")
        print()

        # WITHOUT SCALING
        self._train_data = raw_data[train_idxs]
        self._val_data = raw_data[val_idxs]
        self._test_data = raw_data[test_idxs]
        
        self._scaler_train = StandardScaler()
        self._scaler_val = StandardScaler()
        self._scaler_test = StandardScaler()

        num_instances, num_time_steps, num_features = np.shape(self._train_data)
        self._train_data = np.reshape(self._train_data, newshape=(-1, num_features))
        self._scaler_train = self._scaler_train.fit(self._train_data)
        print(f"\nself._scaler_train.scale_: {self._scaler_train.scale_}\n")
        print(f"\nself._scaler_train.mean_: {self._scaler_train.mean_}\n")
        self._train_data = np.reshape(self._train_data, newshape=(num_instances, num_time_steps, num_features))

        num_instances, num_time_steps, num_features = np.shape(self._val_data)
        self._val_data = np.reshape(self._val_data, newshape=(-1, num_features))
        self._scaler_val = self._scaler_val.fit(self._val_data)
        print(f"\nself._scaler_val.scale_: {self._scaler_val.scale_}\n")
        print(f"\nself._scaler_val.mean_: {self._scaler_val.mean_}\n")
        self._val_data = np.reshape(self._val_data, newshape=(num_instances, num_time_steps, num_features))

        num_instances, num_time_steps, num_features = np.shape(self._test_data)
        self._test_data = np.reshape(self._test_data, newshape=(-1, num_features))
        self._scaler_test = self._scaler_test.fit(self._test_data)
        print(f"\nself._scaler_test.scale_: {self._scaler_test.scale_}\n")
        print(f"\nself._scaler_test.mean_: {self._scaler_test.mean_}\n")
        self._test_data = np.reshape(self._test_data, newshape=(num_instances, num_time_steps, num_features))

        
        self._train_data, self._scaler_train = self.apply_scaling(raw_data[train_idxs])
        print(f"\n\n---------------self._train_data {self._train_data}---------------------\n\n")
        self._val_data, self._scaler_val = self.apply_scaling(raw_data[val_idxs])
        print(f"\n\n---------------self._val_data {self._val_data}---------------------\n\n")
        self._test_data, self._scaler_test = self.apply_scaling(raw_data[test_idxs])
        print(f"\n\n---------------self._test_data {self._test_data}---------------------\n\n")
        

        print(self._train_data.shape)
        print(self._val_data.shape)
        print(self._test_data.shape)

        # Plotting dataset sample after scaling
        yc = self._test_data
        #Statistical properties mod
        print(f"yc.shape: {yc.shape}")
        mean_yc = np.mean(yc, axis=(0,1))
        std_dev_yc = np.std(yc, axis=(0,1))
        print("\n\n-----------Statistical properties after scaling------------")
        print(f"mean yc: {mean_yc}")
        print(f"std_dev yc: {std_dev_yc}")
        print()

        

    def get_slice(self, split, trial, start, stop, skip):
        assert split in ["train", "val", "test"]
        if split == "train":
            return self._train_data[trial, start:stop:skip], self._train_data_unscaled[trial, start:stop:skip]
        elif split == "val":
            return self._val_data[trial, start:stop:skip], self._val_data_unscaled[trial, start:stop:skip]
        else:
            return self._test_data[trial, start:stop:skip], self._test_data_unscaled[trial, start:stop:skip]


    def apply_scaling(self, array):
        print(np.shape(array))
        num_instances, num_time_steps, num_features = np.shape(array)
        array = np.reshape(array, newshape=(-1, num_instances*num_features))
        print(np.shape(array))
        scaler = StandardScaler()
        array = scaler.fit_transform(array)
        print(scaler.scale_)
        array = np.reshape(array, newshape=(num_instances, num_time_steps, num_features))
        return array, scaler

    def apply_scaling_with_data(self, array, scaler):
        print(np.shape(array))
        num_instances, num_time_steps, num_features = np.shape(array)
        array = np.reshape(array, newshape=(-1, num_instances*num_features))
        print(np.shape(array))
        array = scaler.transform(array)
        array = np.reshape(array, newshape=(num_instances, num_time_steps, num_features))
        return array

    def reverse_scaling(self, array, scaler=None):
        if scaler == None:
            scaler = self._scaler_train
        array_type = type(array)
        if array_type.__module__ != np.__name__:  # not numpy
            #array = torch.from_numpy(array).float()
            #array = array.to(torch.cuda.current_device()).float()
            array = torch.Tensor.cpu(array)
        num_instances, num_time_steps, num_features = np.shape(array)
        array = np.reshape(array, newshape=(-1, num_instances*num_features))
        new_array = scaler.inverse_transform(array)
        new_array = np.reshape(new_array, newshape=(num_instances, num_time_steps, num_features))

        if array_type.__module__ != np.__name__:
            new_array = torch.from_numpy(new_array).float()
            new_array = new_array.to(torch.cuda.current_device()).float()
        return new_array

    '''
    def apply_scaling_eeg(self, array):
        num_instances, num_time_steps, num_features = np.shape(array)
        array = np.reshape(array, newshape=(-1, num_features))
        array = self._scaler.transform(array)
        array = np.reshape(array, newshape=(num_instances, num_time_steps, num_features))
        return array

    def reverse_scaling_eeg(self, array):
        num_instances, num_time_steps, num_features = np.shape(array)
        array = np.reshape(array, newshape=(-1, num_features))
        array = self._scaler.inverse_transform(array)
        array = np.reshape(array, newshape=(num_instances, num_time_steps, num_features))
        return array
    '''

    '''
    def apply_scaling_old(self, array):
        dim = array.shape[-1]
        return (array - self._scaler.mean_[:dim]) / self._scaler.scale_[:dim]

    def apply_scaling_df_old(self, df):
        scaled = df.copy(deep=True)
        # scaled[self.target_cols] = self._scaler.transform(df[self.target_cols].values)
        cols = self.target_cols + self.exo_cols
        dtype = df[cols].values.dtype
        scaled[cols] = (
            df[cols].values - self._scaler.mean_.astype(dtype)
        ) / self._scaler.scale_.astype(dtype)
        return scaled

    def reverse_scaling_df_old(self, df):
        scaled = df.copy(deep=True)
        # scaled[self.target_cols] = self._scaler.inverse_transform(df[self.target_cols].values)
        cols = self.target_cols + self.exo_cols
        dtype = df[cols].values.dtype
        scaled[cols] = (
            df[cols].values * self._scaler.scale_.astype(dtype)
        ) + self._scaler.mean_.astype(dtype)
        return scaled

    def reverse_scaling_old(self, array):
        # self._scaler is fit for target_cols + exo_cols
        # if the array dim is less than this length we start
        # slicing from the target cols
        dim = array.shape[-1]
        print(f"\n\n----------\n self._scaler.scale_[:dim] {self._scaler.scale_[:dim]}")
        print(f"self._scaler.mean_[:dim] {self._scaler.mean_[:dim]} \n----------------------\n\n")
        return (array * self._scaler.scale_[:dim]) + self._scaler.mean_[:dim]
        # return self._scaler.inverse_transform(array)
    '''

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def test_data(self):
        return self._test_data

    def length(self, split):
        return {
            "train": np.shape(self.train_data)[1],
            "val": np.shape(self.val_data)[1],
            "test": np.shape(self.test_data)[1],
        }[split]

    def num_trials(self, split):
        return {
            "train": np.shape(self.train_data)[0],
            "val": np.shape(self.val_data)[0],
            "test": np.shape(self.test_data)[0],
        }[split]

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str, default="auto")


class EEGTorchDset(Dataset):
    def __init__(
        self,
        eeg_time_series: EEGTimeSeries,
        split: str = "train",
        context_points: int = 12,
        target_points: int = 1,
        time_resolution: int = 1,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.series = eeg_time_series
        self.context_points = context_points
        self.target_points = target_points
        self.time_resolution = time_resolution

        print(f"self.series.length(split) {self.series.length(split)}")
        print(f"+ time_resolution * (-target_points - context_points)+ 1: {time_resolution * (-target_points - context_points)+ 1}")

        self._slice_start_points = [
            i
            for i in range(
                0,
                self.series.length(split)
                + time_resolution * (-target_points - context_points)
                + 1,
            )
        ]
        random.shuffle(self._slice_start_points)
        #self._slice_start_points = self._slice_start_points
        print(f"self._slice_start_points: {self._slice_start_points}")
        print(f"self.series.num_trials(split): {self.series.num_trials(split)}")

        self._trials_idxs = [i for i in range(0, self.series.num_trials(split))]
        random.shuffle(self._trials_idxs)

    def __len__(self):
        return len(self._slice_start_points) * self.series.num_trials(self.split)

    def _torch(self, *data):
        return tuple(torch.from_numpy(x).float() for x in data)

    def __getitem__(self, i):
        start = self._slice_start_points[i%len(self._slice_start_points)]
        trial_idx = self._trials_idxs[int(i/len(self._slice_start_points))]
        #print(f"start: {start}")
        #print(f"trial_idx: {trial_idx}")
        #print(f"i {i}")
        #print(f"len(self._slice_start_points) {len(self._slice_start_points)}")
        #print(f"len(self._trials_idxs) {len(self._trials_idxs)}")
        series_slice, series_slice_unscaled = self.series.get_slice(
            self.split,
            trial=trial_idx,
            start=start,
            stop=start
            + self.time_resolution * (self.context_points + self.target_points),
            skip=self.time_resolution,
        )
        time_slice = self.series.time_features[start: \
            start+ self.time_resolution * (self.context_points + self.target_points): \
            self.time_resolution]

        ctxt_x = time_slice[: self.context_points]
        trgt_x = time_slice[self.context_points :]
        ctxt_y = series_slice[: self.context_points]
        trgt_y = series_slice[self.context_points :]

        ctxt_y_unscaled = series_slice_unscaled[: self.context_points]
        trgt_y_unscaled = series_slice_unscaled[self.context_points :]

        return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y), self._torch(ctxt_x, ctxt_y_unscaled, trgt_x, trgt_y_unscaled)

    @classmethod
    def add_cli(self, parser):
        parser.add_argument(
            "--context_points",
            type=int,
            default = 10,  # real parameter
            help="number of previous timesteps given to the model in order to make predictions",
        )
        parser.add_argument(
            "--target_points",
            type=int,
            default = 1,
            help="number of future timesteps to predict",
        )
        parser.add_argument(
            "--time_resolution",
            type=int,
            default=1,
        )


if __name__ == "__main__":
    test = CSVTimeSeries(
        "/p/qdatatext/jcg6dn/asos/temperature-v1.csv",
        ["ABI", "AMA", "ACT", "ALB", "JFK", "LGA"],
    )
    breakpoint()
    dset = CSVTorchDset(test)
    base = dset[0][0]
    for i in range(len(dset)):
        assert base.shape == dset[i][0].shape
    breakpoint()

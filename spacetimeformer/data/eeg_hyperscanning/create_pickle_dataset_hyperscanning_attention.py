import scipy.io
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
import scipy.signal
import random
import sys
#sys.path.insert(0,"J:/Il mio Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer")
sys.path.insert(0,"C:/Users/lucam/Google Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer")
import config_file as cf
#from spacetimeformer.spacetimeformer_model.utils.general_utils import *

def save_dset_per_condition(trials, condition_str):
    print(f"_____{condition_str}______")
    if not os.path.exists(f"./{RUN_NICKNAME}"):
        os.makedirs(f"./{RUN_NICKNAME}")
    with open(f"./{RUN_NICKNAME}/ch{NUM_CHANNELS}_sub_{SUBJECT_ID}_cond_{condition_str}.pkl", 'wb') as file:
        pkl.dump(trials, file)

RUN_NICKNAME = cf.read("run_nickname")
#list_of_files = get_list_of_files(f'./plots_checkpoints_logs/{args.run_name}/checkpoints/{channel}/')
SUBJECTS_FOLDER = 'C:/Users/lucam/Google Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer/data/eeg_hyperscanning/EEG data couples'
subjects = os.listdir(SUBJECTS_FOLDER)
conditions = ['Solo', 'Joint']


for i, SUBJECT_ID in enumerate(subjects):
    if 'desktop.ini' in SUBJECT_ID:
        continue
    for condition in conditions:
    
        #SINGLE_SUBJECT = os.path.join(SUBJECTS_FOLDER, "sub-"+SUBJECT_ID+"/ProcessedData/data_ica.mat")
        SINGLE_SUBJECT = f"{SUBJECTS_FOLDER}/{SUBJECT_ID}/{SUBJECT_ID}_{condition}Condition1.mat"
        print(f"SUBJECT_ID: {SUBJECT_ID}")

        mat = scipy.io.loadmat(SINGLE_SUBJECT)
        samples_mat = mat['EEG'][0]['samp'][0]
        print(samples_mat.shape)
        labels_mat = mat['EEG'][0]['label'][0]

        NUM_CHANNELS = len(np.squeeze(labels_mat))
        print(f"NUM_CHANNELS: {NUM_CHANNELS}")

        samples_mat = np.squeeze(samples_mat)
        #samples_mat.shape: samples, channels, trials
        trials = np.transpose(samples_mat, [2,0,1])
        #trials.shape: trials, samples, channels

        #plt.plot(trials[30,:,:])
        #plt.show()

        trials = scipy.signal.decimate(trials, q=2, axis=1)
        trials = np.gradient(trials, axis=1)
        print(np.shape(trials))

        plt.plot(trials[30,:,:])
        plt.show()

        save_dset_per_condition(trials, condition.lower())
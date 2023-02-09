import scipy.io
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.preprocessing import StandardScaler

def apply_scaling(array):
    print(np.shape(array))
    num_instances, num_time_steps, num_features = np.shape(array)
    array = np.reshape(array, newshape=(-1, num_instances*num_features))
    print(np.shape(array))
    scaler = StandardScaler()
    array = scaler.fit_transform(array)
    print(scaler.scale_)
    array = np.reshape(array, newshape=(num_instances, num_time_steps, num_features))
    return array

#SUBJECT_ID = "SNR_10000_5"
NUM_CHANNELS = 4

#channels = [0, 1, 2, 30, 31, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 46, 22, 23, 24, 49, 26, 27, 28]

#SUBJECTS_FOLDER = 'J:/Il mio Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer/data/Generated EEG'
SUBJECTS_FOLDER = "noise_test_standardized_1"

#SNRs = ['inf', '10000', '100', '3']
SNRs = ['inf']
REPETITIONS = 1
for snr in SNRs:
    for rep in range(1, REPETITIONS+1):
        SUBJECT_ID = f"SNR_{snr}_{rep}"

        SINGLE_SUBJECT = os.path.join(SUBJECTS_FOLDER, f"GeneratedEEG 4chs_{SUBJECT_ID}.mat")

        mat = scipy.io.loadmat(SINGLE_SUBJECT)
        struct = mat['EEG'][0][0]
        eeg = struct[0]
        noise = struct[1]
        model = struct[2]
        delay_mat = struct[3]
        flag_out = struct[4]
        eeg_clean = eeg - noise
        print(np.shape(eeg))
        print(np.shape(noise))
        print(np.shape(model))
        print(np.shape(delay_mat))
        print(np.shape(flag_out))
        print(np.shape(eeg_clean))
        print(eeg_clean)

        # eeg.shape: samples, channels, trials
        eeg = np.transpose(eeg, [2, 0, 1])
        # eeg.shape: trials, samples, channels
        trials = apply_scaling(eeg)

        print(np.shape(trials))
        plt.plot(trials[0,:,:])
        plt.show()

        #with open(f"./{SUBJECTS_FOLDER}/eeg_generated_ch{NUM_CHANNELS}_sub_"+SUBJECT_ID+".pkl", 'wb') as file:
            #pkl.dump(trials, file)

        #trials = scipy.signal.decimate(trials, q=4, axis=1)
        #trials = np.gradient(trials, axis=1)

        for channel in range(NUM_CHANNELS):
            new_trials = np.delete(trials, channel, axis=2)

            print(np.shape(new_trials))
            #plt.plot(new_trials[0,:,:])
            #plt.show()

            #with open(f"./{SUBJECTS_FOLDER}/eeg_generated_ch{channel}_sub_"+SUBJECT_ID+".pkl", 'wb') as file:
                #pkl.dump(new_trials, file)

import scipy.io
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
import scipy.signal

SUBJECT_ID = "1"

#channels = [0, 1, 2, 30, 31, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 46, 22, 23, 24, 49, 26, 27, 28]

#SUBJECTS_FOLDER = 'J:/Il mio Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer/data/Generated EEG'
SUBJECTS_FOLDER = ""
SINGLE_SUBJECT = os.path.join(SUBJECTS_FOLDER, "GeneratedEEG 4chs_SNRinf.mat")

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

'''
trials = []
#eeg = np.transpose(eeg, [2, 1, 0])
for i, t in enumerate(eeg_clean):
	trials.append(t[:, :])
'''

trials = np.transpose(eeg, [2, 0, 1])
#new_shape(157, 200, 29)
#new_shape(trials, samples, channels)
#trials = np.transpose(trials, [0, 2, 1])
print(np.shape(trials))

plt.plot(trials[0,:,:])
plt.show()

#trials = scipy.signal.decimate(trials, q=4, axis=1)
#trials = np.gradient(trials, axis=1)
print(np.shape(trials))

plt.plot(trials[0,:,:])
plt.show()

with open("./eeg_generated_SNRinf_4chs_sub_"+SUBJECT_ID+".pkl", 'wb') as file:
    pkl.dump(trials, file)
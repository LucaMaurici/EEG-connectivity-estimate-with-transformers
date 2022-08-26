import scipy.io
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt

SUBJECT_ID = "20"

SUBJECTS_FOLDER = 'E:/Datasets/Social memory cuing full dataset/derivatives/EEGPreprocessedDataTableStudy'
SINGLE_SUBJECT = os.path.join(SUBJECTS_FOLDER, "sub-"+SUBJECT_ID+"/ProcessedData/data_ica.mat")

mat = scipy.io.loadmat(SINGLE_SUBJECT)
trial = mat['trial']

trials = []
trial = np.transpose(np.squeeze(trial))
for i, t in enumerate(trial):
	trials.append(t[ :5, 1250:1750])

trials = np.transpose(trials, [0, 2, 1])
print(np.shape(trials))

plt.plot(trials[0,:,:])
plt.show()

with open("./eeg_social_memory_sub_"+SUBJECT_ID+".pkl", 'wb') as file:
    pkl.dump(trials, file)
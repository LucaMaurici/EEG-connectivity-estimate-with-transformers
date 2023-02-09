import numpy as np
import math

import matplotlib.pyplot as plt

def eeg_time_features(samples, dim):

    time_features = np.zeros((dim, samples))

    for feature_idx in range(dim):
        time_features[feature_idx,:] = [math.sin((i/(samples-1))*0.5*math.pi*(2**feature_idx)) for i in range(samples)]
    
    return np.transpose(time_features, [1, 0])

'''
time = eeg_time_features(250, 6)
plt.plot(time)
plt.show()
'''

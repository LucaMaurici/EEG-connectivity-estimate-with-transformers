import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pickle

def mask_matrix(cov_matrix):
    samples = np.shape(cov_matrix)[0]
    channels = np.shape(cov_matrix)[2]
    for sample in range(samples-1):
        cov_matrix[sample, sample+1:samples, :, :] = 0
    return cov_matrix

def generate_cov_matrix(samples, channels):
    cov_matrix = np.random.rand(samples, samples, channels, channels)-0.5
    cov_matrix = mask_matrix(cov_matrix)
    return cov_matrix

def generate_value(sample, channel, cov_matrix, val):
    #print(np.squeeze(np.squeeze(np.asarray(cov_matrix[sample, :, channel, :]))))
    #print(np.shape(np.asarray(cov_matrix[sample, :, channel, :])))
    return (val + 0.025*float(np.sum(np.asarray(cov_matrix[sample, :, channel, :]))))*50

trials = 224
samples = 500
channels = 29

cov_matrix = generate_cov_matrix(31, channels)
dataset = np.zeros([trials,samples,channels])

for trial in range(trials):
    for channel in range(channels):
        val = 0
        for sample in range(samples):
            val += (random.random()-0.5)*0.1 - val*0.07
            dataset[trial,sample,channel] = generate_value(sample%31, channel, cov_matrix, val)

print(f"np.shape(dataset) {np.shape(dataset)}")

plt.plot(dataset[0,:,0:3])
plt.show()

dataset = dataset

print(dataset)

print(f"type(dataset) {type(dataset)}")
print(f"np.shape(dataset) {np.shape(dataset)}")

with open("./toy_eeg.pkl", 'wb') as file:
    pickle.dump(dataset, file)
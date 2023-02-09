import pickle
import numpy as np
import os
from spacetimeformer.spacetimeformer_model.utils.general_utils import  *
import cv2
import torch
import spacetimeformer as stf
import scipy.io
import config_file as cf

threshold = 0.02

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def threshold_median_plus(data):
    median = np.median(data) + 0.1
    print(f"median: {median}")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] < median:
                data[i,j] = 0
    return data

def threshold_num(data):
    print(f"threshold: {threshold}")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] < threshold:
                data[i,j] = 0
    return data

def matrix_loss(matrix1, matrix2):
    return np.sum(np.absolute(matrix1 - matrix2)**2)

cf.reset()
NUM_CHANNELS = cf.read('num_channels')
NUM_CHANNELS_GT = 4

run_name, run_type, run_id = read_config_file()
print(f"run_name: {run_name}")

granger_test_folder = f"./plots_checkpoints_logs/{run_name}/plots/{run_id}/granger_test_val"

'''
print("LOADING PDC...")
mat = scipy.io.loadmat(f"./data/Generated EEG/PDC/PDC_GeneratedEEG_{NUM_CHANNELS_GT}chs_PoptComputed.mat")
struct = mat['APP'][0][0]
PDC = struct[0]
PDCtr = struct[1]
PDCpat = struct[2]

PDC = np.mean(PDC, axis=2)
PDC = np.nan_to_num(PDC)
mask = np.ones((NUM_CHANNELS_GT, NUM_CHANNELS_GT))
np.fill_diagonal(mask, 0.0)
PDC = PDC * mask
PDC = normalize_data(PDC)

PDCtr = np.mean(PDCtr, axis=2)
PDCtr = np.nan_to_num(PDCtr)
mask = np.ones((NUM_CHANNELS_GT, NUM_CHANNELS_GT))
np.fill_diagonal(mask, 0.0)
PDCtr = PDCtr * mask
PDCtr = normalize_data(PDCtr)

'''

print("LOADING mean_granger_matrix...")
with open(f"{granger_test_folder}/mean_granger_matrix.pkl", "rb") as file:
    mean_granger_matrix = pickle.load(file)

#mean_granger_matrix_normalized = normalize_data(mean_granger_matrix)
mean_granger_matrix_normalized = mean_granger_matrix
mask = np.ones((NUM_CHANNELS, NUM_CHANNELS))
np.fill_diagonal(mask, 0.0)
mean_granger_matrix = mean_granger_matrix_normalized * mask
mean_granger_matrix_thresholded = threshold_num(mean_granger_matrix)

print("LOADING variance_granger_matrix...")
with open(f"{granger_test_folder}/variance_granger_matrix.pkl", "rb") as file:
    variance_granger_matrix = pickle.load(file)

variance_granger_matrix_normalized = normalize_data(variance_granger_matrix)
mask = np.ones((NUM_CHANNELS, NUM_CHANNELS))
np.fill_diagonal(mask, 0.0)
variance_granger_matrix = variance_granger_matrix_normalized * mask
variance_granger_matrix_thresholded = threshold_num(variance_granger_matrix)

'''
print("LOADING Ground Truth...")
mat = scipy.io.loadmat(f"./data/Generated EEG/GeneratedEEG {NUM_CHANNELS_GT}chs.mat")
struct = mat['EEG'][0][0]
eeg = struct[0]
noise = struct[1]
model = struct[2]

model = np.mean(model, axis = -1)
model = np.absolute(model)
model = normalize_data(model)
'''
if not os.path.exists(f"{granger_test_folder}/evaluation/"):
    os.makedirs(f"{granger_test_folder}/evaluation/")

plotting = stf.plot.AttentionMatrixCallback([])

'''
pdc_img = stf.plot.show_image(PDC, "PDC")
cv2.imwrite(f"{granger_test_folder}/evaluation/pdc.png", pdc_img)
plotting.save_image(matrix=PDC, title="PDC", path=f"{granger_test_folder}/evaluation/pdc_by.png")
print("PDC: ", matrix_loss(PDC, model))

pdc_thresholded_img = stf.plot.show_image(PDCtr, "PDC thresholded")
#cv2.imwrite(f"{granger_test_folder}/evaluation/pdc_thresholded.png", pdc_thresholded_img)
plotting.save_image(matrix=PDCtr, title="PDC thresholded", path=f"{granger_test_folder}/evaluation/pdc_thresholded_by.png")
print("PDC_thresholded: ", matrix_loss(PDCtr, model))
'''

mean_granger_matrix_normalized_img = stf.plot.show_image(mean_granger_matrix_normalized, "Mean Granger matrix")
#cv2.imwrite(f"{granger_test_folder}/evaluation/mean_granger_matrix_normalized.png", mean_granger_matrix_normalized_img)
plotting.save_image(matrix=mean_granger_matrix_normalized, title="Mean Granger matrix", path=f"{granger_test_folder}/evaluation/mean_granger_matrix_normalized_by.png")
#print("mean_granger_matrix_normalized: ", matrix_loss(mean_granger_matrix_normalized, model))

variance_granger_matrix_normalized_img = stf.plot.show_image(variance_granger_matrix_normalized, "Variance Granger matrix")
#cv2.imwrite(f"{granger_test_folder}/evaluation/variance_granger_matrix_normalized.png", variance_granger_matrix_normalized_img)
plotting.save_image(matrix=variance_granger_matrix_normalized, title="Variance Granger matrix", path=f"{granger_test_folder}/evaluation/variance_granger_matrix_normalized_by.png")
#print("variance_granger_matrix_normalized: ", matrix_loss(variance_granger_matrix_normalized, model))

mean_granger_matrix_thresholded_img = stf.plot.show_image(mean_granger_matrix_thresholded, f"Mean Granger matrix, threshold = {threshold}")
#cv2.imwrite(f"{granger_test_folder}/evaluation/mean_granger_matrix_thresholded.png", mean_granger_matrix_thresholded_img)
plotting.save_image(matrix=mean_granger_matrix_thresholded, title=f"Mean Granger matrix, threshold = {threshold}", path=f"{granger_test_folder}/evaluation/mean_granger_matrix_thresholded_by.png")
#print("mean_granger_matrix_thresholded: ", matrix_loss(mean_granger_matrix_thresholded, model))

variance_granger_matrix_thresholded_img = stf.plot.show_image(variance_granger_matrix_thresholded, f"Variance Granger matrix, threshold = {threshold}")
#cv2.imwrite(f"{granger_test_folder}/evaluation/variance_granger_matrix_thresholded.png", variance_granger_matrix_thresholded_img)
plotting.save_image(matrix=variance_granger_matrix_thresholded, title=f"Variance Granger matrix, threshold = {threshold}", path=f"{granger_test_folder}/evaluation/variance_granger_matrix_thresholded_by.png")
#print("variance_granger_matrix_thresholded: ", matrix_loss(variance_granger_matrix_thresholded, model))

'''
model_img = stf.plot.show_image(model, "Ground truth")
#cv2.imwrite(f"{granger_test_folder}/evaluation/model.png", model_img)
plotting.save_image(matrix=model, title="Ground truth", path=f"{granger_test_folder}/evaluation/model_by.png")
'''
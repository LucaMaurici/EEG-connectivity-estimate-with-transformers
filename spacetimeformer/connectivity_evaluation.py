import pickle
import numpy as np
import os
from spacetimeformer.spacetimeformer_model.utils.general_utils import  *
import cv2
import torch
import spacetimeformer as stf
import scipy.io
import config_file as cf

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def threshold(data):
    median = np.median(data) + 0.1
    print(f"median: {median}")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] < median:
                data[i,j] = 0
    return data

def matrix_loss(matrix1, matrix2):
    return np.sum(np.absolute(matrix1 - matrix2)**2)

NUM_CHANNELS = cf.read('num_channels')
NUM_CHANNELS_GT = 9

run_name, run_type, run_id = read_config_file()
print(f"run_name: {run_name}")
'''
if run_type == 'test_cross_attn_matrix':
    pass
else:
    raise Exception("MYERROR: Unknown or invalid run type")
'''

print("LOADING CROSS ATTENTION MATRIX...")
#attn_folder = f"./plots_checkpoints_logs/{run_name}/plots/{run_id}/{run_type}"
attn_folder = f"./plots_checkpoints_logs/{run_name}/plots/{run_id}/test_cross_attn_matrix"
with open(f"{attn_folder}/cross_attention_matrices.pkl", "rb") as file:
    cross_attn = pickle.load(file)

cross_attn_stack = torch.stack(cross_attn)
cross_attn_avg = torch.mean(cross_attn_stack, dim=0)
cross_attn_avg = cross_attn_avg.squeeze()

cross_attn = normalize_data(cross_attn[-1].cpu().numpy())
mask = np.ones((NUM_CHANNELS, NUM_CHANNELS))
np.fill_diagonal(mask, 0.0)
cross_attn = cross_attn * mask
cross_attn = threshold(cross_attn)
#cross_attn = np.rot90(cross_attn, 2)

cross_attn_avg = normalize_data(cross_attn_avg.cpu().numpy())
mask = np.ones((NUM_CHANNELS, NUM_CHANNELS))
np.fill_diagonal(mask, 0.0)
cross_attn_avg = cross_attn_avg * mask
cross_attn_avg = threshold(cross_attn_avg)
#cross_attn_avg = np.rot90(cross_attn_avg, 2)

print("LOADING SELF ATTENTION MATRIX...")
attn_folder = f"./plots_checkpoints_logs/{run_name}/plots/{run_id}/test_self_attn_matrix"
with open(f"{attn_folder}/self_attention_matrices.pkl", "rb") as file:
    self_attn = pickle.load(file)

self_attn_stack = torch.stack(self_attn)
self_attn_avg = torch.mean(self_attn_stack, dim=0)
self_attn_avg = self_attn_avg.squeeze()

self_attn_avg = normalize_data(self_attn_avg.cpu().numpy())
mask = np.ones((NUM_CHANNELS, NUM_CHANNELS))
np.fill_diagonal(mask, 0.0)
self_attn_avg = self_attn_avg * mask
self_attn_avg = threshold(self_attn_avg)
#self_attn_avg = np.rot90(self_attn_avg, 2)


print("LOADING PDC...")
mat = scipy.io.loadmat(f"./data/Generated EEG/PDC/PDC_GeneratedEEG_{NUM_CHANNELS_GT}chs_PoptComputed.mat")
struct = mat['APP'][0][0]
PDC = struct[0]
PDCtr = struct[1]
PDCpat = struct[2]

PDC = np.mean(PDCtr, axis=2)
PDC = np.nan_to_num(PDC)
mask = np.ones((NUM_CHANNELS_GT, NUM_CHANNELS_GT))
np.fill_diagonal(mask, 0.0)
PDC = PDC * mask
PDC = normalize_data(PDC)


print("LOADING Ground Truth...")
mat = scipy.io.loadmat(f"./data/Generated EEG/GeneratedEEG {NUM_CHANNELS_GT}chs.mat")
struct = mat['EEG'][0][0]
eeg = struct[0]
noise = struct[1]
model = struct[2]

model = model[:,:,0]
model = np.absolute(model)
model = normalize_data(model)


if not os.path.exists(f"{attn_folder}/evaluation/"):
    os.makedirs(f"{attn_folder}/evaluation/")

cross_attn_img = stf.plot.show_image(cross_attn, "Cross attention layer -1")
cv2.imwrite(f"{attn_folder}/evaluation/cross_attn_matrix_layer-1.png", cross_attn_img)
print("cross_attn: ", matrix_loss(cross_attn, model))

cross_attn_avg_img = stf.plot.show_image(cross_attn_avg, "Cross attention average")
cv2.imwrite(f"{attn_folder}/evaluation/cross_attn_avg_matrix.png", cross_attn_avg_img)
print("cross_attn_avg: ", matrix_loss(cross_attn_avg, model))

self_attn_avg_img = stf.plot.show_image(self_attn_avg, "Self attention average")
cv2.imwrite(f"{attn_folder}/evaluation/self_attn_avg_matrix.png", self_attn_avg_img)
print("self_attn_avg: ", matrix_loss(self_attn_avg, model))

pdc_img = stf.plot.show_image(PDC, "PDC")
cv2.imwrite(f"{attn_folder}/evaluation/pdc.png", pdc_img)
print("PDC: ", matrix_loss(PDC, model))

model_img = stf.plot.show_image(model, "Ground truth")
cv2.imwrite(f"{attn_folder}/evaluation/model.png", model_img)


'''
cross_attn = torch.stack(cross_attn)
cross_attn_avg = torch.mean(cross_attn, dim=0)
cross_attn_avg = cross_attn_avg.squeeze()
'''

'''
print("SAVING ATTENTION MATRIX...")
with open(f"{attn_folder}/cross_attn_matrix_avg.pkl", "wb") as file:
pickle.dump(cross_attn_avg, file)
'''

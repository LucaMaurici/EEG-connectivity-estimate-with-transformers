import torch
import numpy as np
from spacetimeformer.spacetimeformer_model.utils.general_utils import *
import config_file as cf
import pickle
import math

cf.reset()
num_channels = int(cf.read('num_channels'))
CHANNEL_START = 0
CHANNEL_END = num_channels
mean_loss_matrix = []
var_loss_matrix = []
for channel in range(CHANNEL_START, CHANNEL_END+1):
    write_run_id(channel)
    if channel == CHANNEL_START:
        num_channels -= 1
        cf.write('num_channels', num_channels)
    elif channel == CHANNEL_END:
        num_channels += 1
        cf.write('num_channels', num_channels)
        
    run_name, run_type, run_id = read_config_file()
    print(f"run_name: {run_name}")
    if run_type == 'test':
        pass
    else:
        raise Exception("MYERROR: Unknown or invalid run type")

    plots_folder = f"./plots_checkpoints_logs/{run_name}/plots/{channel}/{run_type}"
    channels_loss_path = f"{plots_folder}/channels_loss_val.pkl"
    #checkpoint_path = './plots_checkpoints_logs/{run_name}/checkpoints/{run_id}/epoch=21-val/norm_mse=0.03.ckpt'
    #checkpoint_path = get_list_of_files(f'./plots_checkpoints_logs/{run_name}/checkpoints/{channel}/')[0]
    #print(f"checkpoint_path: {checkpoint_path}")
    print(channels_loss_path)
    with open(channels_loss_path, 'rb') as file:
        channels_loss_list = pickle.load(file)
        #channels_loss_list.append(channels_loss)

    print(len(channels_loss_list))
    #shape: batches, losses, samples, channels
    new_tensor = torch.cat(channels_loss_list, dim=1)
    print()
    print(new_tensor.shape)
    #shape: losses, tot_samples, channels

    print()
    mean_tensor = new_tensor[2].mean(axis=-2)  # channel loss important
    var_tensor = new_tensor[2].var(axis=-2, unbiased=True)
    mean_loss_matrix.append(mean_tensor)
    var_loss_matrix.append(var_tensor)


mean_loss_EM = mean_loss_matrix[num_channels]
var_loss_EM = var_loss_matrix[num_channels]

mean_granger_matrix = np.zeros((num_channels, num_channels))
variance_granger_matrix = np.zeros((num_channels, num_channels))

for r in range(num_channels):
    chS = 0
    for chG in range(num_channels):
        if r == chG:
            continue

        mean_granger_matrix[r,chG] = math.log(mean_loss_matrix[r][chS]/mean_loss_EM[chG])
        variance_granger_matrix[r,chG] = math.log(var_loss_matrix[r][chS]/var_loss_EM[chG])

        chS +=1

mean_granger_matrix = mean_granger_matrix.transpose()
variance_granger_matrix = variance_granger_matrix.transpose()

print(mean_granger_matrix)
print(variance_granger_matrix)


output_folder = f"./plots_checkpoints_logs/{run_name}/plots/{run_id}/granger_test_val/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

filename = 'mean_granger_matrix.pkl'
with open(output_folder+filename, 'wb') as file:
    pickle.dump(mean_granger_matrix, file)

filename = 'variance_granger_matrix.pkl'
with open(output_folder+filename, 'wb') as file:
    pickle.dump(variance_granger_matrix, file)

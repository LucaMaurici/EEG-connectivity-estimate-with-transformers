import torch
import numpy as np
from spacetimeformer.spacetimeformer_model.utils.general_utils import *
import config_file as cf
import pickle
import math
import spacetimeformer as stf
import scipy.io
import results_file
rf = results_file.Results_File()

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def matrix_loss(pred, gt):
    diff = np.absolute(pred - gt)
    squared_diff = diff**2
    sum_gt = np.sum(np.absolute(gt))
    print(sum_gt)
    mse = np.sum(squared_diff)
    mape =  np.sum(diff)/sum_gt
    mspe = np.sum(squared_diff)/sum_gt
    pred_no_neg = np.maximum(pred, 0)
    gt_no_neg = np.maximum(gt, 0)
    jd = 1 - (np.sum(np.minimum(pred_no_neg, gt_no_neg))/np.sum(np.maximum(pred_no_neg, gt_no_neg)))
    print(mse, mape, mspe, jd)
    return mse, mape, mspe, jd

def standardize_data(data):
    return (data - np.mean(data)) / (np.std(data))

def threshold_data(data):
    return np.maximum(data, 0)

def threshold_std_neg_data(data):
    data_neg = np.minimum(data, 0)
    print(f"data_neg: {data_neg}")
    std_neg = np.std(data_neg)
    data_scaled = data - std_neg
    data_scaled = np.maximum(data_scaled, 0)
    data = data_scaled + std_neg
    return data

def zero_diagonal(data):
    mask = np.ones((data.shape[0], data.shape[1]))
    np.fill_diagonal(mask, 0.0)
    data = data * mask
    return data

cf.reset()

#SNRs = ['inf', '10000', '100', '3']
SNRs = ['inf']
REPETITIONS = 10
#SNRs = ['inf']
for snr in SNRs:
    for rep in range(1, REPETITIONS+1):
        cf.write('subject_id', f"SNR_{snr}_{rep}")

        num_channels = int(cf.read('num_channels'))
        CHANNEL_START = 0
        CHANNEL_END = num_channels
        mean_loss_matrix = []
        var_loss_matrix = []
        for channel in range(CHANNEL_START, CHANNEL_END+1):
            write_run_id(f"ch{channel}_sub_{cf.read('subject_id')}")
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

            plots_folder = f"./plots_checkpoints_logs/{run_name}/plots/{run_id}/{run_type}"
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
            var_tensor = new_tensor[1].var(axis=-2, unbiased=True)
            mean_loss_matrix.append(mean_tensor)
            var_loss_matrix.append(var_tensor)

        mean_loss_EM = mean_loss_matrix[num_channels].cpu().numpy()
        var_loss_EM = var_loss_matrix[num_channels]

        print(type(mean_loss_EM))

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

        ####
        k = num_channels
        gcval = np.zeros((k, k))
        for i in range(k):
            arix = [j for j in range(k) if i != j]
            for e, c in enumerate(arix):
                gcval[c, i] = math.log(mean_loss_matrix[i].cpu().numpy()[e]/mean_loss_EM[c]) # modificato con variances
        print(f"\n----gcval----\n{gcval}")

        ####

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


        NUM_CHANNELS_GT = 4

        ###################################################

        # mean_granger_matrix

        mean_granger_matrix_standardized = standardize_data(mean_granger_matrix)
        mean_granger_matrix_th = threshold_data(mean_granger_matrix)
        mean_granger_matrix_th_normalized = normalize_data(mean_granger_matrix_th)
        mean_granger_matrix_th_std_neg = threshold_std_neg_data(mean_granger_matrix)
        mean_granger_matrix_th_std_neg_normalized = normalize_data(mean_granger_matrix_th_std_neg)

        gcval_standardized = standardize_data(gcval)
        gcval_standardized_th = threshold_data(gcval_standardized)
        #mask = np.ones((NUM_CHANNELS, NUM_CHANNELS))
        #np.fill_diagonal(mask, 0.0)
        #mean_granger_matrix = mean_granger_matrix_standardized * mask
        #mean_granger_matrix_thresholded = threshold_num(mean_granger_matrix)

        granger_test_folder = f"./plots_checkpoints_logs/{run_name}/plots/{run_id}/granger_test_val"

        print("LOADING Ground Truth...")
        mat = scipy.io.loadmat(f'./data/Generated EEG/noise_test_standardized_1/GeneratedEEG 4chs_SNR_{snr}_{rep}.mat')
        struct = mat['EEG'][0][0]
        eeg = struct[0]
        noise = struct[1]
        model = struct[2]

        model = np.absolute(model)
        #model = np.mean(model, axis = -1)  # only for granger_test_3 data
        model = zero_diagonal(model)
        model_th = threshold_data(model)
        model_th_normalized = normalize_data(model_th)
        model = standardize_data(model)
        model_normalized = normalize_data(model)

        if not os.path.exists(f"{granger_test_folder}/evaluation/"):
            os.makedirs(f"{granger_test_folder}/evaluation/")

        plotting = stf.plot.AttentionMatrixCallback([])

        mean_granger_matrix_standardized_img = stf.plot.show_image(mean_granger_matrix_standardized, "Granger matrix spacetimeformer")
        #cv2.imwrite(f"{granger_test_folder}/evaluation/mean_granger_matrix_standardized.png", mean_granger_matrix_standardized_img)
        plotting.save_image(matrix=mean_granger_matrix_standardized, title="Granger causality spacetimeformer", path=f"{granger_test_folder}/evaluation/granger_matrix_spacetimeformer_by.png")
        plotting.save_image(matrix=mean_granger_matrix_th_normalized, title="Granger causality spacetimeformer", path=f"{granger_test_folder}/evaluation/granger_matrix_spacetimeformer_by_th.png")
        plotting.save_image(matrix=mean_granger_matrix_th_std_neg_normalized, title="Granger causality spacetimeformer", path=f"{granger_test_folder}/evaluation/granger_matrix_spacetimeformer_by_th_std_neg.png")
        mse, mape, mspe, jd = matrix_loss(mean_granger_matrix_standardized, model)
        mse_th, mape_th, mspe_th, jd_th = matrix_loss(mean_granger_matrix_th_normalized, model_th_normalized)
        mse_th_std_neg, mape_th_std_neg, mspe_th_std_neg, jd_th_std_neg = matrix_loss(mean_granger_matrix_th_std_neg_normalized, model_normalized)
        print("mean_granger_matrix_standardized: ", mse)

        model_img = stf.plot.show_image(model, "Ground truth")
        #cv2.imwrite(f"{granger_test_folder}/evaluation/model.png", model_img)
        plotting.save_image(matrix=model, title="Ground truth", path=f"{granger_test_folder}/evaluation/model_by.png")
        plotting.save_image(matrix=model_th_normalized, title="Ground truth", path=f"{granger_test_folder}/evaluation/model_by_th.png")
        plotting.save_image(matrix=model_normalized, title="Ground truth", path=f"{granger_test_folder}/evaluation/model_by_normalized.png")

        '''
        gcval_img = stf.plot.show_image(gcval_standardized, "Granger matrix spacetimeformer")
        #cv2.imwrite(f"{granger_test_folder}/evaluation/mean_granger_matrix_standardized.png", mean_granger_matrix_standardized_img)
        plotting.save_image(matrix=gcval_standardized, title="Granger causality spacetimeformer", path=f"{granger_test_folder}/evaluation/gcval_spacetimeformer_by.png")
        plotting.save_image(matrix=gcval_standardized_th, title="Granger causality spacetimeformer thresholded", path=f"{granger_test_folder}/evaluation/gcval_spacetimeformer_by_th.png")
        mse_gcval, mape_gcval, mspe_gcval, jd_gcval = matrix_loss(gcval_standardized, model)
        '''

        rf.select_file('transformer_granger_test_standardized_1_results.csv', \
            ['snr', 'rep', \
            'mse_spacetimeformer', 'mape_spacetimeformer', 'mspe_spacetimeformer', 'jd_spacetimeformer', \
            #'mse_gcval', 'mape_gcval', 'mspe_gcval', 'jd_gcval', \
            'mse_th_normalized', 'mape_th_normalized', 'mspe_th_normalized', 'jd_th_normalized', \
            'mse_th_std_neg', 'mape_th_std_neg', 'mspe_th_std_neg', 'jd_th_std_neg', \
            'order_spacetimeformer'])

        rf.write([snr, rep, mse, mape, mspe, jd, \
            #mse_gcval, mape_gcval, mspe_gcval, jd_gcval, \
            mse_th, mape_th, mspe_th, jd_th, \
            mse_th_std_neg, mape_th_std_neg, mspe_th_std_neg, jd_th_std_neg, \
            6])


        ##########################################################

        # variance_granger_matrix

        variance_granger_matrix_standardized = standardize_data(variance_granger_matrix)
        variance_granger_matrix_th = threshold_data(variance_granger_matrix)
        variance_granger_matrix_th_normalized = normalize_data(variance_granger_matrix_th)
        variance_granger_matrix_th_std_neg = threshold_std_neg_data(variance_granger_matrix)
        variance_granger_matrix_th_std_neg_normalized = normalize_data(variance_granger_matrix_th_std_neg)

        gcval_standardized = standardize_data(gcval)
        gcval_standardized_th = threshold_data(gcval_standardized)
        #mask = np.ones((NUM_CHANNELS, NUM_CHANNELS))
        #np.fill_diagonal(mask, 0.0)
        #variance_granger_matrix = variance_granger_matrix_standardized * mask
        #variance_granger_matrix_thresholded = threshold_num(variance_granger_matrix)

        granger_test_folder = f"./plots_checkpoints_logs/{run_name}/plots/{run_id}/granger_test_val"

        print("LOADING Ground Truth...")
        mat = scipy.io.loadmat(f'./data/Generated EEG/noise_test_standardized_1/GeneratedEEG 4chs_SNR_{snr}_{rep}.mat')
        struct = mat['EEG'][0][0]
        eeg = struct[0]
        noise = struct[1]
        model = struct[2]

        model = np.absolute(model)
        #model = np.mean(model, axis = -1)  # only for granger_test_3 data
        model = zero_diagonal(model)
        model_th = threshold_data(model)
        model_th_normalized = normalize_data(model_th)
        model = standardize_data(model)
        model_normalized = normalize_data(model)

        if not os.path.exists(f"{granger_test_folder}/evaluation/"):
            os.makedirs(f"{granger_test_folder}/evaluation/")

        plotting = stf.plot.AttentionMatrixCallback([])

        variance_granger_matrix_standardized_img = stf.plot.show_image(variance_granger_matrix_standardized, "Granger matrix spacetimeformer")
        #cv2.imwrite(f"{granger_test_folder}/evaluation/variance_granger_matrix_standardized.png", variance_granger_matrix_standardized_img)
        plotting.save_image(matrix=variance_granger_matrix_standardized, title="Granger causality spacetimeformer", path=f"{granger_test_folder}/evaluation/granger_matrix_spacetimeformer_by.png")
        plotting.save_image(matrix=variance_granger_matrix_th_normalized, title="Granger causality spacetimeformer", path=f"{granger_test_folder}/evaluation/granger_matrix_spacetimeformer_by_th.png")
        plotting.save_image(matrix=variance_granger_matrix_th_std_neg_normalized, title="Granger causality spacetimeformer", path=f"{granger_test_folder}/evaluation/granger_matrix_spacetimeformer_by_th_std_neg.png")
        mse, mape, mspe, jd = matrix_loss(variance_granger_matrix_standardized, model)
        mse_th, mape_th, mspe_th, jd_th = matrix_loss(variance_granger_matrix_th_normalized, model_th_normalized)
        mse_th_std_neg, mape_th_std_neg, mspe_th_std_neg, jd_th_std_neg = matrix_loss(variance_granger_matrix_th_std_neg_normalized, model_normalized)
        print("variance_granger_matrix_standardized: ", mse)

        model_img = stf.plot.show_image(model, "Ground truth")
        #cv2.imwrite(f"{granger_test_folder}/evaluation/model.png", model_img)
        plotting.save_image(matrix=model, title="Ground truth", path=f"{granger_test_folder}/evaluation/model_by.png")
        plotting.save_image(matrix=model_th_normalized, title="Ground truth thresholded", path=f"{granger_test_folder}/evaluation/model_by_th.png")
        plotting.save_image(matrix=model_normalized, title="Ground truth normalized", path=f"{granger_test_folder}/evaluation/model_by_normalized.png")


        gcval_img = stf.plot.show_image(gcval_standardized, "Granger matrix spacetimeformer")
        #cv2.imwrite(f"{granger_test_folder}/evaluation/variance_granger_matrix_standardized.png", variance_granger_matrix_standardized_img)
        plotting.save_image(matrix=gcval_standardized, title="Granger causality spacetimeformer", path=f"{granger_test_folder}/evaluation/gcval_spacetimeformer_by.png")
        plotting.save_image(matrix=gcval_standardized_th, title="Granger causality spacetimeformer thresholded", path=f"{granger_test_folder}/evaluation/gcval_spacetimeformer_by_th.png")
        mse_gcval, mape_gcval, mspe_gcval, jd_gcval = matrix_loss(gcval_standardized, model)

        '''
        rf.select_file('transformer_granger_test_standardized_variance_1_results.csv', \
            ['snr', 'rep', \
            'mse_spacetimeformer', 'mape_spacetimeformer', 'mspe_spacetimeformer', 'jd_spacetimeformer', \
            'mse_gcval', 'mape_gcval', 'mspe_gcval', 'jd_gcval', \
            'mse_th_normalized', 'mape_th_normalized', 'mspe_th_normalized', 'jd_th_normalized', \
            'mse_th_std_neg', 'mape_th_std_neg', 'mspe_th_std_neg', 'jd_th_std_neg', \
            'order_spacetimeformer'])
        
        rf.write([snr, rep, mse, mape, mspe, jd, \
            mse_gcval, mape_gcval, mspe_gcval, jd_gcval, \
            mse_th, mape_th, mspe_th, jd_th, \
            mse_th_std_neg, mape_th_std_neg, mspe_th_std_neg, jd_th_std_neg, \
            6])
        '''
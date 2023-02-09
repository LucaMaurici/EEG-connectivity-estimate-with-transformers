import numpy as np
import matplotlib.pyplot as plt
import connectivipy as cp
import pickle
import scipy
import sails
import os
import results_file
import spacetimeformer as stf
rf = results_file.Results_File()

def get_variances(model, data):
    residuals = model.get_residuals(data, forward_parameters=True)
    #residual.shape: channels samples-order trails
    residuals = np.transpose(residuals, (2,1,0))
    #residual.shape: trails samples-order channels
    residuals_list = list(residuals)
    residuals = np.concatenate(residuals_list, axis=0)
    #residual.shape: samples-order*trails channels 
    #return np.mean(residuals**2, axis=0)
    return np.var(residuals, axis=0)

def calculate_cgci_vm(eeg):
    k, N, T = eeg.shape

    #gciorder, _ = cp.conn.Mvar().order_akaike(eeg, p_max=15, method='vm')
    gciorder = 6
    delay_vect = np.arange(gciorder)

    model_full = sails.VieiraMorfLinearModel.fit_model(eeg, delay_vect)
    var_full = get_variances(model_full, eeg)
    print(f"\n----var_full----\n{var_full}")

    cgci_vm = np.zeros((k, k))
    for i in range(k):
        ar_idx = [j for j in range(k) if i != j]
        model_i = sails.VieiraMorfLinearModel.fit_model(eeg[ar_idx, :], delay_vect)
        var_i = get_variances(model_i, eeg[ar_idx, :])
        for e, c in enumerate(ar_idx):
            #gcval[c, i] = np.log(var_full[i, i]/var_i[e, e])  # originale
            cgci_vm[c, i] = np.log(var_i[e]/var_full[c]) # modificato

    return cgci_vm, gciorder

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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

def zero_diagonal(data):
    mask = np.ones((data.shape[0], data.shape[1]))
    np.fill_diagonal(mask, 0.0)
    data = data * mask
    return data

#SNRs = ['inf', '10000', '100', '3']
SNRs = ['inf']
#SNRs = ['10000']
REPETITIONS = 10

'''
rf.select_file('classic_granger_test_results_standardized_1.csv', \
    ['snr', 'rep', \

    'mse_cgci_yw', 'mse_cgci_yw_lib', 'mse_cgci_vm', 'mse_ar_model_yw', 'mse_transformer', \
    'mape_cgci_yw', 'mape_cgci_yw_lib', 'mape_cgci_vm', 'mape_ar_model_yw', 'mape_transformer', \
    'mspe_cgci_yw', 'mspe_cgci_yw_lib', 'mspe_cgci_vm', 'mspe_ar_model_yw', 'mspe_transformer', \
    'jd_cgci_yw', 'jd_cgci_yw_lib', 'jd_cgci_vm', 'jd_ar_model_yw', 'jd_transformer', \

    'mse_cgci_yw_th_normalized', 'mse_cgci_yw_lib_th_normalized', 'mse_cgci_vm_th_normalized', 'mse_ar_model_yw_th_normalized', 'mse_transformer_th_normalized', \
    'mape_cgci_yw_th_normalized', 'mape_cgci_yw_lib_th_normalized', 'mape_cgci_vm_th_normalized', 'mape_ar_model_yw_th_normalized', 'mape_transformer_th_normalized', \
    'mspe_cgci_yw_th_normalized', 'mspe_cgci_yw_lib_th_normalized', 'mspe_cgci_vm_th_normalized', 'mspe_ar_model_yw_th_normalized', 'mspe_transformer_th_normalized', \
    'jd_cgci_yw_th_normalized', 'jd_cgci_yw_lib_th_normalized', 'jd_cgci_vm_th_normalized', 'jd_ar_model_yw_th_normalized', 'jd_transformer_th_normalized', \

    'mse_cgci_yw_th_std_neg', 'mse_cgci_yw_lib_th_std_neg', 'mse_cgci_vm_th_std_neg', 'mse_ar_model_yw_th_std_neg', 'mse_transformer_th_std_neg', \
    'mape_cgci_yw_th_std_neg', 'mape_cgci_yw_lib_th_std_neg', 'mape_cgci_vm_th_std_neg', 'mape_ar_model_yw_th_std_neg', 'mape_transformer_th_std_neg', \
    'mspe_cgci_yw_th_std_neg', 'mspe_cgci_yw_lib_th_std_neg', 'mspe_cgci_vm_th_std_neg', 'mspe_ar_model_yw_th_std_neg', 'mspe_transformer_th_std_neg', \
    'jd_cgci_yw_th_std_neg', 'jd_cgci_yw_lib_th_std_neg', 'jd_cgci_vm_th_std_neg', 'jd_ar_model_yw_th_std_neg', 'jd_transformer_th_std_neg', \

    'order_cgci_yw', 'order_cgci_yw_lib', 'order_cgci_vm', 'order_transformer'])
'''
plotting = stf.plot.AttentionMatrixCallback([])

for snr in SNRs:
    for rep in range(1, REPETITIONS+1):

        #PATH = './data/Generated EEG/eeg_generated/ch4_sub_2.pkl'
        PATH = f'./data/Generated EEG/noise_test_standardized_1/eeg_generated_ch4_sub_SNR_{snr}_{rep}.pkl'
        #PATH = './data/Generated EEG/eeg_generated_SNRinf_4chs_sub_1.pkl'

        with open(PATH, 'rb') as file:
            eeg = pickle.load(file)

        #shape: trials samples chs
        eeg = np.transpose(eeg, (2,1,0))
        #shape: chs samples trials

        GCI = cp.conn.GCI()

        #order_cgci_yw, _ = cp.conn.Mvar().order_akaike(eeg, p_max=15, method='yw')
        order_cgci_yw = 6
        cgci_yw, cgci_yw_lib, ar_model_yw = GCI.calculate_multitrial(data=eeg, gcimethod='yw', gciorder=order_cgci_yw)
        #print(f"\n----cgci_yw----\n{cgci_yw.shape}")
        # plot result
        cgci_yw_rounded = np.round(cgci_yw, 2)
        '''
        for i in cgci_yw_rounded:
            print('\t'.join(map(str, i)))
        '''

        cgci_vm, order_cgci_vm = calculate_cgci_vm(eeg)
        #print(f"\n----cgci_vm----\n{cgci_vm}")


        print("LOADING Ground Truth...")
        mat = scipy.io.loadmat(f'./data/Generated EEG/noise_test_standardized_1/GeneratedEEG 4chs_SNR_{snr}_{rep}.mat')
        struct = mat['EEG'][0][0]
        eeg = struct[0]
        noise = struct[1]
        model = struct[2]

        print(model)
        model = np.absolute(model)
        #model = np.mean(model, axis = -1)  # only if noise_test_3
        model = zero_diagonal(model)
        model_th_std_neg = threshold_std_neg_data(model)
        model_th_std_neg_normalized = normalize_data(model_th_std_neg)
        model_th = threshold_data(model)
        model_th_normalized = normalize_data(model_th)
        model = standardize_data(model)
        
        
        cgci_yw_th_std_neg = threshold_std_neg_data(cgci_yw)
        cgci_yw_th_std_neg_normalized = normalize_data(cgci_yw_th_std_neg)
        cgci_yw_th = threshold_data(cgci_yw)
        cgci_yw_th_normalized = normalize_data(cgci_yw_th)
        cgci_yw = standardize_data(cgci_yw)
        mse_cgci_yw, mape_cgci_yw, mspe_cgci_yw, jd_cgci_yw = matrix_loss(cgci_yw, model)
        mse_cgci_yw_th, mape_cgci_yw_th, mspe_cgci_yw_th, jd_cgci_yw_th = matrix_loss(cgci_yw_th_normalized, model_th_normalized)
        mse_cgci_yw_th_std_neg, mape_cgci_yw_th_std_neg, mspe_cgci_yw_th_std_neg, jd_cgci_yw_th_std_neg = matrix_loss(cgci_yw_th_std_neg_normalized, model_th_std_neg_normalized)

        cgci_yw_lib_th_std_neg = threshold_std_neg_data(cgci_yw_lib)
        cgci_yw_lib_th_std_neg_normalized = normalize_data(cgci_yw_lib_th_std_neg)
        cgci_yw_lib_th = threshold_data(cgci_yw_lib)
        cgci_yw_lib_th_normalized = normalize_data(cgci_yw_lib_th)
        cgci_yw_lib = standardize_data(cgci_yw_lib)
        mse_cgci_yw_lib, mape_cgci_yw_lib, mspe_cgci_yw_lib, jd_cgci_yw_lib = matrix_loss(cgci_yw_lib, model)
        mse_cgci_yw_lib_th, mape_cgci_yw_lib_th, mspe_cgci_yw_lib_th, jd_cgci_yw_lib_th = matrix_loss(cgci_yw_lib_th_normalized, model_th_normalized)
        mse_cgci_yw_lib_th_std_neg, mape_cgci_yw_lib_th_std_neg, mspe_cgci_yw_lib_th_std_neg, jd_cgci_yw_lib_th_std_neg = matrix_loss(cgci_yw_lib_th_std_neg_normalized, model_th_std_neg_normalized)

        cgci_vm_th_std_neg = threshold_std_neg_data(cgci_vm)
        cgci_vm_th_std_neg_normalized = normalize_data(cgci_vm_th_std_neg)
        cgci_vm_th = threshold_data(cgci_vm)
        cgci_vm_th_normalized = normalize_data(cgci_vm_th)
        cgci_vm = standardize_data(cgci_vm) 
        mse_cgci_vm, mape_cgci_vm, mspe_cgci_vm, jd_cgci_vm = matrix_loss(cgci_vm, model)
        mse_cgci_vm_th, mape_cgci_vm_th, mspe_cgci_vm_th, jd_cgci_vm_th = matrix_loss(cgci_vm_th_normalized, model_th_normalized)
        mse_cgci_vm_th_std_neg, mape_cgci_vm_th_std_neg, mspe_cgci_vm_th_std_neg, jd_cgci_vm_th_std_neg = matrix_loss(cgci_vm_th_std_neg_normalized, model_th_std_neg_normalized)

        ar_model_yw = zero_diagonal(ar_model_yw)
        ar_model_yw = np.abs(ar_model_yw)
        ar_model_yw_th_std_neg = threshold_std_neg_data(ar_model_yw)
        ar_model_yw_th_std_neg_normalized = normalize_data(ar_model_yw_th_std_neg)
        ar_model_yw_th = threshold_data(ar_model_yw)
        ar_model_yw_th_normalized = normalize_data(ar_model_yw_th)
        ar_model_yw = standardize_data(ar_model_yw)
        mse_ar_model_yw, mape_ar_model_yw, mspe_ar_model_yw, jd_ar_model_yw = matrix_loss(ar_model_yw, model)
        mse_ar_model_yw_th, mape_ar_model_yw_th, mspe_ar_model_yw_th, jd_ar_model_yw_th = matrix_loss(ar_model_yw_th_normalized, model_th_normalized)
        mse_ar_model_yw_th_std_neg, mape_ar_model_yw_th_std_neg, mspe_ar_model_yw_th_std_neg, jd_ar_model_yw_th_std_neg = matrix_loss(ar_model_yw_th_std_neg_normalized, model_th_std_neg_normalized)

        
        #['name', 'snr', 'rep', 'cgci_yw', 'cgci_vm', 'transformer']
        '''
        rf.write([snr, rep, \
            mse_cgci_yw, mse_cgci_yw_lib, mse_cgci_vm, mse_ar_model_yw, None, \
            mape_cgci_yw, mape_cgci_yw_lib, mape_cgci_vm, mape_ar_model_yw, None, \
            mspe_cgci_yw, mspe_cgci_yw_lib, mspe_cgci_vm, mspe_ar_model_yw, None, \
            jd_cgci_yw, jd_cgci_yw_lib, jd_cgci_vm, jd_ar_model_yw, None, \

            mse_cgci_yw_th, mse_cgci_yw_lib_th, mse_cgci_vm_th, mse_ar_model_yw_th, None, \
            mape_cgci_yw_th, mape_cgci_yw_lib_th, mape_cgci_vm_th, mape_ar_model_yw_th, None, \
            mspe_cgci_yw_th, mspe_cgci_yw_lib_th, mspe_cgci_vm_th, mspe_ar_model_yw_th, None, \
            jd_cgci_yw_th, jd_cgci_yw_lib_th, jd_cgci_vm_th, jd_ar_model_yw_th, None, \

            mse_cgci_yw_th_std_neg, mse_cgci_yw_lib_th_std_neg, mse_cgci_vm_th_std_neg, mse_ar_model_yw_th_std_neg, None, \
            mape_cgci_yw_th_std_neg, mape_cgci_yw_lib_th_std_neg, mape_cgci_vm_th_std_neg, mape_ar_model_yw_th_std_neg, None, \
            mspe_cgci_yw_th_std_neg, mspe_cgci_yw_lib_th_std_neg, mspe_cgci_vm_th_std_neg, mspe_ar_model_yw_th_std_neg, None, \
            jd_cgci_yw_th_std_neg, jd_cgci_yw_lib_th_std_neg, jd_cgci_vm_th_std_neg, jd_ar_model_yw_th_std_neg, None, \

            order_cgci_yw, order_cgci_vm, 6, None])
        '''

        SUBFOLDER = "./data/Generated EEG/noise_test_standardized_1/plots_normalization"
        if not os.path.exists(SUBFOLDER):
            os.makedirs(SUBFOLDER)
        plotting.save_image(matrix=cgci_yw, title=f"Yule-Walker GC standardized", path=f"{SUBFOLDER}/YW_GC_GeneratedEEG 4chs_SNR_{snr}_{rep}.png")
        plotting.save_image(matrix=cgci_yw_lib, title=f"Yule-Walker GC", path=f"{SUBFOLDER}/YW_lib_GC_GeneratedEEG 4chs_SNR_{snr}_{rep}.png")
        plotting.save_image(matrix=cgci_vm, title=f"Vieira-Morf GC standardized", path=f"{SUBFOLDER}/VM_GC_GeneratedEEG 4chs_SNR_{snr}_{rep}.png")
        plotting.save_image(matrix=ar_model_yw, title=f"VAR model parameters", path=f"{SUBFOLDER}/AR_GeneratedEEG 4chs_SNR_{snr}_{rep}.png")
        plotting.save_image(matrix=model, title=f"Ground truth standardized", path=f"{SUBFOLDER}/GT_GeneratedEEG 4chs_SNR_{snr}_{rep}.png")

        plotting.save_image(matrix=cgci_yw_th_normalized, title=f"Yule-Walker GC normalized", path=f"{SUBFOLDER}/YW_GC_GeneratedEEG 4chs_SNR_{snr}_{rep}_th.png")
        plotting.save_image(matrix=cgci_yw_lib_th, title=f"Yule-Walker GC", path=f"{SUBFOLDER}/YW_lib_GC_GeneratedEEG 4chs_SNR_{snr}_{rep}_th.png")
        plotting.save_image(matrix=cgci_vm_th_normalized, title=f"Vieira-Morf GC normalized", path=f"{SUBFOLDER}/VM_GC_GeneratedEEG 4chs_SNR_{snr}_{rep}_th.png")
        plotting.save_image(matrix=ar_model_yw_th, title=f"VAR model parameters", path=f"{SUBFOLDER}/AR_GeneratedEEG 4chs_SNR_{snr}_{rep}_th.png")
        plotting.save_image(matrix=model_th_normalized, title=f"Ground truth normalized", path=f"{SUBFOLDER}/GT_GeneratedEEG 4chs_SNR_{snr}_{rep}_th.png")

        plotting.save_image(matrix=cgci_yw_th_std_neg, title=f"Yule-Walker GC", path=f"{SUBFOLDER}/YW_GC_GeneratedEEG 4chs_SNR_{snr}_{rep}_th_std_neg.png")
        plotting.save_image(matrix=cgci_yw_lib_th_std_neg, title=f"Yule-Walker GC", path=f"{SUBFOLDER}/YW_lib_GC_GeneratedEEG 4chs_SNR_{snr}_{rep}_th_std_neg.png")
        plotting.save_image(matrix=cgci_vm_th_std_neg, title=f"Vieira-Morf GC", path=f"{SUBFOLDER}/VM_GC_GeneratedEEG 4chs_SNR_{snr}_{rep}_th_std_neg.png")
        plotting.save_image(matrix=ar_model_yw_th_std_neg, title=f"VAR model parameters", path=f"{SUBFOLDER}/AR_GeneratedEEG 4chs_SNR_{snr}_{rep}_th_std_neg.png")
        plotting.save_image(matrix=model_th_std_neg_normalized, title=f"Ground truth", path=f"{SUBFOLDER}/GT_GeneratedEEG 4chs_SNR_{snr}_{rep}_th_std_neg.png")
        


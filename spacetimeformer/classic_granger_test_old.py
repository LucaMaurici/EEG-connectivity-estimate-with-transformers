import numpy as np
import matplotlib.pyplot as plt
import connectivipy as cp
import pickle

#PATH = './data/Generated EEG/eeg_generated/ch4_sub_2.pkl'
PATH = './data/Generated EEG/noise_test_3/eeg_generated_ch4_sub_SNR_10000_4.pkl'
#PATH = './data/Generated EEG/eeg_generated_SNRinf_4chs_sub_1.pkl'

with open(PATH, 'rb') as file:
    eeg = pickle.load(file)

#shape: trials samples chs
#np.random.seed(0)
#np.random.shuffle(eeg)
eeg = np.transpose(eeg, (2,1,0))
#shape: chs samples trials

GCI = cp.conn.GCI()

# connectivity analysis
data = cp.Data(data=eeg, fs=200, chan_names=["0", "1", "2", "3"])

# plot data (in multitrial case only one trial is shown)
#data.plot_data()

# compute connectivity using granger causality index
'''
gci_values = data.conn('gci', gciorder=None)[0]  # gciorder None means it will be estimated

# plot result
gci_values = np.round(gci_values, 2)
for i in gci_values:
    print('\t'.join(map(str, i)))
'''

cgci_yw = GCI.calculate_multitrial(data=eeg, gcimethod='yw', gciorder=12)[0]

# plot result
cgci_yw_rounded = np.round(cgci_yw, 2)
for i in cgci_yw_rounded:
    print('\t'.join(map(str, i)))


#g_significance = data.significance(Nrep=50, alpha=0.05)
#data.plot_conn('Granger causality (GCI)', signi=False)
'''
matrices = []
for t in range(np.shape(eeg)[-1]):
    matrices.append(GCI.calculate(eeg[:,:,t], gcimethod='yw', gciorder=10)[0])

print(np.mean(matrices, axis=0))
'''
import sails

def get_variances(model, data):
    residuals = model.get_residuals(data, forward_parameters=True)
    #residual.shape: channels samples-order trails
    residuals = np.transpose(residuals, (2,1,0))
    #residual.shape: trails samples-order channels
    residuals_list = list(residuals)
    residuals = np.concatenate(residuals_list, axis=0)
    #residual.shape: samples-order*trails channels 
    return np.var(residuals, axis=0)


k, N, T = eeg.shape

delay_vect = np.arange(12)

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
        cgci_vm[c, i] = np.log(var_i[e]/var_full[i]) # modificato

print(f"\n----cgci_vm----\n{cgci_vm}")


print("LOADING Ground Truth...")
mat = scipy.io.loadmat(f'./data/Generated EEG/noise_test_3/GeneratedEEG 4chs_SNR_10000_1.mat')
struct = mat['EEG'][0][0]
eeg = struct[0]
noise = struct[1]
model = struct[2]

model = np.mean(model, axis = -1)
model = np.absolute(model)
model = normalize_data(model)
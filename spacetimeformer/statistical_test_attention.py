
def modularity(G, communities, weight="weight", resolution=1):

    if not isinstance(communities, list):
        communities = list(communities)

    directed = True 
    
    out_degree = in_degree = dict(G.degree(weight=weight))
    deg_sum = sum(out_degree.values())
    m = deg_sum / 2
    norm = 1 / deg_sum**2

    def community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum = sum(in_degree[u] for u in comm) if directed else out_degree_sum

        return L_c / m - resolution * out_degree_sum * in_degree_sum * norm

    return sum(map(community_contribution, communities))

import networkx as nx
import networkx.algorithms.community as nx_comm
import pickle
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from spacetimeformer.spacetimeformer_model.utils.general_utils import *
import spacetimeformer as stf
plotting = stf.plot.AttentionMatrixCallback([])

def plot_matrix(matrix):
    plt.imshow(matrix)
    plt.colorbar()
    plt.show()

def standardize(lst):
    return (lst - np.mean(lst))/np.std(lst)

def standardize_torch(tensor, tensor_ref):
    tensor = (tensor - torch.mean(tensor))/torch.std(tensor)
    tensor_ref = (tensor_ref - torch.mean(tensor_ref))/torch.std(tensor_ref)
    min_value_tensor = torch.min(tensor)
    min_value_tensor_ref = torch.min(tensor_ref)
    tensor -= torch.minimum(min_value_tensor, min_value_tensor_ref)
    return tensor

def normalize_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def save_boxplot(solo, joint, title, path):
    matplotlib.use("Agg")
    plt.boxplot(np.transpose(np.array([solo, joint])))
    plt.title(title)
    plt.xticks([1, 2], ['solo', 'joint'])
    plt.savefig(path)
    plt.close()

run_name = 'hyperscanning_am_1_gradient'

#MATRIX_NAME = 'test_cross_attn_matrix/cross_attn_matrix_avg'
MATRIX_NAME = 'test_cross_self_attn_matrix/self_self_multiply_by'

subject_names_folder = 'C:/Users/lucam/Google Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer/data/eeg_hyperscanning/EEG data couples'
subjects = os.listdir(subject_names_folder)
subjects = [item for item in subjects.copy() if 'desktop.ini' not in item]
print(subjects)

conditions = ['solo', 'joint']
matrix_list = {
    'solo': [],
    'joint': []
}

sum_11_list = {
    'solo': [],
    'joint': []
}
sum_12_list = {
    'solo': [],
    'joint': []
}
sum_21_list = {
    'solo': [],
    'joint': []
}
sum_22_list = {
    'solo': [],
    'joint': []
}

sum_intra_list = {
    'solo': [],
    'joint': []
}
sum_inter_list = {
    'solo': [],
    'joint': []
}
sum_total_list = {
    'solo': [],
    'joint': []
}
density_intra_list = {
    'solo': [],
    'joint': []
}
density_inter_list = {
    'solo': [],
    'joint': []
}

divisibility_list = {
    'solo': [],
    'joint': []
}

modularity_list = {
    'solo': [],
    'joint': []
}

import results_file
from scipy import stats
rf = results_file.Results_File()

preprocessing = 'normalized'


matrix_names = get_list_of_files("C:/Users/lucam/Google Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer/plots_checkpoints_logs/hyperscanning_am_1_gradient/plots/ch30_sub_BUAL-ABFL_solo")
for i, names in enumerate(matrix_names.copy()):
    matrix_names[i] = matrix_names[i].replace( \
        'C:/Users/lucam/Google Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer/plots_checkpoints_logs/hyperscanning_am_1_gradient/plots/ch30_sub_BUAL-ABFL_solo', \
        ""
    )
    matrix_names[i] = matrix_names[i].replace("\\", '/')

print(matrix_names)

for matrix_name in matrix_names:
    if 'desktop.ini' in matrix_name \
    or '.png' in matrix_name \
    or 'matrices' in matrix_name:
        continue
    MATRIX_NAME = matrix_name

    matrix_list = {
        'solo': [],
        'joint': []
    }

    sum_11_list = {
        'solo': [],
        'joint': []
    }
    sum_12_list = {
        'solo': [],
        'joint': []
    }
    sum_21_list = {
        'solo': [],
        'joint': []
    }
    sum_22_list = {
        'solo': [],
        'joint': []
    }

    sum_intra_list = {
        'solo': [],
        'joint': []
    }
    sum_inter_list = {
        'solo': [],
        'joint': []
    }
    sum_total_list = {
        'solo': [],
        'joint': []
    }
    density_intra_list = {
        'solo': [],
        'joint': []
    }
    density_inter_list = {
        'solo': [],
        'joint': []
    }

    divisibility_list = {
        'solo': [],
        'joint': []
    }

    modularity_list = {
        'solo': [],
        'joint': []
    }

    for i, subject in enumerate(subjects):
        for cond in conditions:
            # invertire le label per sopperire all'errore delle label scambiate
            if cond == 'solo':
                subject_folder = f"ch30_sub_{subject}_joint"
            elif cond == 'joint':
                subject_folder = f"ch30_sub_{subject}_solo"
            #subject_folder = f"ch30_sub_{subject}_{cond}"
            matrix_path = f'./plots_checkpoints_logs/{run_name}/plots/{subject_folder}{MATRIX_NAME}'

            assert os.path.exists(matrix_path)
            with open(matrix_path, 'rb') as handle:
                matrix = pickle.load(handle)

            if cond == 'solo':
                #cond_ref = 'joint' 
                cond_ref = 'solo' # invertire le label per sopperire all'errore delle label scambiate
            else:
                #cond_ref = 'solo'
                cond_ref = 'joint' # invertire le label per sopperire all'errore delle label scambiate
            subject_folder = f"ch30_sub_{subject}_{cond_ref}"
            matrix_path = f'./plots_checkpoints_logs/{run_name}/plots/{subject_folder}{MATRIX_NAME}'
            assert os.path.exists(matrix_path)
            with open(matrix_path, 'rb') as handle:
                matrix_ref = pickle.load(handle)

            if preprocessing == 'standardized':
                matrix = standardize_torch(matrix, matrix_ref)
            elif preprocessing == 'normalized':
                matrix = normalize_torch(matrix)
            elif preprocessing == 'raw':
                pass
            else:
                raise Exception("MYERROR: preprocessing, invalid keyword")

            matrix_list[cond].append(matrix)

            #sum of subjects clusters
            sum_11 = torch.sum(matrix[:15, :15]).item()
            sum_11_list[cond].append(sum_11)
            sum_12 = torch.sum(matrix[:15, 15:]).item()
            sum_12_list[cond].append(sum_12)
            sum_21 = torch.sum(matrix[15:, :15]).item()
            sum_21_list[cond].append(sum_21)
            sum_22 = torch.sum(matrix[15:, 15:]).item()
            sum_22_list[cond].append(sum_22)

            # INTRA SUM
            sum_intra = sum_11+sum_22
            sum_intra_list[cond].append(sum_intra)
            

            # INTER SUM
            sum_inter = sum_12+sum_21
            sum_inter_list[cond].append(sum_inter)

            # TOTAL SUM
            sum_total = sum_intra + sum_inter
            sum_total_list[cond].append(sum_total)

            # INTRA DENSITY
            density_intra = sum_intra / (sum_total)
            density_intra_list[cond].append(density_intra)

            # INTER DENSITY
            density_inter = sum_inter / (sum_total)
            density_inter_list[cond].append(density_inter)

            # DIVISIBILITY
            divisibility = sum_total/(sum_total + sum_inter)
            divisibility_list[cond].append(divisibility)

            # MODULARITY
            modularity_res = modularity(nx.from_numpy_array(np.array(matrix)), [list(np.arange(0, 15, 1)), list(np.arange(15, 30, 1))])
            modularity_list[cond].append(modularity_res)

    solo_mean = torch.mean(torch.stack(matrix_list['solo']), axis=0)
    joint_mean = torch.mean(torch.stack(matrix_list['joint']), axis=0)

    if not os.path.exists(f"./statistical_test_attention/attn_means/{preprocessing}/{MATRIX_NAME[:-4]}/"):
        os.makedirs(f"./statistical_test_attention/attn_means/{preprocessing}/{MATRIX_NAME[:-4]}/")
    plotting.save_image(solo_mean, "Solo mean", f"./statistical_test_attention/attn_means/{preprocessing}/{MATRIX_NAME[:-4]}/solo.png")
    plotting.save_image(joint_mean, "Joint mean", f"./statistical_test_attention/attn_means/{preprocessing}/{MATRIX_NAME[:-4]}/joint.png")
    joint_minus_solo_mean = joint_mean-solo_mean
    plotting.save_image(joint_minus_solo_mean, "Joint minus solo mean", f"./statistical_test_attention/attn_means/{preprocessing}/{MATRIX_NAME[:-4]}/joint_minus_solo.png")
    plotting.save_image(torch.abs(joint_minus_solo_mean), "abs(Joint minus solo) mean", f"./statistical_test_attention/attn_means/{preprocessing}/{MATRIX_NAME[:-4]}/joint_minus_solo_abs.png")
    plotting.save_image(torch.abs(torch.maximum(joint_minus_solo_mean, torch.zeros(30,30))), "Joint minus solo mean - positives", f"./statistical_test_attention/attn_means/{preprocessing}/{MATRIX_NAME[:-4]}/joint_minus_solo_pos.png")
    plotting.save_image(torch.abs(torch.minimum(joint_minus_solo_mean, torch.zeros(30,30))), "Joint minus solo mean - negatives", f"./statistical_test_attention/attn_means/{preprocessing}/{MATRIX_NAME[:-4]}/joint_minus_solo_neg.png")
    
    '''
    print("sum_intra: " + str(stats.ttest_rel(sum_intra_list['solo'], sum_intra_list['joint'])))
    print("sum_inter: " + str(stats.ttest_rel(sum_inter_list['solo'], sum_inter_list['joint'])))
    print("sum_total: " + str(stats.ttest_rel(sum_total_list['solo'], sum_total_list['joint'])))
    print("density_intra: " + str(stats.ttest_rel(density_intra_list['solo'], density_intra_list['joint'])))
    print("density_inter: " + str(stats.ttest_rel(density_inter_list['solo'], density_inter_list['joint'])))
    print("divisibility: " + str(stats.ttest_rel(divisibility_list['solo'], divisibility_list['joint'])))
    print("modularity: " + str(stats.ttest_rel(modularity_list['solo'], modularity_list['joint'])))
    '''
    print(f"divisibility_list: {divisibility_list}")
    print(f"divisibility_list.keys(): {divisibility_list.keys()}")
    print(f"divisibility_list['solo'].len: {len(divisibility_list['solo'])}")

    stat_sum_intra, p_sum_intra = stats.ttest_rel(sum_intra_list['solo'], sum_intra_list['joint'])
    stat_sum_inter, p_sum_inter = stats.ttest_rel(sum_inter_list['solo'], sum_inter_list['joint'])
    stat_sum_total, p_sum_total = stats.ttest_rel(sum_total_list['solo'], sum_total_list['joint'])
    stat_density_intra, p_density_intra = stats.ttest_rel(density_intra_list['solo'], density_intra_list['joint'])
    stat_density_inter, p_density_inter = stats.ttest_rel(density_inter_list['solo'], density_inter_list['joint'])
    stat_divisibility, p_divisibility = stats.ttest_rel(divisibility_list['solo'], divisibility_list['joint'])
    stat_modularity, p_modularity = stats.ttest_rel(modularity_list['solo'], modularity_list['joint'])


    if p_sum_intra <= 0.05:
        if not os.path.exists(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/"):
            os.makedirs(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/")
        save_boxplot(sum_intra_list['solo'], sum_intra_list['joint'], \
            f"Intra-subject sum, stat={round(stat_sum_intra, 2)}, p-value={round(p_sum_intra, 4)}", f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/sum_intra.png")
    if p_sum_inter <= 0.05:
        if not os.path.exists(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/"):
            os.makedirs(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/")
        save_boxplot(sum_inter_list['solo'], sum_inter_list['joint'], \
            f"Inter-subject sum, stat={round(stat_sum_inter, 2)}, p-value={round(p_sum_inter, 4)}", f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/sum_inter.png")
    if p_sum_total <= 0.05:
        if not os.path.exists(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/"):
            os.makedirs(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/")
        save_boxplot(sum_total_list['solo'], sum_total_list['joint'], \
            f"Total sum, stat={round(stat_sum_total, 2)}, p-value={round(p_sum_total, 4)}", f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/sum_total.png")
    if p_density_intra <= 0.05:
        if not os.path.exists(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/"):
            os.makedirs(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/")
        save_boxplot(density_intra_list['solo'], density_intra_list['joint'], \
            f"Intra-subject density, stat={round(stat_density_intra, 2)}, p-value={round(p_density_intra, 4)}", f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/density_intra.png")
    if p_density_inter <= 0.05:
        if not os.path.exists(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/"):
            os.makedirs(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/")
        save_boxplot(density_inter_list['solo'], density_inter_list['joint'], \
            f"Inter-subject density, stat={round(stat_density_inter, 2)}, p-value={round(p_density_inter, 4)}", f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/density_inter.png")
    if p_divisibility <= 0.05:
        if not os.path.exists(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/"):
            os.makedirs(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/")
        save_boxplot(divisibility_list['solo'], divisibility_list['joint'], \
            f"Divisibility, stat={round(stat_divisibility, 2)}, p-value={round(p_divisibility, 4)}", f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/divisibility.png")
    if p_modularity <= 0.05:
        if not os.path.exists(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/"):
            os.makedirs(f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/")
        save_boxplot(modularity_list['solo'], modularity_list['joint'], \
            f"Modularity, stat={round(stat_modularity, 2)}, p-value={round(p_modularity, 4)}", f"./statistical_test_attention/attn_box_plots/{preprocessing}/{MATRIX_NAME[:-4]}/modularity.png")

    mean_sum_intra_solo = np.mean(sum_intra_list['solo'])
    mean_sum_inter_solo = np.mean(sum_inter_list['solo'])
    mean_sum_total_solo = np.mean(sum_total_list['solo'])
    mean_density_intra_solo = np.mean(density_intra_list['solo'])
    mean_density_inter_solo = np.mean(density_inter_list['solo'])
    mean_divisibility_solo = np.mean(divisibility_list['solo'])
    mean_modularity_solo = np.mean(modularity_list['solo'])

    std_sum_intra_solo = np.std(sum_intra_list['solo'])
    std_sum_inter_solo = np.std(sum_inter_list['solo'])
    std_sum_total_solo = np.std(sum_total_list['solo'])
    std_density_intra_solo = np.std(density_intra_list['solo'])
    std_density_inter_solo = np.std(density_inter_list['solo'])
    std_divisibility_solo = np.std(divisibility_list['solo'])
    std_modularity_solo = np.std(modularity_list['solo'])

    mean_sum_intra_joint = np.mean(sum_intra_list['joint'])
    mean_sum_inter_joint = np.mean(sum_inter_list['joint'])
    mean_sum_total_joint = np.mean(sum_total_list['joint'])
    mean_density_intra_joint = np.mean(density_intra_list['joint'])
    mean_density_inter_joint = np.mean(density_inter_list['joint'])
    mean_divisibility_joint = np.mean(divisibility_list['joint'])
    mean_modularity_joint = np.mean(modularity_list['joint'])

    std_sum_intra_joint = np.std(sum_intra_list['joint'])
    std_sum_inter_joint = np.std(sum_inter_list['joint'])
    std_sum_total_joint = np.std(sum_total_list['joint'])
    std_density_intra_joint = np.std(density_intra_list['joint'])
    std_density_inter_joint = np.std(density_inter_list['joint'])
    std_divisibility_joint = np.std(divisibility_list['joint'])
    std_modularity_joint = np.std(modularity_list['joint'])
    
    '''
    if not os.path.exists(f"./statistical_test_attention/ttest/"):
        os.makedirs(f"./statistical_test_attention/ttest/")
    rf.select_file(f'./statistical_test_attention/ttest/statistical_test_attention_{preprocessing}.csv', [ \
        'Matrix type', \
        'Stat. Sum intra', 'P-value Sum intra', \
        'Stat. Sum inter', 'P-value Sum inter', \
        'Stat. Sum total', 'P-value Sum total', \
        'Stat. Density intra', 'P-value Density intra', \
        'Stat. Density inter', 'P-value Density inter', \
        'Stat. Divisibility', 'P-value Divisibility', \
        'Stat. Modularity', 'P-value Modularity', \
    ])
    
    
    rf.write([
        matrix_name, \
        stat_sum_intra, p_sum_intra, \
        stat_sum_inter, p_sum_inter, \
        stat_sum_total, p_sum_total, \
        stat_density_intra, p_density_intra, \
        stat_density_inter, p_density_inter, \
        stat_divisibility, p_divisibility, \
        stat_modularity, p_modularity
    ])
    '''
    if not os.path.exists(f"./statistical_test_attention/indices/"):
        os.makedirs(f"./statistical_test_attention/indices/")
    rf.select_file(f'./statistical_test_attention/indices/mean_std_indices_attention_{preprocessing}.csv', [ \
        'Matrix type', \
        'Mean Sum intra SOLO', 'Mean Sum intra JOINT', \
        'STD Sum intra SOLO', 'STD Sum intra JOINT', \
        'Mean Sum inter SOLO', 'Mean Sum inter JOINT', \
        'STD Sum inter SOLO', 'STD Sum inter JOINT', \
        'Mean Sum total SOLO', 'Mean Sum total JOINT', \
        'STD Sum total SOLO', 'STD Sum total JOINT', \
        'Mean Density intra SOLO', 'Mean Density intra JOINT', \
        'STD Density intra SOLO', 'STD Density intra JOINT', \
        'Mean Density inter SOLO', 'Mean Density inter JOINT', \
        'STD Density inter SOLO', 'STD Density inter JOINT', \
        'Mean Divisibility SOLO', 'Mean Divisibility JOINT', \
        'STD Divisibility SOLO', 'STD Divisibility JOINT', \
        'Mean Modularity SOLO', 'Mean Modularity JOINT', \
        'STD Modularity SOLO', 'STD Modularity JOINT', \
    ])
    
    
    rf.write([ \
        matrix_name, \
        mean_sum_intra_solo, mean_sum_intra_joint, \
        std_sum_intra_solo, std_sum_intra_joint, \
        mean_sum_inter_solo, mean_sum_inter_joint, \
        std_sum_inter_solo, std_sum_inter_joint, \
        mean_sum_total_solo, mean_sum_total_joint, \
        std_sum_total_solo, std_sum_total_joint, \
        mean_density_intra_solo, mean_density_intra_joint, \
        std_density_intra_solo, std_density_intra_joint, \
        mean_density_inter_solo, mean_density_inter_joint, \
        std_density_inter_solo, std_density_inter_joint, \
        mean_divisibility_solo, mean_divisibility_joint, \
        std_divisibility_solo, std_divisibility_joint, \
        mean_modularity_solo, mean_modularity_joint, \
        std_modularity_solo, std_modularity_joint, \
    ])
    
    
'''
sum_intra_mean_solo = np.mean(sum_intra_list['solo'])
sum_intra_std_solo = np.std(sum_intra_list['solo'])
print(f"sum_intra_mean['solo']: {sum_intra_mean_solo}\n")
print(f"sum_intra_std['solo']: {sum_intra_std_solo}\n")

sum_intra_mean_joint = np.mean(sum_intra_list['joint'])
sum_intra_std_joint = np.std(sum_intra_list['joint'])
print(f"sum_intra_mean['joint']: {sum_intra_mean_joint}\n")
print(f"sum_intra_std['joint']: {sum_intra_std_joint}\n")

print(f"sum_intra_mean['joint']: {sum_intra_list['joint']}\n")
print()
print(f"sum_inter_list['solo']: {sum_inter_list['solo']}\n")
print(f"sum_inter_list['joint']: {sum_inter_list['joint']}\n")
print()
print(f"density_intra_list['solo']: {density_intra_list['solo']}\n")
print(f"density_intra_list['joint']: {density_intra_list['joint']}\n")
print()
print(f"density_inter_list['solo']: {density_inter_list['solo']}\n")
print(f"density_inter_list['joint']: {density_inter_list['joint']}\n")
print()
'''

solo_mean = torch.mean(torch.stack(matrix_list['solo']), axis=0)
joint_mean = torch.mean(torch.stack(matrix_list['joint']), axis=0)

plot_matrix(solo_mean)
plot_matrix(joint_mean)

from scipy import stats

print()
print()
sum_intra_list['solo'] = standardize(sum_intra_list['solo'])
sum_intra_list['joint'] = standardize(sum_intra_list['joint'])
print("sum_intra: " + str(stats.ttest_rel(sum_intra_list['solo'], sum_intra_list['joint'])))

sum_inter_list['solo'] = standardize(sum_inter_list['solo'])
sum_inter_list['joint'] = standardize(sum_inter_list['joint'])
print("sum_inter: " + str(stats.ttest_rel(sum_inter_list['solo'], sum_inter_list['joint'])))

sum_total_list['solo'] = standardize(sum_total_list['solo'])
sum_total_list['joint'] = standardize(sum_total_list['joint'])
print("sum_total: " + str(stats.ttest_rel(sum_total_list['solo'], sum_total_list['joint'])))

density_intra_list['solo'] = standardize(density_intra_list['solo'])
density_intra_list['joint'] = standardize(density_intra_list['joint'])
print("density_intra: " + str(stats.ttest_rel(density_intra_list['solo'], density_intra_list['joint'])))

density_inter_list['solo'] = standardize(density_inter_list['solo'])
density_inter_list['joint'] = standardize(density_inter_list['joint'])
print("density_inter: " + str(stats.ttest_rel(density_inter_list['solo'], density_inter_list['joint'])))

divisibility_list['solo'] = standardize(divisibility_list['solo'])
divisibility_list['joint'] = standardize(divisibility_list['joint'])
print("divisibility: " + str(stats.ttest_rel(divisibility_list['solo'], divisibility_list['joint'])))

modularity_list['solo'] = standardize(modularity_list['solo'])
modularity_list['joint'] = standardize(modularity_list['joint'])
print("modularity: " + str(stats.ttest_rel(modularity_list['solo'], modularity_list['joint'])))


'''
joint_minus_solo_mean = joint_mean - solo_mean
plot_matrix(joint_minus_solo_mean)

joint_minus_solo_mean_abs = torch.absolute(joint_minus_solo_mean)
plot_matrix(joint_minus_solo_mean)

joint_minus_solo_mean_pos_thr = torch.max(joint_minus_solo_mean, torch.zeros(30, 30))
plot_matrix(joint_minus_solo_mean_pos_thr)

joint_minus_solo_mean_neg_thr = torch.min(joint_minus_solo_mean, torch.zeros(30, 30))
plot_matrix(joint_minus_solo_mean_neg_thr)
'''
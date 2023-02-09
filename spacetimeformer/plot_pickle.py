import pickle
import numpy as np

PATH = 'C:\\Users\\lucam\\Google Drive\\Documenti\\Scuola\\Università\\Sapienza\\Tesi\\EEG-connectivity-estimate-with-transformers\\spacetimeformer\\spacetimeformer\\plots_checkpoints_logs\\generated_gc_epochs40, 4ch, el 2, dl 2, h 3, small model, bs 128, ids 1, cp 15, tp 1, loss mse, gsa full, gca full, lsa full, lca none\\plots\\4\\granger_test_val\\mean_granger_matrix.pkl'
with open(PATH, 'rb') as file:
    mat = pickle.load(file)

print(mat)
mat = np.round(mat, 2)

for i in mat:
    print('\t'.join(map(str, i)))
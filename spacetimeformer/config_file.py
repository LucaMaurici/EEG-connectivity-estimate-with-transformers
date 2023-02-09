import pandas as pd
pd.options.mode.chained_assignment = None

#folder_path = "J:/Il mio Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer/"
#folder_path = "C:/Users/lucam/Google Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer/"
folder_path = './'
filename = "config.csv"

def read(key):
	raw_df = pd.read_csv(folder_path+filename, delimiter=';')
	index = list(raw_df['key']).index(key)
	output = raw_df['value'][index]
	if output.isnumeric():
		output = int(output)
	return output

def write(key, value):
	raw_df = pd.read_csv(folder_path+filename, delimiter=';')
	index = list(raw_df['key']).index(key)
	raw_df['value'][index] = value
	raw_df.to_csv(filename, index=False, sep=';')

def reset():
    write('num_channels', 30)
    write('subject_id', "PEBE-STPH")
    write('run_nickname', 'hyperscanning_am_1_gradient')
    write('condition', 'solo')
    write('x_dim', 4)

    #print(cf.read('num_channels'))
    #print(cf.read('subject_id'))

def reset2():
    write('num_channels', 4)
    write('subject_id', "SNR_inf_1")
    write('run_nickname', 'eeg_generated')
    write('condition', 'solo')
    write('x_dim', 4)
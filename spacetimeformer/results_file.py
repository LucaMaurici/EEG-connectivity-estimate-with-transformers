import pandas as pd
import os
pd.options.mode.chained_assignment = None

#folder_path = "J:/Il mio Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer/"
#folder_path = "C:/Users/lucam/Google Drive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/spacetimeformer/"

class Results_File:

	folder_path = './'
	_filename = None
	_columns = None

	def select_file(self, filename, columns):
		self._filename = filename
		self._columns = columns
		if not os.path.exists(self.folder_path+self._filename):
			raw_df = pd.DataFrame(columns=self._columns)
			raw_df.to_csv(self._filename, index=False, sep=';')

	def write(self, row):
		print(self._filename)
		raw_df = pd.read_csv(self.folder_path+self._filename, delimiter=';')
		raw_df = pd.concat([raw_df, pd.DataFrame(data=[row], columns=self._columns)])
		raw_df.to_csv(self._filename, index=False, sep=';')

'''
def read(key):
	raw_df = pd.read_csv(folder_path+filename, delimiter=';')
	index = list(raw_df['key']).index(key)
	output = raw_df['value'][index]
	if output.isnumeric():
		output = int(output)
	return output

def write(key, value):
	raw_df = pd.read_csv(folder_path+filename, delimiter=';')
	try:
		index = list(raw_df['key']).index(key)
		raw_df['value'][index] = value
	except:
		raw_df = pd.concat([raw_df, pd.DataFrame(data=[[key, value]], columns=['key', 'value'])])
	raw_df.to_csv(filename, index=False, sep=';')
'''
'''
def write_row(row):
	raw_df = pd.read_csv(folder_path+filename, delimiter=';')
	try:
		index = list(raw_df['key']).index(key)
		raw_df['value'][index] = value
	except:
		raw_df = pd.concat([raw_df, pd.DataFrame(data=[row])])
	raw_df.to_csv(filename, index=False, sep=';')
'''



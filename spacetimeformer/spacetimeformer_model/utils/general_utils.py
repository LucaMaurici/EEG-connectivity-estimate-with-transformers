import os

CONFIG_FILE = 'config_2.txt'

def compute_channel_loss(v1, v2):
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    return (v1-v2)**2

def read_config_file():
    run_name = ''
    run_type = ''
    with open(CONFIG_FILE) as f:
        lines = f.readlines()
        for line in lines:
            if line[:10] == 'run_name: ':
                run_name = line[10:-1]
            if line[:10] == 'run_type: ':
                run_type = line[10:-1]
            if line[:8] == 'run_id: ':
                run_id = line[8:-1]

    if run_name == '':
        raise Exception('Unable to read run_name from config.txt')
    if run_type == '':
        raise Exception('Unable to read run_type from config.txt')
    if run_id == '':
        raise Exception('Unable to read run_id from config.txt')

    return run_name, run_type, run_id

def write_run_id(run_id):
    run_id = str(run_id)
    new_lines = []
    with open(CONFIG_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            new_line = line[:]
            if line[:8] == 'run_id: ':
                new_line = 'run_id: ' + run_id
            new_lines.append(new_line)
    new_lines.append('\n')
    with open(CONFIG_FILE, 'w') as f:   
        f.writelines(new_lines)

#write_run_id('prova')


'''
    For the given path, get the List of all files in the directory tree 
'''
def get_list_of_files(dir_name):
    # create a list of file and sub directories 
    # names in the given directory 
    list_of_files = os.listdir(dir_name)
    all_files = list()
    # Iterate over all the entries
    for entry in list_of_files:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)
                
    return all_files

def index_of_not_substring(string_list, substring):
    for index, elem in enumerate(string_list):
        if substring in elem:
            print('entrato', index)
            pass
        else:
            return index
    return 0
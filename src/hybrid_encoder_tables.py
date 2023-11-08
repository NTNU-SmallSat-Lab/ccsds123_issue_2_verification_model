import numpy as np

table_folder = "src/hybrid_encoder_tables/"

input_symbol_limit = np.array([12,10,8,6,6,4,4,4,2,2,2,2,2,2,2,0])
threshold = np.array([303336,225404,166979,128672,95597,69670,50678,34898,23331,14935,9282,5510,3195,1928,1112,408])

code_table_max_length = 257
code_table_input = np.full((16,code_table_max_length), fill_value='Z', dtype='U256')
code_table_output = np.full((16,code_table_max_length), fill_value='Z', dtype='U256')

flush_table_max_length = 256
flush_table_prefix = np.full((16,flush_table_max_length), fill_value='Z', dtype='U256')
flush_table_word = np.full((16,flush_table_max_length), fill_value='Z', dtype='U256')


def tables_init():
    for i in range(16):
        data = np.genfromtxt(table_folder + "code_" + str(i).zfill(2) + ".txt", delimiter=',', dtype='U256')
        data = np.pad(data, ((0,code_table_max_length-data.shape[0]), (0,0)), mode='constant', constant_values="Z")
        code_table_input[i] = data[:,0]
        code_table_output[i] = data[:,1]
    
    for i in range(16):
        data = np.genfromtxt(table_folder + "flush_" + str(i).zfill(2) + ".txt", delimiter=',', dtype='U256')
        data = np.pad(data, ((0,flush_table_max_length-data.shape[0]), (0,0)), mode='constant', constant_values="Z")
        flush_table_prefix[i] = data[:,0]
        flush_table_word[i] = data[:,1]

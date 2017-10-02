import os


pre_process_data_folder_name = 'pre_process_data_siamese'

data_folder_path = os.environ['AML_DATA'] + '/aml_dl/'+pre_process_data_folder_name+'/'

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)


pre_process_siam_config={
    'data_folder_path':data_folder_path,
    'file_name_prefix':'test_push_data_pre_processed',
    'samples_per_file':20,
    'data_file_range':range(1,461),
    'model_type':'siam',
}
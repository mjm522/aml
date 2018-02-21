import os

data_storage_path = os.environ['AML_DATA'] + '/aml_demos/baxter_data_collection/'


if not os.path.exists(data_storage_path):
    os.makedirs(data_storage_path)


collect_robot_data_config = {
    'data_folder_path': data_storage_path,
    'num_samples_per_file':5,
    'data_name_prefix':'robot_data_collection',
    'sampling_rate':30,
}

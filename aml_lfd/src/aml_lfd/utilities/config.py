import os

data_storage_path = os.environ['AML_DATA'] + '/aml_lfd/dmp/'

print "DMP demo storage path is:=\t", data_storage_path

if not os.path.exists(data_storage_path):
    os.makedirs(data_storage_path)

sim_record_config = {
    'screen_height':400,
    'screen_width':640,
    'ppm':100.,
    'bg_color':(0,0,0),
    'line_color':(255, 0, 0),
    'line_thickness':5,
    'file_name':data_storage_path+'recorded_trajectory_tmp.txt',
}
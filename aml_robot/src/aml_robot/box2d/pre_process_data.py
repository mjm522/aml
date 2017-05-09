import numpy as np
from data_manager import DataManager
from aml_io.io_tools import save_data
from aml_io.convert_tools import string2image
from config import config, pre_process_config
from aml_visual_tools.visual_tools import show_image

def check():

    dataman = DataManager.from_file(config['training_data_file'])

    for sample in dataman._data[1:]:

        print sample.keys()
        print sample['push_action']
        print sample['state_start']['position']
        break
        
        # print "Sample id \t", sample['sample_id']
        # start_image = [sample['state_start']['image_rgb']]
        # final_image = [sample['state_end']['image_rgb']]

        # print "Start position \t", sample['state_start']['position']
        # print "Final position \t", sample['state_end']['position'] 

        # show_image(start_image, "Start image")
        # show_image(final_image, "Final image")


def get_data():
    x_data = []
    y_data = []

    dataman = DataManager.from_file(config['training_data_file'])

    for sample in dataman._data[1:]:

        x_data.append((np.transpose(string2image(sample['state_start']['image_rgb']), axes=[2,1,0]).flatten(), 
                         np.transpose(string2image(sample['state_end']['image_rgb']), axes=[2,1,0]).flatten()))
        y_data.append(np.r_[sample['state_start']['position'], sample['state_end']['position'], sample['push_action'][0][2:4]]) #ix, iy

    return x_data, y_data


def save_files():
    x_data, y_data = get_data()
    num_data_per_file = pre_process_config['samples_per_file']
    file_name_prefix  = pre_process_config['file_name_prefix']
    data_folder_path  = pre_process_config['data_folder_path']
    under = 0
    data_file_idx = 0
    total_data = len(x_data)
    finished = False
    print "Total len", total_data
    while not finished:
        print("starting to write into queue")
        upper = under + num_data_per_file
        print("try to enqueue ", under, " to ", upper)
        if upper <= total_data:
            curr_data = x_data[under:upper]
            curr_target = y_data[under:upper]
            under = upper
        else:
            rest = upper - total_data
            curr_data = np.concatenate((x_data[under:total_data], x_data[0:rest]))
            curr_target = np.concatenate((y_data[under:total_data], y_data[0:rest]))
            under = rest
            finished = True

        data = {'x':curr_data, 'y':curr_target}
        data_file_idx += 1
        print "Len of x", len(curr_data)
        print "Len of y", len(curr_target)
        filename = data_folder_path +'/' + file_name_prefix + '_%02d.pkl'%data_file_idx
        save_data(data=data, filename=filename)


if __name__ == '__main__':
    save_files()


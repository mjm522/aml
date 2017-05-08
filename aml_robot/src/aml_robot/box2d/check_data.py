from config import config
from data_manager import DataManager
from aml_visual_tools.visual_tools import show_image

dataman = DataManager.from_file(config['training_data_file'])


for sample in dataman._data[1:]:
    
    print "Sample id \t", sample['sample_id']
    start_image = [sample['state_start']['image_rgb']]
    final_image = [sample['state_end']['image_rgb']]

    print "Start position \t", sample['state_start']['position']
    print "Final position \t", sample['state_end']['position'] 

    show_image(start_image, "Start image")
    show_image(final_image, "Final image")


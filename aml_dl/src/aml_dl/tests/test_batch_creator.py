from aml_dl.utilities.tf_batch_creator import BatchCreator


def main():
    batch_params = {
                    'buffer_size':45, 
                    'batch_size': 20, 
                    'data_file_indices': range(1,10), 
                    'model_type':'siam', 
                    'use_random_batches':False,
                    'files_per_read':10,
                    'load_pre_processd_data':True}

    batch_creator = BatchCreator(batch_params)

    x_batch1 = None; y_batch1 = None
    x_batch2 = None; y_batch2 = None
    
    #while loops are given since if we make batch_creator multi threaded, this will be 
    #useful
    while x_batch1 is None and y_batch1 is None:
        x_batch1, y_batch1, _ =  batch_creator.get_batch()
    
    while x_batch2 is None and y_batch2 is None:
        x_batch2, y_batch2, _ =  batch_creator.get_batch()

    x_check_list = []
    y_check_list = []
    for itm1, itm2, itm3, itm4 in zip(x_batch1, x_batch2, y_batch1, y_batch2):
        x_check_list.append((itm1==itm2).all())
        y_check_list.append((itm3==itm4).all())

    if any(x_check_list):
        print "Found a match in x_batch1 and x_batch2"
    else:
        print "No matches found in x_batch1 and x_batch2"

    if any(y_check_list):
        print "Found a match in y_batch1 and y_batch2"
    else:
        print "No matches found in y_batch1 and y_batch2"


if __name__ == '__main__':
    main()
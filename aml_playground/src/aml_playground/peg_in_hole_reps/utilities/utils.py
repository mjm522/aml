import numpy as np

def convert_str_ft_reading(ft_reading):
    
    strft = ft_reading.split(' ')
    ft_reading_array = np.zeros(6)
    k = 0
    for j in range(len(strft)):
        try:
            val = float(strft[j])
        except Exception as e:
            continue
        ft_reading_array[k] = val
        k += 1
        if k > 5:
            break

    return ft_reading_array

def convert_list_str_ft_reading(ft_reading_list):
    
    ft_reading_array = np.zeros((len(ft_reading_list), 6))

    k = 0
    for ft_reading in ft_reading_list:

        ft_reading_array[k,:] = convert_str_ft_reading(ft_reading)
        k += 1
    
    return ft_reading_array
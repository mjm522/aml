import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualise_push_data(filename = '/Users/ermanoarruda/Projects/ros_ws/PI2/data/push_data_good.csv', image_file_name=None):

    df = pd.DataFrame.from_csv(filename)

    print df

    x0s = np.array(df['xi'])
    y0s = np.array(df['yi'])


    xfs = np.array(df['xf'])
    yfs = np.array(df['yf'])

    plt.scatter(x0s, y0s,  s=80, c='r', marker='+')
    plt.scatter(xfs, yfs,  s=80, c='g', marker='+')


    plt.show(True)

    if image_file_name is not None:
        plt.savefig(image_file_name)

if __name__ == '__main__':
    visualise_push_data()
import numpy as np

import tensorflow as tf

from aml_robot.box2d.data_manager import DataManager

import matplotlib.pyplot as plt

from aml_dl.mdn.training.config import network_params

from aml_dl.mdn.model.mdn_push_inv_model import MDNPushInverseModel




def generate_y_test(inverse_model):

    data_manager = DataManager.from_file('tests/data_test.pkl')
    data_x = data_manager.pack_data_x(['state_start','state_end'])
    data_y = data_manager.pack_data_y()

    # Training ground truth

    h = inverse_model.run_op('z_hidden',data_x)
    
    N_SAMPLES = 1000
    
    x_test = np.float32(np.random.uniform(-5.5, 5.5, (4, N_SAMPLES))).T
    
    h_test = inverse_model.run_op('z_hidden',x_test)
    
    y_test = inverse_model.sample_out(x_test, 10)

    plt.figure(figsize=(8, 8))
    plt.plot(h,data_y,'ro', h_test, y_test,'bo',alpha=0.1)
    plt.show()


def plot_training_data(inverse_model):

    data_manager = DataManager.from_file('tests/data_test.pkl')
    data_x = data_manager.pack_data_x(['state_start','state_end'])
    data_y = data_manager.pack_data_y()

    h = inverse_model.run_op('z_hidden', data_x)

    plt.figure(figsize=(8, 8))
    plt.plot(h, data_y,'ro', alpha=0.3)
    plt.show()


def plot_arrow(ax,v, head_width=0.06, head_length=0.15, fc='g', ec='g'):

    ax.arrow(0, 0, v[0]/np.linalg.norm(v), v[1]/np.linalg.norm(v), head_width=head_width, head_length=head_length, fc=fc, ec=ec)


def main():


    sess = tf.Session()

    inverse_model = MDNPushInverseModel(sess=sess, network_params=network_params)
    inverse_model.init_model()
    

    # plot_training_data(inverse_model=inverse_model)

    # generate_y_test(inverse_model=inverse_model)
    quit = False

    while not quit:

        avg_angle = 0.0
        avg_push = 0.0
        N_SAMPLES = 100
        tgt = np.random.randn(2)

        v0 = np.zeros(2)#np.random.randn(2)

        # v0 = v0/np.linalg.norm(v0)
        # tgt = tgt/np.linalg.norm(tgt)

        ax = plt.axes()


        plot_arrow(ax,v0,head_width=0.06, head_length=0.15, fc='g', ec='g')
 
        plot_arrow(ax,tgt,head_width=0.06, head_length=0.15, fc='b', ec='b')


        input_x = np.expand_dims(np.r_[v0,tgt],0)
        theta = inverse_model.sample_out(input_x,N_SAMPLES)[0]
        print "Theta:", theta
        mean_theta = np.mean(theta)

        for i in range(0,N_SAMPLES):
            
            

            push_direction = np.array([np.cos(theta[i]),np.sin(theta[i])])

            avg_push += push_direction

            
            ax.set_ylim([-2,2])
            ax.set_xlim([-2,2])

            plot_arrow(ax,push_direction, head_width=0.05, head_length=0.1, fc='k', ec='k')

            
            
            print "Tgt: ", tgt, " PushDir:", push_direction, " Angle:", theta[i]

            
            plt.show(block=False)
            plt.draw()

        avg_push = avg_push/N_SAMPLES
        

        print "AVG_ANGLE: ", mean_theta

        plot_arrow(ax,v0,head_width=0.06, head_length=0.15, fc='g', ec='g')
 
        plot_arrow(ax,tgt,head_width=0.06, head_length=0.15, fc='b', ec='b')

        plot_arrow(ax,avg_push,head_width=0.05, head_length=0.1, fc='r', ec='r')


        plt.show()


if __name__ == '__main__':
    main()
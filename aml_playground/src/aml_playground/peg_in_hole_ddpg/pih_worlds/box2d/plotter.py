import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg

import pygame as pg

import pylab

import numpy as np



class Plotter(object):

    def __init__(self):
        self._surf = None


    def hist(self, nb=100, fig_id = 1):
        '''Draw a histogram of random numbers between 0 and 1'''
        
        try:
            nb=nb()
        except:
            pass
        fig = pylab.figure(fig_id, figsize=[4, 4], # Inches
                           dpi=50,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        ax = fig.gca()
        ax.hist([np.random.random() for x in range(nb)])

        self.make_fig(fig)

    def plot(self, xs, ys, fig_id = 1):
        '''Draw a histogram of random numbers between 0 and 1'''
        
        try:
            nb=nb()
        except:
            pass
        fig = pylab.figure(fig_id, figsize=[4, 4], # Inches
                           dpi=50,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        ax = fig.gca()
        ax.plot(xs,ys)

        self.make_fig(fig)

    def contour(self, X, Y, Z, fig_id = 1):
        '''Draw a histogram of random numbers between 0 and 1'''
        
        try:
            nb=nb()
        except:
            pass
        fig = pylab.figure(fig_id,figsize=[6.4, 4.8], # Inches
                           dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        
        pylab.axes(frameon = 0)
        ax = fig.gca()
        cp = ax.contour(X, Y, Z)
        ax.clabel(cp, inline=True, 
          fontsize=5)
        pylab.title('Contour Plot')
        pylab.xlabel('x (m)')
        pylab.ylabel('y (m)')

        self.make_fig(fig)


    def make_fig(self,fig):
        #Interface with matplotlib, draw plot on a given UI_Item "view"

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        size = canvas.get_width_height()

        #Create pygame surface from string
        self._surf = pg.image.fromstring(raw_data, size, "RGB")


    def draw(self, screen):

        if self._surf:
            screen.blit(self._surf,(0,0))





import numpy as np
from PIL import Image
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

import matplotlib.pyplot as plt
import matplotlib
import threading

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)

import copy

from config import config

class PyGameViewer(object):

    def __init__(self,config):

        self._config = config
        # --- pygame setup ---
        self._screen = pygame.display.set_mode((config['image_width'], config['image_height']), 0, 32)
        pygame.display.set_caption(config['window_caption'])
        self._clock = pygame.time.Clock()

        self._running = True
        self._loop_thread = None
        self._last_screen = None

    def save_screen(self,img,filename):

        # img = pygame.surfarray.pixels3d(screen) 
        # img = img.transpose(1,0,2)
        # print "SHAPE:", img.shape
        # im = Image.fromarray(img)
        # im.save("your_file.jpeg")

        if img is not None:
            matplotlib.image.imsave(filename, img)

    def handle_events(self):
        # Check the event queue
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                self._running = False

    def clear_screen(self):
        self._screen.fill((0, 0, 0, 0))

    def flip(self):

        img = pygame.surfarray.pixels3d(self._screen) 
        self._last_screen = copy.deepcopy(img.transpose(1,0,2))

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        
        self._clock.tick(self._config['fps'])

    def quit(self):

        pygame.quit()

            


        

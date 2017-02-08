import numpy as np
from PIL import Image
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_UP, K_DOWN, K_r, K_f)

import matplotlib.pyplot as plt
import matplotlib
import threading

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)

import copy

from config import config

class PyGameViewer(object):

    def __init__(self, world, config):

        self._config = config

        self._steps_per_frame = self._config['steps_per_frame']

        self._world = world

        # --- pygame setup ---
        self._screen = pygame.display.set_mode((config['image_width'], config['image_height']), 0, 32)
        pygame.display.set_caption(config['window_caption'])
        self._clock = pygame.time.Clock()


        pygame.font.init()
        self._text_font = pygame.font.SysFont("monospace", 15)

        self._running = True
        self._loop_thread = None
        self._last_screen = None


    def create_text_surface(self, text, colour = (255,255,0)):

        label = self._text_font.render(text, 1, colour)

        return label

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

            elif  (event.type == KEYDOWN and event.key == K_UP):
                self._steps_per_frame += 1
                print "simulation steps per frame", self._steps_per_frame
            elif  (event.type == KEYDOWN and event.key == K_DOWN):
                self._steps_per_frame -= 1
                print "simulation steps per frame", self._steps_per_frame

            elif  (event.type == KEYDOWN and event.key == K_r):
                self._steps_per_frame = 1
                print "simulation steps per frame", self._steps_per_frame

            elif  (event.type == KEYDOWN and event.key == K_f):
                self._steps_per_frame += 50
                print "simulation steps per frame", self._steps_per_frame


            self._world.handle_event(event)


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


    def loop(self):

        # --- main game loop ---
        while self._running:

            self.handle_events()

            self.clear_screen()

            for i in range(self._steps_per_frame):
                self._world.update(self)
                self._world.step()


            # Draw the world
            self._world.draw(self)
            # self.save_screen()

            
            self.flip()

            


        

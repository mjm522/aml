import copy
import pygame
import threading
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_UP, K_DOWN, K_r, K_f)

class Box2DViewer(object):

    def __init__(self, world, config, is_thread_loop = True):

        self._config = config

        self._steps_per_frame = self._config['steps_per_frame']

        self._world = world

        # --- pygame setup ---
        # pygame.init()
        self._screen = pygame.display.set_mode((config['image_width'], config['image_height']), 0, 32)
        pygame.display.set_caption(config['window_caption'])
        self._clock = pygame.time.Clock()

        pygame.font.init()
        self._text_font = pygame.font.SysFont("monospace", 15)

        self._running = True
        self._loop_thread = None
        self._last_screen = None

        if is_thread_loop:
            self.threaded_loop()


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


    def clear_screen(self, color=(0,0,0,0)):
        self._screen.fill(color)

    def flip(self):

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        
        self._clock.tick(self._config['fps'])

    def quit(self):

        pygame.quit()


    def store_screen(self):
        img = copy.deepcopy(pygame.surfarray.pixels3d(self._screen))
        self._last_screen = copy.deepcopy(img.transpose(1,0,2))


    def draw(self, view_info=True):
        # Draw the world
        #view_info to true to see the text, arrow as well as point of pushing
        self._world.draw(self, view_info=view_info)
        # self.save_screen()
        self.store_screen()
        self.flip()


    def loop(self):

        # --- main game loop ---
        while self._running:

            self.loop_once()

    def threaded_loop(self):

        self._thread = threading.Thread(target=self.loop, args=())
        self._thread.daemon = True # Daemonize thread
        self._thread.start() # Start the execution

    
    def loop_once(self):
        
        self.handle_events()

        self.clear_screen(color=(255,255,255,255))

        for i in range(self._steps_per_frame):
            self._world.update(self)               
            self._world.step()

        self.draw(view_info=False)





            
            

            


        

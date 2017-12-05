import copy
import pygame
import threading
import numpy as np
from PIL import Image
from aml_io.io_tools import save_data
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_UP, K_DOWN, K_r, K_f, K_s, K_c, K_i, K_j, K_k, K_l, K_d, K_e, K_w)

class Box2DViewer(object):

    def __init__(self, world, config, is_thread_loop = True):


        pygame.init()

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

        self._clear_screen_enabled = True

        if 'cam_pos' in self._config.keys():
            self._cam_pos = self._config['cam_pos']
        else:
            self._cam_pos = [0., 0.]

        if is_thread_loop:
            self.threaded_loop()

        #for learning from demonstration
        self._start_demo_record = False
        self._demo_traj = []
        self._demo_point_list = []
        self._clear_demo = False


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

            elif  (event.type == KEYDOWN and event.key == K_s):
                self._world.reset()
                print "Reseting world"

            elif  (event.type == KEYDOWN and event.key == K_c):
                self._clear_screen_enabled = not self._clear_screen_enabled

                self._world._clear_screen_enabled = self._clear_screen_enabled
                print "Toggle clear screen: ", self._clear_screen_enabled

            elif  (event.type == KEYDOWN and event.key == K_i):
                self._cam_pos[1] = self._cam_pos[1] - 5

                print "Camera pos", self._cam_pos

            elif  (event.type == KEYDOWN and event.key == K_k):
                self._cam_pos[1] = self._cam_pos[1] + 5

                print "Camera pos", self._cam_pos

            elif  (event.type == KEYDOWN and event.key == K_j):
                self._cam_pos[0] = self._cam_pos[0] - 5

                print "Camera pos", self._cam_pos

            elif  (event.type == KEYDOWN and event.key == K_l):
                self._cam_pos[0] = self._cam_pos[0] + 5

                print "Camera pos", self._cam_pos

            elif (event.type == KEYDOWN and event.key == K_d):
                self._start_demo_record = True
                self._clear_demo = False

                print "Demo recording started, move mouse ..."

            elif (event.type == KEYDOWN and event.key == K_e):
                self._start_demo_record = False
                if self._config['save_demo']:
                    save_data(np.asarray(self._demo_traj), self._config['demo_storage_path']+'demo.pkl')
                    print "Saved demo in path:", self._config['demo_storage_path']
                print "Demo recording ended!"

            elif (event.type == KEYDOWN and event.key == K_w):
                self._clear_demo = True
                self._demo_point_list = []
                self._demo_traj = []

            elif (event.type == pygame.MOUSEMOTION) and self._start_demo_record:
                x, y = event.pos
                if (0 < x < self._config['image_width']-1) and (0 < y < self._config['image_height']-1):
                    self._demo_point_list.append([x,y])
                    self._demo_traj.append([float(x+self._config['cam_pos'][0])/self._config['pixels_per_meter'], float(self._config['image_height']-y-self._config['cam_pos'][1])/self._config['pixels_per_meter']])

            self._world.handle_event(event)


    def clear_screen(self, color=(0,0,0,0)):
        if self._clear_screen_enabled:
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


    def draw_demo(self):
        if self._clear_demo:
            return
        if len(self._demo_point_list) > 1:
            # curr_frame = self._world.get_frame(self._cam_pos)  => needed?
            pygame.draw.lines(self._screen, (255, 0, 0), False, self._demo_point_list, 3)
            # self._screen.blit(curr_frame, (0, 0)) => needed?

    def draw(self):
        # Draw the world
        #view_info to true to see the text, arrow as well as point of pushing
        self._world.draw(self._screen, self._cam_pos)
        self.draw_demo()
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


        # action = self._world.sample_push_action()

        for i in range(self._steps_per_frame):
            # self._world.update(self)               
            self._world.step()
            pass

        self.draw()





            
            

            


        

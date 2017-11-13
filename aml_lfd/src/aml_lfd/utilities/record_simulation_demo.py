#how to use: The script starts with a window, press the left button to start
#recording of the trajectory and click again to stop th recording.

import pygame
import numpy as np
from config import sim_record_config

x = y = 0
count = 0
running = True
pointlist = []
traj_to_save = []
start_record = False
PPM = sim_record_config['ppm']
SCREEN_WIDTH = sim_record_config['screen_width']
SCREEN_HEIGHT = sim_record_config['screen_height']

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

while running:
    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        running = False
    
    if pygame.mouse.get_pressed()[0]:
        if not start_record:
            start_record = True
        else:
            start_record = False

    if (event.type == pygame.MOUSEMOTION) and start_record:
        x, y = event.pos
        if (0 < x < SCREEN_WIDTH-1) and (0 < y < SCREEN_HEIGHT-1):
            count += 1
            pointlist.append([x,y])
            traj_to_save.append([float(x)/PPM, float(SCREEN_HEIGHT-y)/PPM])
   
    screen.fill(sim_record_config['bg_color'])

    if count > 1:
    #since atleat this function needs two points
        pygame.draw.lines(screen, sim_record_config['line_color'], False, pointlist, sim_record_config['line_thickness'])
    
    pygame.display.flip()

pointlist = np.asarray(pointlist)
traj_to_save = np.asarray(traj_to_save)
np.savetxt(sim_record_config['file_name'], traj_to_save)
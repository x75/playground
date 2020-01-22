# Constants used by each solution
from math import sin, cos, pi, sqrt
import random
import numpy as np
THETA = pi / 3.0 # Angle from one point to the next
HEXES_HIGH = 8 # How many rows of hexes
HEXES_WIDE = 5 # How many hexes in a row
RADIUS = 30 # Size of a hex
HALF_RADIUS = RADIUS / 2.0
HALF_HEX_HEIGHT = sqrt(RADIUS ** 2 - HALF_RADIUS ** 2)
IMAGE_WIDTH = int(RADIUS * (HEXES_WIDE * 3 + .5))
IMAGE_HEIGHT = int(HALF_HEX_HEIGHT * (HEXES_HIGH + 1))
# Functions (generators) used by each solution
def hex_points(x,y):
    '''Given x and y of the origin, return the six points around the origin of RADIUS distance'''
    for i in range(6):
        yield cos(THETA * i) * RADIUS + x, sin(THETA * i) * RADIUS + y
        
def hex_centres():
    hexes = []
    # hexes = np.zeros()
    for x in range(HEXES_WIDE):
        for y in range(HEXES_HIGH):
            cell_x = (x * 3 + 1) * RADIUS + RADIUS * 1.5 * (y % 2)
            cell_y = (y + 1) * HALF_HEX_HEIGHT
            # cell_c = (50,50,50)
            if x == 0:
                cell_c = np.array((255, 255, 255))
            else:
                cell_c = np.array((0, 0, 0))
                
            print('coord = {0}'.format(cell_x))
            # yield 
            if y == 0:
                hexes.append([])
            hexes[x].append({'x': cell_x, 'y': cell_y, 'c': cell_c})
    return hexes

def pygame_colours():
    while True:
        yield 255, 0, 0 # red
        yield 255, 255, 0 # yellow
        yield 0, 0, 255 # blue
        yield 0, 255, 0 # green
        yield 0, random.randint(0, 40), 0 # green
        
def pygame_hex():
    '''Requires PyGame 1.8 or better to save as PNG'''
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT))
    colours = pygame_colours()
    print('colours', colours)
    for x,y in hex_centres():
        pygame.draw.polygon(screen, next(colours), list(hex_points(x,y)))
    pygame.image.save(screen, 'pygame_hexes.png')


def pygame_hex_live():
    import pygame
    from pygame.locals import QUIT, KEYDOWN
    pygame.init()
    srf = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT))

    colours = pygame_colours()

    fps = 25
    dt = 1.0/fps
    loopFlag = True
    clk = pygame.time.Clock()
    clk_elapsed = 0
    hexes = hex_centres()

    while loopFlag:
        events = pygame.event.get()
        for e in events:
            if e.type==QUIT:
                loopFlag=False
            if e.type==KEYDOWN:
                loopFlag=False

        # Clear the screen
        srf.fill((255,255,255))

        # print('colours', colours)
        # for x_ in hexes:
        #     for y_ in x_:
        #         x,y = y_[0], y_[1]
        #         print('coord = {0},{1}'.format(x,y))
        #         pygame.draw.polygon(srf, next(colours), list(hex_points(x,y)))

        for x_ in range(HEXES_WIDE):
            for y_ in range(HEXES_HIGH):
                # print('coord = {0},{1},{2}'.format(x,y,c))
                # x,y = y_[0], y_[1]
                cell = hexes[x_][y_]
                x = cell['x']
                y = cell['y']
                c = cell['c']
                # print('coord = {0},{1},{2}'.format(x,y,c))
                print('coord = {0}'.format(hexes[x_][y_]))
                # pygame.draw.polygon(srf, next(colours), list(hex_points(x,y)))

                # rules
                if x_ == 0 and (clk_elapsed % 100 == 99):
                    # hexes[x_][y_]['c'] = np.abs(hexes[x_][y_]['c'] - 255)
                    # print('c = {0}'.format(hexes[x_][y_]['c']))
                    hexes[x_][y_]['c'] = np.random.randint(0, 255, (3,))
                    print('c = {0}'.format(hexes[x_][y_]['c']))

                if x_ > 0:
                    cell_w = hexes[x_ - 1][y_]
                    print('c = {0}'.format(c))
                    c = c + 0.1 * (cell_w['c'] - c)
                    print('c = {0}'.format(c))
                    # cell['c'] = c
                    hexes[x_][y_]['c'] = c.copy()
                    print('c = {0}'.format(hexes[x_][y_]))

                pygame.draw.polygon(srf, c, list(hex_points(x,y)))
                
                
        pygame.display.flip()

        # Next simulation step
        # world.step(dt)

        clk_elapsed += 1
        # Try to keep the specified framerate    
        clk.tick(fps)
    
if __name__ == '__main__':
    # pygame_hex()

    pygame_hex_live()

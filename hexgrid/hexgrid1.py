# Constants used by each solution
from math import sin, cos, pi, sqrt
import random
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
    for x in range(HEXES_WIDE):
        for y in range(HEXES_HIGH):
            yield (x * 3 + 1) * RADIUS + RADIUS * 1.5 * (y % 2), (y + 1) * HALF_HEX_HEIGHT

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
        for x,y in hex_centres():
            pygame.draw.polygon(srf, next(colours), list(hex_points(x,y)))

        pygame.display.flip()

        # Next simulation step
        # world.step(dt)

        # Try to keep the specified framerate    
        clk.tick(fps)
    
if __name__ == '__main__':
    # pygame_hex()

    pygame_hex_live()

from hlt import *
from networking import *
from random import choice, shuffle
from itertools import product
from scipy.signal import convolve2d
import logging
import sys
import numpy as np
import time


# Initial setup
logging.basicConfig(filename='potential_bot.log',
                    filemode='w',
                    level=logging.DEBUG)

myID, gameMap = getInit()
sendInit("PotentialBot")
gameMap = getFrame()
width, height = gameMap.width, gameMap.height
prodmap = np.zeros((height, width))
ownermap = np.zeros((height, width)).astype('i')
strmap = np.zeros((height, width))

xyiter = lambda: product(range(height), range(width))

# set up the production map
for y, x in xyiter():
    prodmap[y][x] = gameMap.getSite(Location(x, y)).production
prodmap = prodmap / 255 # normalize

# set up getLocation with lightweight tuples
def dloc(loc, direction=0):
    dlocs = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
    y, x = loc
    x = (x + dlocs[direction][0]) % width
    y = (y + dlocs[direction][1]) % height
    return (y, x)

def conv2d(x, filter):
    return convolve2d(x, filter, mode='same', boundary='wrap')

def roll(x):
    return np.stack([
        x,
        np.roll(x,  1, axis=1),
        np.roll(x, -1, axis=0),
        np.roll(x, -1, axis=1),
        np.roll(x,  1, axis=0),
    ], axis=0)

turn = 0
while True:
    t = time.time()
    moves = []
    for y, x in xyiter():
        site = gameMap.getSite(Location(x, y))
        ownermap[y][x] = site.owner
        strmap[y][x] = site.strength
    strmap /= 255

    movedmap = np.zeros((height, width))

    # masks
    my_units = (ownermap == myID)
    neutral_mask = (ownermap == 0)
    enemy_mask = myUnitMask * (ownermap != 0)


    filter = np.array([
        [.25, 1, .25],
        [1, -5, 1],
        [.25, 1, .25],
    ])
    preferred_loc = conv2d(enemy_mask * strmap, filter) 

    self_str = strmap * myUnitMask
    neutral_str = roll(strmap * neutral_mask)
    attack_score = np.tanh(self_str - neutral_str)

    scoremap = np.zeros((5, height, width))

    
    scoremap += attack_score
    exp = np.exp(scoremap)
    softmax = exp / np.sum(exp, axis=0)

    for loc in xyiter():
        if ownermap[loc] == myID:
            # TODO

    sendFrame(moves)
    gameMap = getFrame()
    turn += 1

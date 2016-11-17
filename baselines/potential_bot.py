from hlt import *
from networking import *
from random import choice, shuffle
from itertools import product
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import argparse
import logging
import sys
import numpy as np
import time


# Initial setup
logging.basicConfig(filename='potential_bot.log',
                    filemode='w',
                    level=logging.DEBUG)
parser = argparse.ArgumentParser(description='Superbot with params!')
parser.add_argument('-d', '--debug', default=0,
                    help="Whether to do debugging checks")
parser.add_argument('--p_a', default=1, help="Attack modifier")
parser.add_argument('--p_p', default=1, help="production modifier")
parser.add_argument('--p_n', default=1, help="neutral attack modifier")
parser.add_argument('--p_np', default=1, help="head towards weaker neutrals")
parser.add_argument('--p_i', default=1, help="inertial modifier")
parser.add_argument('--p_e', default=1, help="fan-out modifier")
args = parser.parse_args()


myID, gameMap = getInit()
sendInit("PotentialBot")
gameMap = getFrame()
width, height = gameMap.width, gameMap.height
prodmap = np.zeros((width, height))
ownermap = np.zeros((width, height)).astype('i')
strmap = np.zeros((width, height))
n_tiles = width * height
tot_turns = 10 * np.sqrt(width * height)

xyiter = lambda: product(range(width), range(height))

# set up the production map
for x, y in xyiter():
    prodmap[x][y] = gameMap.getSite(Location(x, y)).production
prodmap = prodmap / 255 # normalize

# set up getLocation with lightweight tuples
def dloc(loc, direction=0):
    dlocs = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
    x, y = loc
    x = (x + dlocs[direction][0]) % width
    y = (y + dlocs[direction][1]) % height
    return (y, x)

def conv2d(x, filter):
    return convolve2d(x, filter, mode='same', boundary='wrap')

def gaussian(x, d=10):
    return gaussian_filter(x, (width/d, height/d), mode='wrap')

def roll(x):
    return np.stack([
        x,
        np.roll(x,  1, axis=1),
        np.roll(x, -1, axis=0),
        np.roll(x, -1, axis=1),
        np.roll(x,  1, axis=0),
    ], axis=0)

def filtnorm(x):
    return x / x.sum()

enemy_str_filter = filtnorm(np.array([
    [ 0,  0, .5,  0,  0],
    [ 0, .5,  1, .5,  0],
    [.5,  1,  1,  1, .5],
    [ 0, .5,  1, .5,  0],
    [ 0,  0, .5,  0,  0],
]))

# Production
PRODUCTION_FORCE = gaussian(prodmap, d=25)

def calc_score(t, prodmap, ownermap, strmap, myID):
    # masks
    my_units = (ownermap == myID)
    neutral_mask = (ownermap == 0)
    enemy_mask = (ownermap != myID) * (ownermap != 0)
    self_str = strmap * my_units
    progress = (neutral_mask.sum() / n_tiles)

    # Potential to attack enemy areas
    enemy_str = conv2d(enemy_mask * strmap, enemy_str_filter) * enemy_mask
    if args.debug and \
       (enemy_str < 0).any() and (enemy_str > 1).any():
        logging.debug("enemy attack map is not between 0 and 1! :(")

    # Potential to attack neutrals that are weaker
    neutral_bias = 0.1
    neutral_str = strmap * neutral_mask
    neutral_score = np.tanh(self_str - roll(neutral_str) + neutral_bias * progress) * roll(neutral_mask)
    neutral_score[0].fill(0)
    if args.debug and \
       (neutral_score < -1).any() and (neutral_score > 1).any():
        logging.debug("neutral attack map is not between -1 and 1! :(")

    # Potential to head towards areas of weaker neutrals
    neutral_pull = gaussian(neutral_mask, d=20)
    neutral_pull = neutral_pull - roll(neutral_pull)


    # Potential to move towards edges
    edge_potential = gaussian(self_str) #my_units.astype('f'))
    edge_potential = edge_potential - roll(edge_potential)
    # imsave("debug/{0:03d}.png".format(t), edge_potential)

    # Inertial
    inertial_force = np.zeros((5, width, height))
    inertial_force[0] = 0.2 / (self_str + 1e-3) - 1

    # Go towards areas of higher production
    prod_force = PRODUCTION_FORCE * (1 - my_units)

    pointwise_scores = (
        args.p_a * 3 * enemy_str + 
        args.p_p * 0.003 * prod_force
    )

    directional_scores = (
        args.p_n * neutral_score +
        args.p_i * (progress+0.1) * inertial_force +
        args.p_e * (progress)**2 * edge_potential +
        args.p_np* neutral_pull
    )

    scoremap = directional_scores + roll(pointwise_scores)
    exp = np.exp(scoremap)
    softmax = scoremap # exp / np.sum(exp, axis=0)

    return softmax


np.set_printoptions(precision=2, threshold=10000, linewidth=300)
if __name__ == '__main__':
    turn = 0
    while True:
        t = time.time()
        moves = []
        for x, y in xyiter():
            site = gameMap.getSite(Location(x, y))
            ownermap[x][y] = site.owner
            strmap[x][y] = site.strength
        strmap /= 255

        movedmap = np.zeros((width, height))

        softmax = calc_score(turn, prodmap, ownermap, strmap, myID)
        direction = softmax.argmax(axis=0)

        for loc in xyiter():
            if ownermap[loc] == myID:
                moves.append(Move(Location(*loc), direction[loc]))

        logging.debug("Time this turn: {}".format(time.time() - t))
        sendFrame(moves)
        gameMap = getFrame()
        turn += 1

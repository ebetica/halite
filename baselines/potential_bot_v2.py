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
logging.basicConfig(filename='potential_bot-v2.log',
                    filemode='w',
                    level=logging.DEBUG)
class Args:
    debug = 0
    pe = 0.40247075405984134
    pn_me = 0.4443103640390592
    pa_mp = 0.4941953538843009
    pe_ms = 0.3437945277846455
    pw = 0.22718750772594337
    pa_me = 0.30912461928545026
    pw_me = 0.7347909419789169
    pn = 0.03177362358767333
    pt_ms = 0.7121879553967283
    pp_mp = -0.16240710217436538
    pp = -0.03530139501638829
    pp_ms = 0.7271176518283937
    pi_mp = -0.16856697072599525
    pa = 0.31332968362875685
    pi = 0.6500089668827351
    pn_ms = 0.4361237844126271
    pi_ms = 0.4465364957414072
    pt_mp = 0.5388143687087323
    pp_me = 0.558310524514928
    pw_mp = 0.8194557603442666
    pa_ms = 0.018480399734157715
    pt_me = 0.33179980767857703
    pe_me = 0.49611888486617295
    pn_mp = -0.5275458384871545
    pe_mp = 0.3532332965151932
    pt = 0.22796681189604995
    pi_me = 0.5285601224619615
    pw_ms = 0.3405191430723605

args = Args()


myID, gameMap = getInit()
sendInit("PotentialBot-v2")
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
    return (x, y)

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

def calc_score(turn, prodmap, ownermap, strmap, myID):
    # masks
    my_units = (ownermap == myID)
    neutral_mask = (ownermap == 0)
    enemy_mask = (ownermap != myID) * (ownermap != 0)
    self_str = strmap * my_units
    progress = (neutral_mask.sum() / n_tiles)
    time = turn / tot_turns

    # Potential to attack enemy areas
    enemy_str = conv2d(enemy_mask * strmap, enemy_str_filter) * enemy_mask
    if args.debug and \
       (enemy_str < 0).any() and (enemy_str > 1).any():
        logging.debug("enemy attack map is not between 0 and 1! :(")

    # Potential to attack neutrals that are weaker
    neutral_str = strmap * neutral_mask
    neutral_score = np.tanh(3 * (self_str - roll(neutral_str))) * roll(neutral_mask)
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
    # imsave("debug/{0:03d}.png".format(turn), edge_potential)

    # Inertial
    inertial_force = np.zeros((5, width, height))
    inertial_force[0] = 0.2 / (self_str + 1e-3) - 1

    # Go towards areas of higher production
    prod_force = PRODUCTION_FORCE * (1 - my_units)

    # Maximize territory as time goes on
    area_force = gaussian((ownermap != myID).astype('f'), d=20)

    area_mod = lambda s, e, p: (e - s) * progress**p + s
    time_mod = lambda s, e, p: (e - s) * time**p + s

    pointwise_scores = (
        args.pa * 3    * area_mod(args.pa_ms, args.pa_me, args.pa_mp) * enemy_str +
        args.pp * 0.01 * area_mod(args.pp_ms, args.pp_me, args.pp_mp) * prod_force +
        args.pt * 0.02 * time_mod(args.pt_ms, args.pt_me, args.pt_mp)    * area_force
    )
    logging.debug(pointwise_scores.max())
    logging.debug(area_force.max())

    directional_scores = (
        args.pw * area_mod(args.pw_ms, args.pw_me, args.pw_mp) * neutral_score +
        args.pi * area_mod(args.pi_ms, args.pi_me, args.pi_mp) * inertial_force +
        args.pe * area_mod(args.pe_ms, args.pe_me, args.pe_mp) * edge_potential +
        args.pn * area_mod(args.pn_ms, args.pn_me, args.pn_mp) * neutral_pull
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
                best = -1000
                bestd = direction[loc]
                for d in CARDINALS:
                    nloc = dloc(loc, d)
                    if ownermap[nloc] != myID and strmap[nloc] < strmap[loc]:
                        if softmax[d][loc] > best:
                            best = softmax[d][loc]
                            bestd = d
                moves.append(Move(Location(*loc), bestd))

        logging.debug("Time this turn: {}".format(time.time() - t))
        sendFrame(moves)
        gameMap = getFrame()
        turn += 1

from hlt import *
from networking import *
from random import choice, shuffle
from itertools import product
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import logging
import numpy as np
import time


# Initial setup
logging.basicConfig(filename='potential_bot-v2.log',
                    filemode='w',
                    level=logging.DEBUG)
class Args:
    debug = 0
    pw_mp = np.exp(1.155825091012094)
    pe_ms = 0.30204359919919826
    pt_ms = 0.7003277996515748
    pp_ms = 0.2935446916022964
    pe = 0.2159553284908694
    pw_me = 0.6342752270086478
    pi_mp = np.exp(1.2171643670962782)
    pn_mp = np.exp(-0.3057085553137206)
    pe_me = 0.6607303453032115
    pt_mp = np.exp(0.8803560858218601)
    pe_mp = np.exp(0.8453680549732252)
    pa_mp = np.exp(-1.053165369028016)
    pa = 0.5545709996491105
    pw = 0.582967587491569
    pt_me = 0.40181280509518325
    pp_mp = np.exp(0.06494058989250073)
    pi_me = 0.03915556497109551
    pi_ms = 0.277060966580303
    pi = 0.2669097969851137
    pn = 0.6231141705884672
    pn_me = 0.1493725369979343
    pn_ms = 0.40553374977970347
    pp = 0.4222831886487678
    pt = 0.4754144686961997
    pw_ms = 0.7113280485941802
    pa_ms = 0.6643447321150748
    pp_me = 0.5752116995383341
    pa_me = 0.7327941422739227

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

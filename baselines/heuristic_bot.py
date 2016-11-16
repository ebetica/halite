from hlt import *
from networking import *
from random import choice
from itertools import product
import logging
import sys
import numpy as np
import time


# Initial setup
'''
logging.basicConfig(filename='heuristic_bot.log',
                    filemode='w',
                    level=logging.DEBUG)

'''
myID, gameMap = getInit()
sendInit("HeuristicBot")
gameMap = getFrame()
width, height = gameMap.width, gameMap.height
prodmap = np.zeros((height, width))
ownermap = np.zeros((height, width)).astype('i')
strmap = np.zeros((height, width))
rstrmap = np.zeros((height, width))

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

turn = 0
while True:
    t = time.time()
    moves = []
    for y, x in xyiter():
        site = gameMap.getSite(Location(x, y))
        ownermap[y][x] = site.owner
        strmap[y][x] = site.strength
    strmap /= 255
    f = (100 / width / height)
    rstrmap = (ownermap == myID) * strmap * f + rstrmap * (1 - f)

    for loc in xyiter():
        if ownermap[loc] == myID:
            movedPiece = False
            for d in CARDINALS:
                nloc = dloc(loc, d)
                if ownermap[nloc] != myID and strmap[nloc] < strmap[loc]:
                    moves.append(Move(Location(loc[1], loc[0]), d))
                    movedPiece = True
                    break

            if not movedPiece and strmap[loc] < prodmap[loc] * 5:
                moves.append(Move(Location(loc[1], loc[0]), STILL))
                movedPiece = True

            hasEnemy = False
            best = -1000
            bestd = 0
            for d in CARDINALS:
                nloc = dloc(loc, d)
                nstr = strmap[nloc]
                nprod = prodmap[nloc]

                enemy_score = nprod + (1 - nstr) + 10  # always prefer to pwn enemies
                self_score = (1 - rstrmap[nloc])
                if ownermap[nloc] != myID:
                    hasEnemy = True
                if ownermap[nloc] != myID and enemy_score > best:
                    best = enemy_score
                    bestd = d
                if ownermap[nloc] == myID and self_score > best:
                    best = self_score
                    bestd = d
            if not movedPiece: # move piece to weakest tile
                moves.append(Move(Location(loc[1], loc[0]), int(bestd)))
                movedPiece = True


    sendFrame(moves)
    gameMap = getFrame()
    turn += 1

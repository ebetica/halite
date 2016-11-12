from hlt import *
from networking import *

myID, gameMap = getInit()
sendInit("MyPythonBot")

while True:
    moves = []
    gameMap = getFrame()
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            if gameMap.getSite(Location(x, y)).owner == myID:
                movedPiece = False
                if not movedPiece and gameMap.getSite(Location(y, x)).strength == 0:
                    moves.append(Move(Location(x, y), STILL))
                    movedPiece = True;
                if not movedPiece:
                    moves.append(Move(Location(x, y), int(random.random() * 5)))
                    movedPiece = True
    sendFrame(moves)

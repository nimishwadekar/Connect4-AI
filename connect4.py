from copy import deepcopy
import enum
import math
import random
import os
import gzip
from collections import defaultdict

log = open('log', 'w')
log.truncate(0)

QL_DAT_FILE = 'ql.dat'

class GameState(enum.Enum):
    PLAYER_1_WIN = 1
    PLAYER_2_WIN = -1
    DRAW = 2
    CONTINUE = 3


TerminalStatesCache = dict()
TerminalStatesCache[0] = set() # Draw
TerminalStatesCache[1] = set() # Player 1 Win
TerminalStatesCache[-1] = set() # Player 2 Win
TerminalStatesCache[2] = set() # Continue

class Connect4:
    # Player1 and Player2 are the algorithms (functions).
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns

        self.Board = [[0] * columns for i in range(rows)]
        self.Tops = [rows] * columns
        
        self.MAX_TURNS = rows * columns
        self.Turn = 0
        self.lastMove = -1 #column
        self.encodedValue = self._encodeState()


    def __hash__(self):
        return self.encodedValue


    # insertionPoint - 2-tuple for position. Top-left position is (0, 0).
    def checkRowWin(self, player, insertionPoint):
        if self.columns < 4:
            return False

        point = [insertionPoint[0], 0]
        lineSum = 0
        sumToWin = player * 4

        for i in range(4):
            lineSum += self.Board[point[0]][point[1]]
            point[1] += 1
        if lineSum == sumToWin:
            return True
        
        for i in range(self.columns - 4):
            lineSum += self.Board[point[0]][point[1]] - self.Board[point[0]][point[1] - 4]
            point[1] += 1
            if lineSum == sumToWin:
                return True

        return False


    # insertionPoint - 2-tuple for position. Top-left position is (0, 0).
    def checkColumnWin(self, player, insertionPoint):
        if self.rows < 4:
            return False

        point = [0, insertionPoint[1]]
        lineSum = 0
        sumToWin = player * 4

        for i in range(4):
            lineSum += self.Board[point[0]][point[1]]
            point[0] += 1
        if lineSum == sumToWin:
            return True

        for i in range(self.rows - 4):
            lineSum += self.Board[point[0]][point[1]] - self.Board[point[0] - 4][point[1]]
            point[0] += 1
            if lineSum == sumToWin:
                return True

        return False


    # insertionPoint - 2-tuple for position. Top-left position is (0, 0).
    def checkLeftDiagonalWin(self, player, insertionPoint):
        if self.rows < 4 or self.columns < 4:
            return False

        if insertionPoint[0] > insertionPoint[1]:
            point = [insertionPoint[0] - insertionPoint[1], 0]
        else:
            point = [0, insertionPoint[1] - insertionPoint[0]]
        lineSum = 0
        base = [0, 0]
        base[0], base[1] = point[0], point[1]
        sumToWin = player * 4

        if not (point[0] <= self.rows - 4 and point[1] <= self.columns - 4):
            return False

        for i in range(4):
            lineSum += self.Board[point[0]][point[1]]
            point[0] += 1
            point[1] += 1
        if lineSum == sumToWin:
            return True

        while point[0] < self.rows and point[1] < self.columns:
            lineSum += self.Board[point[0]][point[1]] - self.Board[point[0] - 4][point[1] - 4]
            point[0] += 1
            point[1] += 1
            if lineSum == sumToWin:
                return True

        return False


    # insertionPoint - 2-tuple for position. Top-left position is (0, 0).
    def checkRightDiagonalWin(self, player, insertionPoint):
        if self.rows < 4 or self.columns < 4:
            return False

        if self.rows - 1 - insertionPoint[0] > insertionPoint[1]:
            point = [insertionPoint[0] + insertionPoint[1], 0]
        else:
            point = [self.rows - 1, insertionPoint[1] - (self.rows - 1 - insertionPoint[0])]
        lineSum = 0
        base = [0, 0]
        base[0], base[1] = point[0], point[1]
        sumToWin = player * 4

        if not (point[0] >= 3 and point[1] <= self.columns - 4):
            return False

        for i in range(4):
            lineSum += self.Board[point[0]][point[1]]
            point[0] -= 1
            point[1] += 1
        if lineSum == sumToWin:
            return True

        while point[0] >= 0 and point[1] < self.columns:
            lineSum += self.Board[point[0]][point[1]] - self.Board[point[0] + 4][point[1] - 4]
            point[0] -= 1
            point[1] += 1
            if lineSum == sumToWin:
                return True

        return False


    # Returns enum GameState.
    def insert(self, player, column):
        if self.Tops[column] == 0:
            raise Exception('Full column')

        self.Tops[column] -= 1
        point = (self.Tops[column], column)
        self.Board[point[0]][point[1]] = player
        
        self.Turn += 1
        self.lastMove = column
        self.encodedValue = self._encodeState()

        if self.encodedValue in TerminalStatesCache[player]:
            return GameState.PLAYER_1_WIN if player == 1 else GameState.PLAYER_2_WIN
        if self.encodedValue in TerminalStatesCache[0]:
            return GameState.DRAW
        if self.encodedValue in TerminalStatesCache[2]:
            return GameState.CONTINUE

        if self.checkRowWin(player, point) or self.checkColumnWin(player, point) or self.checkLeftDiagonalWin(player, point) or self.checkRightDiagonalWin(player, point):
            TerminalStatesCache[player].add(self.encodedValue)
            return GameState.PLAYER_1_WIN if player == 1 else GameState.PLAYER_2_WIN
        
        if self.Turn == self.MAX_TURNS:
            TerminalStatesCache[0].add(self.encodedValue)
            return GameState.DRAW

        TerminalStatesCache[2].add(self.encodedValue)
        return GameState.CONTINUE


    # Returns encoded value of insert state
    def insertTemporarily(self, player, column):
        if self.Tops[column] == 0:
            raise Exception('Full column')

        self.Tops[column] -= 1
        point = (self.Tops[column], column)
        self.Board[point[0]][point[1]] = player
        
        value = self._encodeState()

        self.Board[point[0]][point[1]] = 0
        self.Tops[column] += 1

        return value


    # Encodes the current state of the game as a number.
    def _encodeState(self):
        encoded = 0
        for i, row in enumerate(self.Board):
            for j, e in enumerate(row):
                encoded += (2 if e == -1 else e) * 3 ** (i * self.columns + j)
        
        return encoded


# Returns a 2-tuple of the decoded board and column tops.
def decodeState(encodedState, rows, columns):
    encodedTops = encodedState // 3 ** (rows * columns)
    encodedBoard = encodedState % 3 ** (rows * columns)
    
    decodedBoard = [[0] * columns for i in range(rows)]
    decodedTops = [0] * columns

    for i in range(rows):
        for j in range(columns):
            decodedBoard[i][j] = encodedBoard % 3
            encodedBoard //= 3

    for i in range(columns):
        decodedTops[i] = encodedTops % (rows + 1)
        encodedTops //= (rows + 1)

    return decodedBoard, decodedTops


class Node:
    def __init__(self, board, player, winState):
        self.state: Connect4 = board
        self.player = player
        self.winState = winState

        self.children = set()
        self.hasBeenExplored = False

        self.rewards = 0
        self.visits = 0

    
    def __hash__(self):
        return self.state.encodedValue


    def __str__(self):
        s = 'Player ' + str(2 if self.player == -1 else self.player) + '\n'
        s += '\n'.join(' '.join('2' if x == -1 else str(x) for x in row) for row in self.state.Board)
        s += '\n'
        return s


    def isTerminalState(self):
        return self.winState != GameState.CONTINUE


    def createChildNodes(self):
        if self.children or self.winState != GameState.CONTINUE:
            return

        nextPlayer = -self.player
        for col, top in enumerate(self.state.Tops):
            if top == 0:
                continue

            child = deepcopy(self.state)
            winState = child.insert(nextPlayer, col)
            self.children.add(Node(child, nextPlayer, winState))


    def hasCreatedChildren(self):
        return len(self.children) != 0


    def getRandomChild(self):
        nextPlayer = -self.player
        tops = [i for i, top in enumerate(self.state.Tops) if top > 0]
        if not tops:
            return None

        col = random.choice(tops)
        child = deepcopy(self.state)
        winState = child.insert(nextPlayer, col)
        return Node(child, nextPlayer, winState)


    def getReward(self, playingPlayer):
        if self.winState == GameState.CONTINUE:
            return 0

        if self.winState == GameState.DRAW:
            return 1

        if (self.winState == GameState.PLAYER_1_WIN and playingPlayer == 1) or (self.winState == GameState.PLAYER_2_WIN and playingPlayer == -1):
            return 2

        return -2


def findInsertedColumn(parent: Node, child: Node):
    for i, (top1, top2) in enumerate(zip(parent.state.Tops, child.state.Tops)):
        if top1 != top2:
            return i
    raise Exception('Same boards')


class Player:
    def __init__(self):
        pass

    def playTurn(self):
        pass

    def updatePlayerNode(self, newNode: Node):
        pass


class MCTS_Player(Player):
    def __init__(self, game, startingPlayer, playouts, uctExploration):
        self.playouts = playouts
        self.uctExploration = uctExploration
        self.startingPlayer = startingPlayer
        self.currentNode: Node = Node(deepcopy(game), -startingPlayer, GameState.CONTINUE)


    def playTurn(self):
        for i in range(self.playouts):
            self.doPlayout()
        return self.makeMove()


    def updatePlayerNode(self, newNode: Node):
        encoded = newNode.state.encodedValue
        for child in self.currentNode.children:
            if encoded == child.state.encodedValue:
                self.currentNode = child
                return
        self.currentNode = Node(deepcopy(newNode.state), newNode.player, newNode.winState)


    def doPlayout(self):
        path = self._select()
        node = path[-1]
        self._expand(node)
        reward = self._simulate(node)
        self._backpropagate(path, reward)


    def makeMove(self):
        if not self.currentNode.children:
            self.currentNode = self.currentNode.getRandomChild()
            return self.currentNode

        def getNodeValue(node: Node):
            return node.rewards / node.visits if node.visits != 0 else -1000000
        
        maxNodeValue = getNodeValue(max(self.currentNode.children, key = getNodeValue))
        self.currentNode = random.choice([node for node in self.currentNode.children if getNodeValue(node) == maxNodeValue])
        return self.currentNode


    def _uctSelect(self, node: Node):
        def uct(n: Node):
            return ((n.rewards / n.visits) + self.uctExploration * math.sqrt(math.log(node.visits) / n.visits)) if n.visits != 0 else float('inf')
        
        maxUCT = uct(max(node.children, key = uct))
        return [child for child in node.children if uct(child) == maxUCT]


    def _select(self):
        path = []
        node = self.currentNode
        while True:
            path.append(node)
            if node.winState != GameState.CONTINUE or not node.hasBeenExplored or not node.hasCreatedChildren():
                return path

            node = random.choice(self._uctSelect(node))


    def _expand(self, node: Node):
        if node.hasBeenExplored:
            return
        node.createChildNodes()
        node.hasBeenExplored = True

    
    def _simulate(self, node: Node):
        while not node.isTerminalState():
            node = node.getRandomChild()
        
        return node.getReward(self.startingPlayer)


    def _backpropagate(self, path, reward):
        rewardToGive = 0
        for node in reversed(path):
            if reward == 1:
                rewardToGive = 1
            else:
                rewardToGive = reward if node.player == self.startingPlayer else -reward

            node.visits += 2
            node.rewards += rewardToGive


QL_WIN_REWARD = 2
QL_LOSE_REWARD = -2
QL_DRAW_REWARD = 0.2
QL_CONTINUE_REWARD = -0.01

QL_DEFAULT_QVALUE = 0.01

QL_DATFILE_NAME = '2019A7PS1004G_NIMISH.dat'


class QL_Player(Player):
    def __init__(self, board: Connect4, player, epsilon, alpha, gamma, loadFromFile = True):
        self.player = player
        self.board = deepcopy(board)
        self.node: Node = Node(self.board, player, GameState.CONTINUE)
        self.qTable = defaultdict(lambda: QL_DEFAULT_QVALUE)
        self.possibleNextStatesCache = defaultdict(self._getPossibleNextStates)
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.previousAfterstate = -2
        
        if loadFromFile and os.path.exists(QL_DATFILE_NAME):
            self.loadData()

    # return node
    def playTurn(self):
        action = self._chooseAction()
        gameState = self._takeAction(action)

        if gameState != GameState.CONTINUE:
            self._updateQ(self.previousAfterstate, 0, self._getReward(gameState))

        self.previousAfterstate = self.board.encodedValue
        return self.node


    def updatePlayerNode(self, newNode: Node):
        action = newNode.state.lastMove
        gameState = self.board.insert(-self.player, action)

        if gameState != GameState.CONTINUE:
            nextStateValue = 0
        else:
            nextStateValue = max(map(lambda x: self.qTable[x[1]], self.possibleNextStatesCache[self.board.encodedValue]))

        self._updateQ(self.previousAfterstate, nextStateValue, self._getReward(gameState))


    def _chooseAction(self):
        possibleNextStates = self.possibleNextStatesCache[self.board.encodedValue]
        if random.random() < self.epsilon:
            return random.choice(possibleNextStates)[0]
        else:
            return max(possibleNextStates, key = lambda e: self.qTable[e[1]])[0]


    # (action, next state)
    def _getPossibleNextStates(self):
        if self.board.encodedValue in self.possibleNextStatesCache:
            return self.possibleNextStatesCache[self.board.encodedValue]
        
        nextStates = []
        for i, top in enumerate(self.board.Tops):
            if top == 0:
                continue
            nextStates.append((i, self.board.insertTemporarily(self.player, i)))
        self.possibleNextStatesCache[self.board.encodedValue] = nextStates
        return nextStates


    # Returns reward
    def _takeAction(self, action):
        gameState = self.board.insert(self.player, action)
        self.node.winState = gameState
        return gameState
        
    
    def _getReward(self, gameState):
        if gameState == GameState.CONTINUE:
            return QL_CONTINUE_REWARD
        if gameState == GameState.DRAW:
            return QL_DRAW_REWARD
        if (gameState == GameState.PLAYER_1_WIN and self.player == 1) or (gameState == GameState.PLAYER_2_WIN and self.player == -1):
            return QL_WIN_REWARD
        return QL_LOSE_REWARD


    def _updateQ(self, currentState, nextStateValue, reward):
        self.qTable[currentState] = (1 - self.alpha) * self.qTable[currentState] +  \
            self.alpha * (reward + self.gamma * nextStateValue)


    def saveData(self):
        with open(QL_DATFILE_NAME, 'w') as datfile:
            datfile.truncate(0)
            for state, qvalue in self.qTable.items():
                datfile.write(str(state) + ' ' + str(qvalue) + '\n')


    def loadData(self):
        with open(QL_DATFILE_NAME, 'r') as datfile:
            for line in datfile:
                row = line.split(' ')
                state = int(row[0])
                qvalue = float(row[1])
                self.qTable[state] = qvalue


# Returns the (winner, moves).
def playGameMC(player1: Player, player2: Player, name1, name2):

    while True:
        print(name1)
        node: Node = player1.playTurn()
        print('Value of next node according to', name1, ':', (node.rewards / node.visits))
        player2.updatePlayerNode(node)
        print(node)
        if node.winState != GameState.CONTINUE:
            print('Game ended:', node.winState)
            break

        print(name2)
        node = player2.playTurn()
        print('Value of next node according to', name2, ':', (node.rewards / node.visits))
        player1.updatePlayerNode(node)
        print(node)
        if node.winState != GameState.CONTINUE:
            print('Game ended:', node.winState)
            break

    if node.winState == GameState.DRAW:
        return 0, node.state.Turn
    if node.winState == GameState.PLAYER_1_WIN:
        return 1, node.state.Turn
    if node.winState == GameState.PLAYER_2_WIN:
        return 2, node.state.Turn
    return 3, node.state.Turn


def playGameQL(player1: Player, player2: QL_Player, namemc):

    while True:
        print(namemc)
        node: Node = player1.playTurn()
        print('Value of next node according to', namemc, ':', (node.rewards / node.visits))
        player2.updatePlayerNode(node)
        print(node)
        if node.winState != GameState.CONTINUE:
            print('Game ended:', node.winState)
            break

        print('Q-L agent')
        node = player2.playTurn()
        print('Value of next state according to Q-L agent:', player2.qTable[node.state.encodedValue])
        player1.updatePlayerNode(node)
        print(node)
        if node.winState != GameState.CONTINUE:
            print('Game ended:', node.winState)
            break

    if node.winState == GameState.DRAW:
        return 0, node.state.Turn
    if node.winState == GameState.PLAYER_1_WIN:
        return 1, node.state.Turn
    if node.winState == GameState.PLAYER_2_WIN:
        return 2, node.state.Turn
    return 3, node.state.Turn
    

def main():
    print('Choose one of the following options:')
    print('1. MC200 vs MC40')
    print('2. MCn vs Q-L')
    choice = int(input('Choose 1 or 2: '))

    if choice == 1:
        player1 = MCTS_Player(Connect4(6, 5), 1, 200, math.sqrt(2))
        player2 = MCTS_Player(Connect4(6, 5), 1, 40, math.sqrt(2))
        
        for j in range(player1.playouts):
            player1.doPlayout()
        for j in range(player2.playouts):
            player2.doPlayout()

        winner, moves = playGameMC(player1, player2, 'MCTS with 200 playouts', 'MCTS with 40 playouts')
        print('Total moves:', moves)

    elif choice == 2:
        ql = QL_Player(Connect4(4, 5), -1, 0, 0, 0.8, True)

        player1 = MCTS_Player(Connect4(4, 5), 1, 25, math.sqrt(2))
        for j in range(player1.playouts):
            player1.doPlayout()

        winner, moves = playGameQL(player1, ql, 'MCTS with 25 playouts')
        print('Total moves:', moves, '\n')
        print('Maximum n = 25 and r = 4 for convergence.')

    else:
        print('Invalid option.')


if __name__=='__main__':
    main()
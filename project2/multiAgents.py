# multiAgents.py


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    legalMoves = gameState.getLegalActions()

    scores = [self.evaluationFunction(gameState, action)
                                      for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(
        len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()

    "*** YOUR CODE HERE ***"
    foodList = newFood.asList()
    score = 0
    manhattanDistance = []
    ghostDistance = []

    if successorGameState.isWin():
      score = 999999
      return score
    else:
      for i in range(0, len(foodList)):
          distance = abs(newPos[0] - foodList[i][0]) + \
                         abs(newPos[1]-foodList[i][1])
          manhattanDistance.append(distance)

      for i in range(0, len(newGhostStates)):
          ghostLoc = currentGameState.getGhostPosition(i+1)
          distance2 = abs(newPos[0] - ghostLoc[0]) + \
                          abs(newPos[1] - ghostLoc[1])
          ghostDistance.append(distance2)

      if (len(foodList) != 0):
          score += (0-len(foodList)*100)
          score += (0-min(manhattanDistance))

      ghost = min(ghostDistance)

      if (ghost <= 2):

          score = score**2
          score = -1 * score

    return score


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
    self.index = 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        def minimize(pacman, ghosts, currentState, depth):
          v = 9999999

          if depth ==0 or currentState.isLose() or currentState.isWin():
            return self.evaluationFunction(currentState)

          for x in currentState.getLegalActions(pacman):
            if pacman >= ghosts:
                v = min(v, maximize(0,ghosts,currentState.generateSuccessor(pacman,x),depth-1 ))
            else:
                v = min(v, minimize(pacman+1, ghosts, currentState.generateSuccessor(pacman,x),depth))

          return v

        def maximize(pacman, ghosts, currentState, depth):
          v = -9999999

          if depth ==0 or currentState.isLose() or currentState.isWin():
            return self.evaluationFunction(currentState)

          for x in currentState.getLegalActions(pacman):
            v = max(v, minimize(1,ghosts,currentState.generateSuccessor(pacman,x),depth))

          return v

        
        currentValue = -9999999
        chosenDirection = Directions.RIGHT

        for x in gameState.getLegalActions():
          destination = gameState.generateSuccessor(0,x)
          oldValue = currentValue
          currentValue = max(currentValue, minimize(1,gameState.getNumAgents()-1,destination,self.depth))
          if currentValue > oldValue:
            chosenDirection = x
          
        return chosenDirection

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minipacman agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minipacman action using self.depth and self.evaluationFunction
        """
        def maxValue(state, alpha, beta, depth):
            if depth + 1 or state.isLose() or state.isWin() == self.depth:
                return self.evaluationFunction(state)
            value = float('-inf')
            for move in state.getLegalActions(0):
                value = max(value, minValue(state.generateSuccessor(0, move), alpha, beta, 1, depth + 1))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        def minValue(state, alpha, beta, agent, depth):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            value = float('inf')
            for move in state.getLegalActions(agent):
                if state.getNumAgents() - 1 != agent:
                    value = min(value, minValue(state.generateSuccessor(agent, move), alpha, beta, agent + 1, depth))
                else:
                    value = min(value, maxValue(state.generateSuccessor(agent, move), alpha, beta, depth)) 
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        bestaction = Directions.RIGHT
        score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            currscore = minValue(gameState.generateSuccessor(0, action), alpha, beta, 1, 0)
            if currscore > score:
                score = currscore
                bestaction = action
            if currscore > beta:
                return bestaction
            alpha = max(alpha, currscore)
        return bestaction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
      "*** YOUR CODE HERE ***"
      def maxValue(gameState,depth):
        value = -1000000
        if gameState.isLose() or gameState.isWin() or depth == 0:
          return self.evaluationFunction(gameState) 
        else:
          for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0,action)
            value = max(value,expValue(state, depth+ 1,1))
        return value

      def expValue(gameState,depth, ghosts):
        total = 0
        expectvalue = 0
        numGhost = gameState.getNumAgents() - 1
        counter = len(gameState.getLegalActions(ghosts))
        if gameState.isLose() or gameState.isWin():
          return self.evaluationFunction(gameState) 
        else:
          for action in gameState.getLegalActions(ghosts):
            state = gameState.generateSuccessor(ghosts,action)
            if ghosts == numGhost:
              expectvalue = maxValue(state,depth)
            else:
              expectvalue = expValue(state,depth,ghosts+1)
            total += expectvalue
          return float(total)/float(counter)

      score = -1000000
      expvalue = 0
      for action in gameState.getLegalActions(0):
        state = gameState.generateSuccessor(0,action)
        expvalue = expValue(state, 0, 1)
        if expvalue > score:
          score = expvalue
          bestAction = action
      return bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <Instead of evaluating actions in reflex agent function, the
      evaluation function here should evaluate states. In the function, I considered
      the minimal distance to food, number of food and capsules, and nearest Ghost.
      The evaluation is modified from question 1.>
    """
    "*** YOUR CODE HERE ***"
    food = currentGameState.getFood()
    ghoststate = currentGameState.getGhostStates()
    position = currentGameState.getPacmanPosition()    

    if food.asList():
        closefood = min([manhattanDistance(position, food) for food in food.asList()])
    else:
        closefood = 0
    closeghost = min([manhattanDistance(position, ghostState.getPosition()) for ghostState in ghoststate])

    if closeghost:
        ghostdistance = 1 / closeghost
    else:
        ghostdistance = 100

    a = -1
    b = -1000
    c=d = -10
    evaluation = currentGameState.getScore() + a * closefood + b * currentGameState.getNumFood() + c * ghostdistance + d * len(currentGameState.getCapsules())

    return evaluation
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# search.py

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    "*** YOUR CODE HERE ***"
    fringeLocations = util.Stack()
    startingPoint = (problem.getStartState(), {})
    fringeLocations.push(startingPoint)
    vistedLocations = set([])

    while True:
        if fringeLocations.isEmpty():
            return 'Fail'
        else:
            nextLocation = fringeLocations.pop()

        if not nextLocation[0] in vistedLocations:
            vistedLocations.add(nextLocation[0])
            successor = problem.getSuccessors(nextLocation[0])

            for i in range(len(successor)):
                path = list(nextLocation[1])
                path.append(successor[i][1])
                fringeLocations.push((successor[i][0], path))

        if problem.isGoalState(nextLocation[0]):
            fringeLocations.push(nextLocation[0])
            return nextLocation[1]

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    # Same setup as DFS
    fringeLocations = util.Queue()
    startingPoint = (problem.getStartState(), {})
    fringeLocations.push(startingPoint)
    vistedLocations = set([])

    while True:
        # Same setup as DFS
        if fringeLocations.isEmpty():
            return 'Fail'
        else:
            nextLocation = fringeLocations.pop()
        # Same setup as DFS
        if not nextLocation[0] in vistedLocations:
            vistedLocations.add(nextLocation[0])
            successor = problem.getSuccessors(nextLocation[0])

            for i in range(len(successor)):
                path = list(nextLocation[1])
                path.append(successor[i][1])
                fringeLocations.push((successor[i][0], path))
                if problem.isGoalState(nextLocation[0]):
                    return nextLocation[1]

    util.raiseNotDefined()

def uniformCostSearch(problem):

    fringeLocations = util.PriorityQueue()
    startingPoint = problem.getStartState()
    fringeLocations.push((startingPoint, [], 0), 0)
    visitedLocations = []

    while not fringeLocations.isEmpty():
        state, actions, pathcost = fringeLocations.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visitedLocations:

            visitedLocations.append(state)

            successors = problem.getSuccessors(state)

            for node, directions, cost in successors:
                # add the front cost and back cost
                fringeLocations.push((node, actions + [directions], pathcost + cost), pathcost + cost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    fringeLocations = util.PriorityQueue()
    startingPoint = (problem.getStartState(), {})
    startEstCost = heuristic(problem.getStartState(), problem)
    fringeLocations.push(startingPoint, problem.getCostOfActions(startingPoint[1]) + startEstCost )
    vistedLocations = set([])

    while True:
        if fringeLocations.isEmpty():
            return 'Fail'
        else:
            nextNode = fringeLocations.pop()
        if problem.isGoalState(nextNode[0]):
            return nextNode[1]
        if not nextNode[0] in vistedLocations:
            vistedLocations.add(nextNode[0])
            successors = problem.getSuccessors(nextNode[0])
            for i in range(len(successors)):
                path = list(nextNode[1])
                path.append(successors[i][1])
                fringeLocations.push((successors[i][0], path), problem.getCostOfActions(path) + heuristic(successors[i][0], problem))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

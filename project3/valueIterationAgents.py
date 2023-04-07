
# valueIterationAgents.py


import mdp
import util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """

  def __init__(self, mdp, discount=0.9, iterations=100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter()  # A Counter is a dict with default 0

    "*** YOUR CODE HERE ***"
    for x in range(0, self.iterations):
            totalVals = util.Counter()
            allStates = self.mdp.getStates()
            for states in allStates:
                terminalState = mdp.isTerminal(states)
                if not terminalState:
                    possActions = mdp.getPossibleActions(states)
                    tempVal = util.Counter()
                    for action in possActions:
                        tempVal[action] = self.getQValue(
                            states, action)
                    totalVals[states] = max(tempVal.values())
            self.values = totalVals

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    value = 0
    stateActionPair = self.mdp.getTransitionStatesAndProbs(state,action)
    for nextState in stateActionPair:
        probability  = nextState[1]
        tempState = nextState[0]
        reward = self.mdp.getReward(state,action,tempState)
        value += probability * (reward + self.discount*self.values[tempState])
    return value

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    if len(self.mdp.getPossibleActions(state)) == 0:
      return None
    if self.mdp.isTerminal(state):
      return None
    actions = self.mdp.getPossibleActions(state)
    bestAction = None
    value = float("-inf")
    for x in actions:
      tmp = self.getQValue(state, x)
      if (tmp >= value) or (value == 0.0 and x == ""):
        bestAction = x
        value = tmp
    return bestAction

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)

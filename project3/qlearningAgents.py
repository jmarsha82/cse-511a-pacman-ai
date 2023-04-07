# qlearningAgents.py


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.qValue = util.Counter()

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    return self.qValue[(state, action)]


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    Q_value = []
    legalActions = self.getLegalActions(state)
    for action in legalActions:
      Q_value.append(self.getQValue(state, action))
    if len(legalActions) != 0:
      return max(Q_value)
    else: 
      return 0.0
    

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    bestQvalue = 0
    bestAction = None
    legalActions = self.getLegalActions(state)

    for action in legalActions:
      eachQvalue = self.getQValue(state,action)
      if eachQvalue > bestQvalue or bestAction is None:
        bestQvalue  = eachQvalue
        bestAction  = action
    return bestAction

  

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    if util.flipCoin(self.epsilon):
      return random.choice(legalActions)
    else:
      return self.getPolicy(state)



  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    NextSate = self.getLegalActions(nextState)

    if len(NextSate) == 0:
      sample = reward
    else:
      fReward = max([self.getQValue(nextState, nextAction) for nextAction in self.getLegalActions(nextState)])
      sample = reward + (self.discount * fReward)

    self.qValue[(state, action)] = (1 - self.alpha) * self.getQValue(state,action) + (sample * self.alpha)



class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    #  wYou mightant to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter()
    self.weights = 0

  def getWeights(self):
    return self.weights

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    qValue = 0
    feature = self.featExtractor.getFeatures(state, action)

    for f in feature:
      qValue += self.weights[f] * feature[f]
    return qValue



  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    feature = self.featExtractor.getFeatures(state, action)
    NextState = self.getLegalActions(nextState)
    
    for f in feature:
      difference_reward = 0
      if len(NextState) == 0:
        difference_reward = reward - self.getQValue(state, action)
      else:
        fReward = max([self.getQValue(nextState, nextAction) for nextAction in self.getLegalActions(nextState)])
        difference_reward = (reward + self.discount * fReward)
      self.weights[f] = self.weights[f] + self.alpha * difference_reward * feature[f]



  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass

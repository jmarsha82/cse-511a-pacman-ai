import math

from conftest import import_from_project


class TinyMdp:
    terminal = "terminal"

    def getStates(self):
        return ["start", "middle", self.terminal]

    def getPossibleActions(self, state):
        return {
            "start": ["fast", "slow"],
            "middle": ["finish"],
            self.terminal: [],
        }[state]

    def getTransitionStatesAndProbs(self, state, action):
        transitions = {
            ("start", "fast"): [("terminal", 1.0)],
            ("start", "slow"): [("middle", 1.0)],
            ("middle", "finish"): [("terminal", 1.0)],
        }
        return transitions.get((state, action), [])

    def getReward(self, state, action, next_state):
        rewards = {
            ("start", "fast", "terminal"): 1,
            ("start", "slow", "middle"): 0,
            ("middle", "finish", "terminal"): 5,
        }
        return rewards[(state, action, next_state)]

    def isTerminal(self, state):
        return state == self.terminal


def test_value_iteration_prefers_discounted_longer_reward():
    value_agents = import_from_project("project3", "valueIterationAgents")

    agent = value_agents.ValueIterationAgent(TinyMdp(), discount=0.9, iterations=5)

    assert agent.getPolicy("start") == "slow"
    assert math.isclose(agent.getValue("start"), 4.5)
    assert agent.getPolicy("terminal") is None


def test_q_learning_update_and_terminal_action_handling():
    qlearning = import_from_project("project3", "qlearningAgents")
    action_map = {"s0": ["left", "right"], "s1": ["finish"], "terminal": []}
    agent = qlearning.QLearningAgent(
        actionFn=lambda state: action_map[state],
        epsilon=0.0,
        alpha=0.5,
        gamma=0.8,
        numTraining=0,
    )

    agent.update("s1", "finish", "terminal", 10)
    agent.update("s0", "right", "s1", 2)

    assert agent.getAction("terminal") is None
    assert agent.getPolicy("s1") == "finish"
    assert agent.getQValue("s1", "finish") == 5
    assert agent.getQValue("s0", "right") == 3


def test_q_learning_value_policy_and_exploration(monkeypatch):
    qlearning = import_from_project("project3", "qlearningAgents")
    agent = qlearning.QLearningAgent(
        actionFn=lambda state: {"s0": ["left", "right"], "terminal": []}[state],
        epsilon=1.0,
        alpha=0.5,
        gamma=0.8,
        numTraining=0,
    )
    agent.qValue[("s0", "left")] = -1
    agent.qValue[("s0", "right")] = -2
    monkeypatch.setattr(qlearning.random, "choice", lambda actions: actions[1])

    assert agent.getValue("terminal") == 0.0
    assert agent.getValue("s0") == -1
    assert agent.getPolicy("s0") == "left"
    assert agent.getAction("s0") == "right"


def test_pacman_q_agent_records_last_action():
    qlearning = import_from_project("project3", "qlearningAgents")
    agent = qlearning.PacmanQAgent(
        actionFn=lambda state: {"s0": ["east"]}[state],
        epsilon=0.0,
        alpha=0.5,
        gamma=0.8,
        numTraining=0,
    )

    assert agent.getAction("s0") == "east"
    assert agent.lastState == "s0"
    assert agent.lastAction == "east"


def test_approximate_q_agent_maintains_counter_weights():
    qlearning = import_from_project("project3", "qlearningAgents")
    agent = qlearning.ApproximateQAgent(
        actionFn=lambda state: {"s0": ["east"], "terminal": []}[state],
        epsilon=0.0,
        alpha=0.25,
        gamma=0.9,
        numTraining=0,
    )

    agent.update("s0", "east", "terminal", 8)

    assert agent.getWeights()[("s0", "east")] == 2
    assert agent.getQValue("s0", "east") == 2

from conftest import import_from_project


class GraphProblem:
    def __init__(self):
        self.edges = {
            "S": [("A", "to-a", 2), ("B", "to-b", 1)],
            "A": [("G", "a-to-g", 2)],
            "B": [("C", "b-to-c", 1), ("G", "b-to-g", 10)],
            "C": [("G", "c-to-g", 1)],
            "G": [],
        }

    def getStartState(self):
        return "S"

    def isGoalState(self, state):
        return state == "G"

    def getSuccessors(self, state):
        return self.edges[state]

    def getCostOfActions(self, actions):
        state = "S"
        total = 0
        for action in actions:
            for successor, edge_action, cost in self.edges[state]:
                if edge_action == action:
                    state = successor
                    total += cost
                    break
            else:
                raise AssertionError("illegal action %s from %s" % (action, state))
        return total


class NoSolutionProblem(GraphProblem):
    def __init__(self):
        self.edges = {"S": [("A", "to-a", 1)], "A": []}


def test_breadth_first_search_returns_fewest_actions():
    search = import_from_project("project1", "search")

    assert search.breadthFirstSearch(GraphProblem()) == ["to-a", "a-to-g"]


def test_depth_first_search_reaches_goal():
    search = import_from_project("project1", "search")

    actions = search.depthFirstSearch(GraphProblem())

    assert actions in (["to-a", "a-to-g"], ["to-b", "b-to-g"], ["to-b", "b-to-c", "c-to-g"])
    assert GraphProblem().getCostOfActions(actions) >= 3


def test_uniform_cost_search_returns_lowest_cost_path():
    search = import_from_project("project1", "search")

    actions = search.uniformCostSearch(GraphProblem())

    assert actions == ["to-b", "b-to-c", "c-to-g"]
    assert GraphProblem().getCostOfActions(actions) == 3


def test_search_algorithms_report_no_solution():
    search = import_from_project("project1", "search")

    assert search.breadthFirstSearch(NoSolutionProblem()) == []
    assert search.uniformCostSearch(NoSolutionProblem()) == []
    assert search.aStarSearch(NoSolutionProblem()) == "Fail"
    assert search.nullHeuristic("S") == 0


def test_a_star_search_uses_heuristic_and_returns_optimal_path():
    search = import_from_project("project1", "search")
    heuristic = lambda state, problem: {"S": 2, "A": 2, "B": 1, "C": 1, "G": 0}[state]

    assert search.aStarSearch(GraphProblem(), heuristic) == ["to-b", "b-to-c", "c-to-g"]


def test_counter_helpers_are_python3_compatible():
    util = import_from_project("project1", "util")
    counts = util.Counter()
    counts["first"] = -2
    counts["second"] = 4
    counts["third"] = 1

    assert counts.argMax() == "second"
    assert counts.sortedKeys() == ["second", "third", "first"]

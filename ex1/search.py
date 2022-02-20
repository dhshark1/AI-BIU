#Daniel Haber
#322230020
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Stack()
    closed_list = []
    frontier.push([problem.getStartState(), [], 0])
    while not frontier.isEmpty():
        state = frontier.pop()
        path = state[1]
        if problem.isGoalState(state[0]):
            return path
        if state[0] not in closed_list:
            closed_list.append(state[0])
            childrens = problem.getSuccessors(state[0])
            for child in list(childrens):
                frontier.push([child[0], path+[child[1]], child[2]])
    return None


    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    closed_list = []
    frontier.push([problem.getStartState(), [], 0])
    while not frontier.isEmpty():
        state = frontier.pop()
        path = state[1]
        if problem.isGoalState(state[0]):
            return path
        if state[0] not in closed_list:
            closed_list.append(state[0])
            childrens = problem.getSuccessors(state[0])
            for child in list(childrens):
                frontier.push([child[0], path + [child[1]], child[2]])
    return None

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    frontier.push([problem.getStartState() ,[], 0], 0)
    closed_list = []
    while not frontier.isEmpty():
        state = frontier.pop()
        path = state[1]
        if problem.isGoalState(state[0]):
            return path
        if state[0] not in closed_list:
            closed_list.append(state[0])
            children = problem.getSuccessors(state[0])
            for child in list(children):
                if child[0] not in closed_list:
                    cost = state[2] + child[2]
                    n = [child[0], path + [child[1]], cost]
                    frontier.push(n, cost)
                elif child in closed_list:
                    cost = state[2] + child[2]
                    n = [child[0], path + [child[1]], cost]
                    frontier.update(n, cost)

    return None

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    def f(n):
        return n[2] + heuristic(n[0], problem)

    frontier = util.PriorityQueueWithFunction(f)
    frontier.push([problem.getStartState(), [], heuristic(problem.getStartState(), problem)])
    closed_list = []
    while not frontier.isEmpty():
        state = frontier.pop()
        path = state[1]
        if problem.isGoalState(state[0]):
            return path
        if state[0] not in closed_list:
            closed_list.append(state[0])
            children = problem.getSuccessors(state[0])
            for child in list(children):
                if child[0] not in closed_list and frontier.heap.count(child) == 0:
                    cost = state[2] + child[2]
                    node = [child[0], path + [child[1]], cost]
                    frontier.push(node)
                elif frontier.heap.count(child) != 0:
                    cost = state[2] + child[2]
                    node = [child[0], path + [child[1]], cost]
                    frontier.update(node, f(node))

    return None
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

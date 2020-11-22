# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, sys

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
        # Collect legal moves and successor states
        legalAction = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalAction]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print "New done position:", legalAction[chosenIndex]

        return legalAction[chosenIndex]

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        # print "New position:", newPos

        newFood = successorGameState.getFood()
        # print "New food:", newFood

        newGhostStates = successorGameState.getGhostStates()
        # print "New gost state:", newPos

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print "New scared times:", newScaredTimes

        "*** YOUR CODE HERE ***"
        minGhostDis = 1000000
        FoodDistantion = 1000000
        score = 0
        curFood = currentGameState.getFood()
        newFoodPositions = curFood.asList()           # we store on a list the position of the new food
        FoodDistantion = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodPositions]    # we try to reduce to the minimum the dist between the pacman and the food  # we try to reduce to the minimum the dist between the pacman and the food 
      
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates if ghostState.scaredTimer == 0]
        closestGhost = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions])
        if closestGhost<=1:
            score = 10000

        return -min(FoodDistantion)-score
        # return successorGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** Here we invoce the MinimaxImplementation of Minimax***"
        return self.MinimaxImplementation(gameState, 1, 0)

    def MinimaxImplementation(self, gameState, depth, agentIndex):
      "check the terminal node"
      if depth > self.depth or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      
      "here you can see the implementation of algorithm"
      legalAction = [action for action in gameState.getLegalActions(agentIndex) if action!='Stop']
      
      # update with next depth
      nextIndex = agentIndex + 1
      nextDepth = depth
      if nextIndex >= gameState.getNumAgents():
          nextIndex = 0
          nextDepth += 1
      
      # Choose one of the best actions or keep query the minimax result
      results = [self.MinimaxImplementation( gameState.generateSuccessor(agentIndex, action) ,\
                                    nextDepth, nextIndex) for action in legalAction]

      # check init state
      if agentIndex == 0 and depth == 1: 
          bestMove = max(results)
          bestIndices = [index for index in range(len(results)) if results[index] == bestMove]
          chosenIndex = random.choice(bestIndices) # Pick randomly among the best
          #print 'pacman %d' % bestMove
          return legalAction[chosenIndex]
      
      if agentIndex == 0:
          bestMove = max(results)
          return bestMove
      else:
          bestMove = min(results)
          return bestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "* YOUR CODE HERE *"
        inf = float('inf')
        action, score = self.alpha_beta(0, 0, gameState, -inf, inf)  # Get the action and score with the alpha-beta pruning
        return action  

    def alpha_beta(self, depth, index, gameState, alpha, beta):
      
      # Returns the best score by using the alpha beta algorithm
      if index >= gameState.getNumAgents():   # increase depth by one if all the agents ended their turn
          index = 0
          depth += 1

      if depth == self.depth:                               # we return evaluation function if we have reached the maximum depth
          return None, self.evaluationFunction(gameState)

      best_score, best_action = None, None

      if index == 0:  
          for action in gameState.getLegalActions(index):                   # we get the minimax score of the successor for each possible action of the pacman
              following_state = gameState.generateSuccessor(index, action)
              _, score = self.alpha_beta(depth, index + 1, following_state, alpha, beta)

              if best_score is None or score > best_score:
                  best_score = score
                  best_action = action
              alpha = max(alpha, score)     # we reupdate the variable alpha with the maximum value

              if alpha > beta:    # we prune the tree using break if alpha is bigger than beta
                  break

      else: 
          for action in gameState.getLegalActions(index):                   # we get the minimax score of the successor for each possible action of the ghost
              following_state = gameState.generateSuccessor(index, action)
              _, score = self.alpha_beta(depth, index + 1, following_state, alpha, beta)

              if best_score is None or score < best_score:
                best_score = score
                best_action = action
              beta = min(beta, score)     # we reupdate the variable alpha with the maximum value

              if beta < alpha:       # we prune the tree using break if beta is smaller than alpha
                  break

      if best_score is None:                              # if our state has not successor states we return the evaluation function
        return None, self.evaluationFunction(gameState)

      return best_action, best_score                      # otherwise we return the best_action and best_score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "* YOUR CODE HERE *"

        action, score = self.expectimax(0, 0, gameState)  # Get the action and score for pacman
        return action  

    def expectimax(self, depth, index, gameState):
        '''
        Returns the best score for an agent using the expectimax algorithm. For max player (agent_index=0), the best
        score is the maximum score among its successor states and for the min player (agent_index!=0), the best
        score is the average of all its successor states. Recursion ends if there are no successor states
        available or depth equals the max depth to be searched until.
        :param depth: the current depth of the tree (int)
        :param agent_index: index of the current agent (int)
        :param gameState: the current state of the game (GameState)
        :return: action, score
        '''

        # Returns the best score by using the expectimax algorithm

        if index >= gameState.getNumAgents():   # increase depth by one if all the agents ended their turn
            index = 0
            depth += 1

        if depth == self.depth:                               # we return evaluation function if we have reached the maximum depth
            return None, self.evaluationFunction(gameState)

        best_score, best_action = None, None

        if index == 0:  
            for action in gameState.getLegalActions(index):  # we get the minimax score of the successor for each possible action of the pacman
                next_game_state = gameState.generateSuccessor(index, action)
                _, score = self.expectimax(depth, index + 1, next_game_state)

                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action

        else:  # If it is min player's (ghost) turn
            ghostActions = gameState.getLegalActions(index)     # we get the expectmax score of the successor for each possible action of the ghost
            if len(ghostActions) is not 0:
                prob = 1.0 / len(ghostActions)
            for action in gameState.getLegalActions(index): 
                following_state = gameState.generateSuccessor(index, action)
                _, score = self.expectimax(depth, index + 1, following_state)

                if best_score is None:
                    best_score = 0.0
                best_score += prob * score
                best_action = action

        if best_score is None:                                  # if our state has not successor states we return the evaluation function
            return None, self.evaluationFunction(gameState)

        return best_action, best_score        # otherwise we return the best_action and best_score

def betterEvaluationFunction(currentGameState):
   
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)

    # Initial all varieble that we needed
    if currentGameState.isWin() :  return sys.maxint
    if currentGameState.isLose() :  return -sys.maxint
    
    currentPosition = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    capsulePosition = currentGameState.getCapsules()
    GhostStates = currentGameState.getGhostStates()

    ratioFood, ratioGhost, ratioCapsule, ratioHunter = 5.0, 5.0, 5.0, 0.0
    ghostScore, capsuleScore, hunterScore = 0.0, 0.0, 0.0
    
    #Found a closest food  
    currentFoodList = currentFood.asList()
    closestFood = min([util.manhattanDistance(currentPosition, foodPos) for foodPos in currentFoodList])
    foodScore = 1.0 / closestFood
    
    #Found others scope
    if GhostStates:
        #take information about ghost
        ghostPositions = [ghostState.getPosition() for ghostState in GhostStates]
        ghostDistances = [util.manhattanDistance(currentPosition, ghostPos) for ghostPos in ghostPositions]
        ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
        #not hunting mode
        if sum(ScaredTimes) == 0 : 
            closestGhost = min(ghostDistances)
            ghostCenterPos = ( sum([ghostPos[0] for ghostPos in ghostPositions])/len(GhostStates),\
                               sum([ghostPos[1] for ghostPos in ghostPositions])/len(GhostStates))
            ghostCenterDist = util.manhattanDistance(currentPosition, ghostCenterPos)
            if ghostCenterDist <= closestGhost and closestGhost >= 1 and closestGhost <= 5:
                if len(capsulePosition) != 0:
                    closestCapsule = min([util.manhattanDistance(capsule,currentPosition) for capsule in capsulePosition])
                    if closestCapsule <= 3:
                        ratioCapsule, capsuleScore = 20.0, (1.0 / closestCapsule)
                        ratioGhost, ghostScore = 3.0, (-1.0 / (ghostCenterDist+1))
                    else:
                        ratioGhost, ghostScore = 10.0, (-1.0 / (ghostCenterDist+1))
                else:
                    ratioGhost, ghostScore = 10.0, (-1.0 / (ghostCenterDist+1))
            else:
                ghostScore = -1.0 / closestGhost
        else: # hunter mode
            normalGhostDist = []
            closestPrey = sys.maxint
            ghostCenterX, ghostCenterY = 0.0, 0.0
            for (index, ghostDist) in enumerate(ghostDistances):
                if ScaredTimes[index] == 0 :
                    normalGhostDist.append(ghostDist)
                    ghostCenterX += ghostPositions[index][0]
                    ghostCenterY += ghostPositions[index][1]
                else:
                    if ghostDist <= ScaredTimes[index] :
                        if ghostDist < closestPrey:
                            closestPrey = ghostDistances[index]
            if normalGhostDist:
                closestGhost = min(normalGhostDist)
                ghostCenterPos = ( ghostCenterX/len(normalGhostDist), ghostCenterY/len(normalGhostDist))
                ghostCenterDist = util.manhattanDistance(currentPosition, ghostCenterPos)
                if ghostCenterDist <= closestGhost and closestGhost >= 1 and closestGhost <= 5:
                    ratioGhost, ghostScore = 10.0, (- 1.0 / (ghostCenterDist+1))
                else:
                    ghostScore = - 1.0 / closestGhost
            ratioHunter, hunterScore = 35.0, (1.0 / closestPrey)
    
    # Take a result
    heuristic = currentGameState.getScore() + \
                ratioFood*foodScore + ratioGhost*ghostScore + \
                ratioCapsule*capsuleScore + ratioHunter*hunterScore
    return heuristic
# Abbreviation
better = betterEvaluationFunction
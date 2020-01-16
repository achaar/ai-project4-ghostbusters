# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined, sample
from random import shuffle

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        # print("Ghost predicted at: ", all[maxIndex][0])
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        # print('total value sum: ' , self.total())
        if (self.total() == 0.0):
            return
        allItems = list(self.items())
        sum = 0
        for x in allItems:
            sum += x[1]

        if sum==0.0:
            return

        # print("sum: ", sum)
        new_dict = DiscreteDistribution()

        for x in allItems:
            # x = list(x)
            new_dict[x[0]] = x[1]/sum
            # print(new_dict[x[0]])

        self.clear()
        self.update(new_dict)
        # self = new_dict

        return




        # raiseNotDefined()

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        self.normalize()
        allItems = list(self.items())
        # print("prev:",allItems)
        prev = 0.0
        temp = []
        for x in allItems:
            x = list(x)
            x[1] = float(prev + float(x[1]))
            prev = x[1]
            temp.append(x)
            # print(x[1])
        # print("after:",temp)
        rand_value = random.random()
        # print("random:", rand_value)
        value_new = [x[1] for x in temp]
        value_list = [x[1] for x in temp if x[1]>=rand_value]
        # print("value greater than random:", value_list)

        if value_list==[]:
            index=0
        else:
            index = value_new.index(min(value_list))

        # print("index: ", index)
        return temp[index][0]

        # raiseNotDefined()


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        # print("noisyDistance, pacmanPosition, ghostPosition, jailPosition", noisyDistance, pacmanPosition, ghostPosition, jailPosition)
        if ghostPosition == jailPosition:
            # return 1  # if noisyDistance is None else 0
            if noisyDistance == None:
                # print("noisyDistance, pacmanPosition, ghostPosition, jailPosition", noisyDistance, pacmanPosition,
                #       ghostPosition, jailPosition, "Returned 1")
                return 1
            else:
                return 0

        if ghostPosition != jailPosition and noisyDistance == None:
            # print("noisyDistance, pacmanPosition, ghostPosition, jailPosition", noisyDistance, pacmanPosition,
            #       ghostPosition, jailPosition, "Returned 0")
            return 0

        return busters.getObservationProbability(noisyDistance, manhattanDistance(pacmanPosition, ghostPosition))


        # raiseNotDefined()

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        # raiseNotDefined()
        new_belief = DiscreteDistribution()
        for p in self.allPositions:
            obsProb = self.getObservationProb(observation, gameState.getPacmanPosition(), p, self.getJailPosition())
            # print("obsProb: ", obsProb)
            new_belief[p] = obsProb*self.beliefs[p]
        new_belief.normalize()

        self.beliefs = new_belief
        return

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        allPositions = self.allPositions
        # print(bcolors.HEADER, "Ghost Index: ", self.index, "allPositions: ", allPositions, bcolors.ENDC)
        # print(bcolors.OKBLUE, "Ghost Index: ", self.index, "Pacman Pos: ", gameState.getPacmanPosition(), "Jail Position: ", self.getJailPosition(), "Ghost Living: ", gameState.getLivingGhosts()[(self.index)], bcolors.ENDC)
        new_beliefs = DiscreteDistribution()

        for oldPos in allPositions:
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            oldBelief = self.beliefs[oldPos]

            for p in self.legalPositions:
                if p == self.getJailPosition():  #  cannot be in jail in the next state
                    new_beliefs[p] = 0.0
                    # print("Ghost cannot be in jail in the next state! ", oldPos)
                else:
                    new_beliefs[p] += float(newPosDist[p])*float(oldBelief)
            # print("old: ", oldPos, "next: ", p, "belief: ", self.beliefs[p])

            # new_beliefs[self.getJailPosition()] = 0.0

        new_beliefs.normalize()

        self.beliefs = new_beliefs
        # print("Ghost to be found at: ", self.beliefs.argMax())
        # print(new_beliefs)
        # if self.getJailPosition() == (3, 1):
        #     print(bcolors.FAIL, "self.getJailPosition(): ", self.getJailPosition(), "P(jail): ",
        #           new_beliefs[self.getJailPosition()], self.beliefs[(3, 1)], bcolors.ENDC)  # len(allPositions)-2]

        # self.beliefs.normalize()


        if gameState.getLivingGhosts()[(self.index)] == False:  # Ghost is already dead, correct
            # print(" xyz Ghost is already dead and in Jail!")
            new_beliefs = DiscreteDistribution()
            new_beliefs[self.getJailPosition()] = 1.0
            self.beliefs = new_beliefs
            # print("Error: ", self.beliefs)

        return
        # # raiseNotDefined()

    def getBeliefDistribution(self):
        # print(bcolors.FAIL, "getBeliefDistribution called!", bcolors.ENDC)
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # raiseNotDefined()
        ghostPositions = self.legalPositions
        list_full = 1
        while list_full==1:
            for pos in ghostPositions:
                if len(self.particles)==self.numParticles:
                    list_full = 0
                    break
                self.particles.append(pos)



    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        if observation == None:
            for i in range(self.numParticles):
                self.particles[i] = self.getJailPosition()

        prevBeliefDist = self.getBeliefDistribution()
        nextBeliefDist = DiscreteDistribution()
        for ghostPos in set(self.particles):  # or self.legalPositions
            obsProb = self.getObservationProb(observation, gameState.getPacmanPosition(), ghostPos, self.getJailPosition())
            nextBeliefDist[ghostPos] = obsProb*prevBeliefDist[ghostPos]

        nextBeliefDist.normalize()

        if nextBeliefDist.total() == 0.0:  # if after reweighing the dist sum is 0.0
            self.initializeUniformly(gameState)
        else:
            self.particles = [nextBeliefDist.sample() for i in range(self.numParticles)]

        return

        # return beliefDist

        # raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        oldDist = self.getBeliefDistribution()
        newDist = DiscreteDistribution()
        for ghostPos in self.legalPositions:
            newPosDist = self.getPositionDistribution(gameState, ghostPos)
            for p in newPosDist:
                newDist[p] += newPosDist[p]*oldDist[ghostPos]
        newDist.normalize()
        if newDist.total() == 0.0:  # if after reweighing the dist sum is 0.0
            self.initializeUniformly(gameState)
        else:
            self.particles = [newDist.sample() for i in range(self.numParticles)]

        return

        # raiseNotDefined()

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.

        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        # raiseNotDefined()
        belief = DiscreteDistribution()
        for p in self.particles:
            belief[p] += 1.0

        belief.normalize()
        return belief


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # allTheCombinations = []

        # for pos in self.legalPositions:
        #     allTheCombinations.append(pos)

        # for i in range(self.numGhosts):
        # print("self.legalPositions", self.legalPositions)
        allTheCombinations = list(itertools.product(self.legalPositions, repeat=self.numGhosts))
        shuffle(allTheCombinations)
        while len(self.particles)<self.numParticles:
            for p in allTheCombinations:
                self.particles.append(p)
                if len(self.particles)>=self.numParticles:
                    break
        # self.particles = []
        # positions = list(itertools.product(self.legalPositions, repeat=self.numGhosts))
        # length = len(positions)
        # random.shuffle(positions)
        # for i in range(self.numParticles):
        #     self.particles.append(positions[i % length])
        #
        # return


        # raiseNotDefined()

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        if len(observation) < self.numGhosts:
            return
        # raiseNotDefined()
        # print(observation)
        # priorBelief = self.getBeliefDistribution()
        # print(bcolors.OKBLUE, observation, gameState.getPacmanPosition(), bcolors.ENDC)
        # print(bcolors.BOLD, "Prev MyDist: ", self.getBeliefDistribution(), bcolors.ENDC)
        nextBelief = DiscreteDistribution()
        ghostJailIndexes = []
        # for i in range(self.numGhosts):
        #     if observation[i] == None:
        #         for p in range(len(self.particles)):
        #             temp = list(self.particles[p])
        #             temp[i] = self.getJailPosition(i)
        #             self.particles[p] = tuple(temp)

        for p in self.particles:
            # oldProb = priorBelief[p]
            # if gameState.getPacmanPosition() in p:
            #     obsProb = 0.0
            # else:
            obsProb = 1.0
            for i in range(self.numGhosts):
                if observation[i]!=None:  # Not valid -> observations with None have p[i] = jailPosition so, their obsProb = 1
                    obsProb *= self.getObservationProb(observation[i], gameState.getPacmanPosition(), p[i], self.getJailPosition(i))  # joint observation prob
            nextBelief[p] += obsProb
            # print(p, obsProb)

        nextBelief.normalize()

        if nextBelief.total() == 0.0:
            # print(bcolors.FAIL, "ALL WEIGHTS 0, REINITIALIZE!", bcolors.ENDC)
            self.initializeUniformly(gameState)
        else:
            # resample = []
            for i in range(len(self.particles)):
                # resample.append(tuple(nextBelief.sample()))
                self.particles[i] = nextBelief.sample()

            # self.particles = resample
            # print(bcolors.BOLD, "AfterMyDist: ", self.getBeliefDistribution(), bcolors.ENDC)
            # print(self.particles)
        for i in range(self.numGhosts):
            if observation[i] == None:
                for p in range(len(self.particles)):
                    temp = list(self.particles[p])
                    temp[i] = self.getJailPosition(i)
                    self.particles[p] = tuple(temp)

        return

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            # raiseNotDefined()
            for i in range(self.numGhosts):
                newPosDist = self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i])
                newParticle[i] = newPosDist.sample()

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        # print(bcolors.OKGREEN, "Marginal Dist: ", dist, "AgentIndex: ", self.index, bcolors.ENDC)
        return dist

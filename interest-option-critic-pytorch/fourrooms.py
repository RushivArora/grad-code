import numpy as np
from gym import core, spaces
from gym.envs.registration import register

class FourroomsDistABS(core.Env):
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        print("Occupancy = ", self.occupancy)

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))
        print("Observation Space = ", self.observation_space)

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        self.states = {}
        statenum = 0
        hallwaytop = (3,6)
        hallwayright = (7,9)
        hallwaybottom = (6,2)
        hallwayleft = (10,6)
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
                    distance_to_hallwaytop = np.abs((hallwaytop[0] - i)) + np.abs((hallwaytop[1] - j))
                    distance_to_hallwaybottom = np.abs((hallwaybottom[0] - i)) + np.abs((hallwaybottom[1] - j))
                    distance_to_hallwayleft = np.abs((hallwayleft[0] - i)) + np.abs((hallwayleft[1] - j))
                    distance_to_hallwayright = np.abs((hallwayright[0] - i)) + np.abs((hallwayright[1] - j))
                    self.states[(i,j)] = (i, j, distance_to_hallwaytop,  distance_to_hallwaybottom, distance_to_hallwayleft, distance_to_hallwayright)
        self.tocell = {v:k for k,v in self.tostate.items()}
        print("To state = ", self.tostate)
        print("To Cell = ", self.tocell)

        self.goal = 62
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return self.states[(self.currentcell[0],self.currentcell[1])]
        #return state

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if self.rng.uniform() < 1/3.:
            #if self.rng.uniform() < 0:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)
        if (reward != 1):
            reward = 0
        #return state, reward, done, None
        return self.states[(self.currentcell[0], self.currentcell[1])], reward, done, {}


class FourroomsDist(core.Env):
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        print("Occupancy = ", self.occupancy)

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))
        print("Observation Space = ", self.observation_space)

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        self.states = {}
        statenum = 0
        hallwaytop = (3,6)
        hallwayright = (7,9)
        hallwaybottom = (6,2)
        hallwayleft = (10,6)
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
                    distance_to_hallwaytop = (hallwaytop[0] - i) + (hallwaytop[1] - j)
                    distance_to_hallwaybottom = (hallwaybottom[0] - i) + (hallwaybottom[1] - j)
                    distance_to_hallwayleft = (hallwayleft[0] - i) + (hallwayleft[1] - j)
                    distance_to_hallwayright = (hallwayright[0] - i) + (hallwayright[1] - j)
                    self.states[(i,j)] = (i, j, distance_to_hallwaytop,  distance_to_hallwaybottom, distance_to_hallwayleft, distance_to_hallwayright)
        self.tocell = {v:k for k,v in self.tostate.items()}
        print("To state = ", self.tostate)
        print("To Cell = ", self.tocell)

        self.goal = 62
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        self.goal = 62
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return self.states[(self.currentcell[0],self.currentcell[1])]
        #return state

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if self.rng.uniform() < 1/3.:
            #if self.rng.uniform() < 0:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)
        if (reward != 1):
            reward = 0
        #return state, reward, done, None
        return self.states[(self.currentcell[0], self.currentcell[1])], reward, done, {}


class FourroomsXY(core.Env):
    # Four Rooms but just uses (x,y) of agent as state_space
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        print("Occupancy = ", self.occupancy)

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))
        print("Observation Space = ", self.observation_space)

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        self.states = {}
        statenum = 0
        hallwaytop = (3,6)
        hallwayright = (7,9)
        hallwaybottom = (6,2)
        hallwayleft = (10,6)
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
                    self.states[(i,j)] = (i, j)
        self.tocell = {v:k for k,v in self.tostate.items()}
        print("To state = ", self.tostate)
        print("To Cell = ", self.tocell)

        self.goal = 62
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return self.states[(self.currentcell[0],self.currentcell[1])]
        #return state

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if self.rng.uniform() < 1/3.:
            #if self.rng.uniform() < 0:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)
        if (reward != 1):
            reward = 0
        #return state, reward, done, None
        return self.states[(self.currentcell[0], self.currentcell[1])], reward, done, {}
        
class FourroomsOnehot(core.Env):
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        print("Occupancy = ", self.occupancy)

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))
        print("Observation Space = ", self.observation_space)

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        self.states = {}
        statenum = 0
        self.obs_to_state = {}
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    one_hot = np.zeros(self.observation_space.n)
                    one_hot[statenum] = 1
                    self.tostate[(i,j)] = statenum
                    statenum += 1
                    self.states[(i,j)] = one_hot
        self.tocell = {v:k for k,v in self.tostate.items()}
        print("To state = ", self.tostate)
        print("To Cell = ", self.tocell)

        self.goal = 62
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail
        
    def reset_goal(self):
        state = self.rng.choice(self.init_states)
        if self.goal == 62:
            self.goal = state
        else:
            self.goal = 62
        return

    def reset(self):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return self.states[(self.currentcell[0],self.currentcell[1])]
        #return state

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if self.rng.uniform() < 1/3.:
            #if self.rng.uniform() < 0:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)
        #reward = 50
        if (reward != 1):
            reward = 0
        #return state, reward, done, None
        return self.states[(self.currentcell[0], self.currentcell[1])], reward, done, {}

register(
    id='FourroomsDistABS-v0',
    entry_point='fourrooms:FourroomsDistABS',
    max_episode_steps=50,
    reward_threshold=1,
)

register(
    id='FourroomsDist-v0',
    entry_point='fourrooms:FourroomsDist',
    max_episode_steps=50,
    reward_threshold=1,
)

register(
    id='FourroomsXY-v0',
    entry_point='fourrooms:FourroomsXY',
    max_episode_steps=50,
    reward_threshold=1,
)

register(
    id='FourroomsOnehot-v0',
    entry_point='fourrooms:FourroomsOnehot',
    max_episode_steps=10000,
    reward_threshold=1,
)

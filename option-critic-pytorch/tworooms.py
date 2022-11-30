import numpy as np
from gym import core, spaces
from gym.envs.registration import register

class Tworooms(core.Env):
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w           w
w           w
w           w
w           w
w          Gw
wwwwwwwwwwwww
"""
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w    Gw
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
        hallway = (3,6)
        for i in range(7):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
                    #distance_to_hallway = np.abs(hallway[0] - i) + np.abs(hallway[1] - j)
                    distance_to_hallway = (hallway[0] - i) + (hallway[1] - j)
                    self.states[(i,j)] = (i, j, distance_to_hallway)
        self.tocell = {v:k for k,v in self.tostate.items()}
        print("To state = ", self.tostate)
        print("To Cell = ", self.tocell)

        self.goal = 50
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

register(
    id='Tworooms-v0',
    entry_point='tworooms:Tworooms',
    max_episode_steps=20,
    reward_threshold=1,
)

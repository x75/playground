"""
Supplement for Embodied AI lecture 20170112

Some Reinforcement Learning examples

2017 Oswald Berthold
"""


import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sensorimotor_loops = [
    "td_0_prediction",
    ]

class Environment(object):
    def __init__(self, agents = []):
        self.agents = agents
        self.t = 0

    def step(self):
        print "%s.step a = %s" % (self.__class__.__name__, a)
        s = a
        self.t += 1
        return s

    def reset(self):
        print "Implement me"

class GridEnvironment(Environment):
    def __init__(self, agents = [], num_x = 3, num_y = 3):
        Environment.__init__(self, agents = agents)
        self.num_x, self.num_y = num_x, num_y
        self.lim_l = np.array([[0], [0]])
        self.lim_u = np.array([[self.num_x-1], [self.num_y-1]])
        # fixed
        self.goal = np.array([[2], [2]])

        self.reset()
        
        # initialize goal
        # print "s", self.s

    def reset(self):
        print "%s.reset" % self.__class__.__name__
        # init state
        self.s = np.zeros((len(self.agents), self.num_x, self.num_y))
        
        # initialize agents
        for agent_idx, agent in enumerate(self.agents):
            x = np.random.randint(0, self.num_x)
            y = np.random.randint(0, self.num_y)
            self.s[agent_idx,x,y] = 1
            agent.terminal  = False
            agent.terminal_ = 1
        # print self.s # [agent_idx,x,y]

        
    def step(self):
        """Actual gridworld mechanics"""
        # loop over agents
        for agent_idx, agent  in enumerate(self.agents):
            # if agent.terminal:
            #     return self.s
            # get agent location as coordinates
            a_pos    = self.decode_state_to_loc(self.s[agent_idx])
            # get agent reward
            a_reward = self.decode_loc_to_reward(a_pos)
            # debug
            print "a_pos, a_reward", a_pos, a_reward
            # compute agent sensors from location and reward
            sensors = np.array([a_pos.flatten().tolist() + [a_reward]]).T

            # step the agent
            a = agent.step(sensors)

            # check terminal for state a_pos, separate from reward computation
            isterminal = self.decode_loc_to_terminal(a_pos)
            agent.terminal = isterminal
            
            self.s[agent_idx] = self.do_action(agent_idx, a)
            
            print "%s.step #%04d a_%d = %s" % (self.__class__.__name__, self.t, agent_idx, a)
        self.t += 1
        return self.s

    def decode_state_to_loc(self, s):
        return np.array([
            [np.sum(np.argmax(s, axis=0))],
            [np.sum(np.argmax(s, axis=1))]
        ])
    
    def decode_loc_to_reward(self, l):
        return (np.sum(l == self.goal) > 1.0) * 1.0

    def decode_loc_to_terminal(self, l):
        return np.all(l == self.goal)
    
    def do_action(self, agent_idx, a):
        s = self.s[agent_idx]
        # print "s", s
        # implement s = self.move(s, a)
        
        # get agent world state: location x,y
        ag_pos = self.decode_state_to_loc(s)

        # decode action
        ag_vel = self.decode_action(a)
        # print "ag_vel = %s" % (ag_vel)
        ag_pos_ = np.clip(ag_pos + ag_vel, self.lim_l, self.lim_u)

        ag_pos  = ag_pos.flatten()
        ag_pos_ = ag_pos_.flatten()
        assert s[ag_pos[0], ag_pos[1]] == 1.0
        # print "s", s[ag_pos[0], ag_pos[1]], s[ag_pos_[0], ag_pos_[1]]

        # move
        s[ag_pos[0],  ag_pos[1] ] = 0.0
        s[ag_pos_[0], ag_pos_[1]] = 1.0
        # print "s  = %s\na  = %s/%s\ns' = %s" % (ag_pos, a, ag_vel, ag_pos_)
        return s

    def decode_action(self, a):
        assert a.shape == (1, 1)
        # 
        if a[0,0] == 0: # stay
            vel = [0, 0]
        elif a[0,0] == 1: # north
            vel = [1, 0]
        elif a[0,0] == 2: # east
            vel = [0, 1]
        elif a[0,0] == 3: # south
            vel = [-1, 0]
        elif a[0,0] == 4: # west
            vel = [0, -1]
        elif a[0,0] == 5: # northeast
            vel = [1, 1]
        elif a[0,0] == 6: # southeast
            vel = [-1, 1]
        elif a[0,0] == 7: # southwest
            vel = [-1, -1]
        elif a[0,0] == 8: # northwest
            vel = [1, -1]
        return np.array([vel]).T

class Agent(object):
    def __init__(self, ndim_s = 2, ndim_a = 1):
        self.ndim_a = ndim_a
        self.ndim_s = ndim_s
        self.a = np.zeros((self.ndim_a, 1))
        self.s = np.zeros((self.ndim_s, 1))
        self.t = 0
        self.terminal  = False
        self.terminal_ = 1

    def step(self, s):
        print "s = %s" % s
        self.t += 1
        a = s
        return a

class TD0PredictionAgent(Agent):
    def __init__(self, ndim_s = 3, ndim_a = 1, ndim_x = 3, ndim_y = 3):
        Agent.__init__(self, ndim_s, ndim_a)

        # world dims
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y

        # learning rate
        self.alpha = 5e-3
        
        # discount factor
        self.gamma = 0.7
        
        # hardcoded gridworld actions
        self.actions = ["nop", "n", "e", "s", "w", "ne", "nw", "se", "sw"]
        self.actions_num = np.arange(len(self.actions), dtype=int).reshape((len(self.actions), 1))
        self.a = np.zeros((self.actions_num.shape[1], 1))
        self.s = np.zeros((ndim_s, 1)) # x, y, r
        self.s_tm1 = self.s.copy()
        self.v = np.zeros((self.ndim_x, self.ndim_y))

    def step(self, s):
        if self.terminal:
            self.terminal_ -= 1
        # sensory measurement
        self.s = s.copy()
        print "%s.step s = %s" % (self.__class__.__name__, self.s)

        # update v
        l_x = int(self.s[0,0])
        l_y = int(self.s[1,0])
        l_x_tm1 = int(self.s_tm1[0,0])
        l_y_tm1 = int(self.s_tm1[1,0])

        print "l", l_x, l_y, "l_tm1", l_x_tm1, l_y_tm1
                
        print "v", l_x, l_y, self.v[l_x, l_y]
        v_s_tm1 = self.v[l_x_tm1, l_y_tm1].copy()
        self.v[l_x_tm1, l_y_tm1] = v_s_tm1 + self.alpha * (self.s[2,0] + self.gamma * self.v[l_x, l_y] - v_s_tm1)
        
        # policy
        self.a = np.random.randint(len(self.actions), size=self.a.shape)
        # print self.a
        self.s_tm1 = self.s.copy()
        self.t += 1
        return self.a

################################################################################
# operations
    
def plot_init(ev):
    plt.ion()
    fig = plt.figure()
    gs = gridspec.GridSpec(len(ev.agents), 2)
    axs = []
    for i, a in enumerate(ev.agents):
        axs.append([fig.add_subplot(gs[2*i]), fig.add_subplot(gs[2*i+1])])
        axs[-1][0].set_title("Agent %d state" % i)
        axs[-1][1].set_title("Agent %d value" % i)
        axs[-1][0].set_aspect(1)
        axs[-1][1].set_aspect(1)
    return fig, gs, axs

def plot_draw_ev(fig, gs, axs, ev):
    for i, a in enumerate(ev.agents):
        # if a.terminal: return
        ax_s = axs[i][0]
        ax_s.pcolormesh(ev.s[i], cmap=plt.get_cmap("gray"))
        ax_s.plot([ev.goal[0,0] + 0.5], [ev.goal[1,0] + 0.5], "ro", markersize = 20, alpha= 0.5)
        ax_v = axs[i][1]
        ax_v.pcolormesh(ev.agents[i].v, cmap=plt.get_cmap("gray"), vmin = 0.0, vmax = 1.0)
    plt.draw()
    plt.pause(1e-3)
        
def td_0_prediction(args):
    numepisodes = args.numepisodes
    maxsteps    = args.maxsteps
    
    ag = TD0PredictionAgent(ndim_s = 3, ndim_a = 1)
    # ag2 = TD0PredictionAgent(ndim_s = 3, ndim_a = 1)
    ev = GridEnvironment(agents = [ag], num_x = 3, num_y = 3)

    s = ag.s
    a = ag.a

    fig, gs, axs = plot_init(ev)
    
    print "environment", ev
    print "      agent", ag

    for i in range(numepisodes):
        # reset agent
        ev.reset()
        t = 0
        terminal = False
        while not terminal and t < maxsteps:
        # for t in range(maxsteps):
            print "epi %d, step %d" % (i, t)
            # step the world
            ev.step()
            # print "td_0_prediction a[t = %d] = %s, s[t = %d] = %s" % (t, a, t, s)
            if t % 50 == 0:
                plot_draw_ev(fig, gs, axs, ev)
            terminal = np.all(np.array([agent.terminal_ < 1 for agent in ev.agents]))
            t += 1

    print "ev.steps = %d" % (ev.t)
    print "ag.steps = %d" % (ag.t)

    plt.ioff()
    plt.show()
    
def main(args):
    if args.sensorimotor_loop == "td_0_prediction":
        td_0_prediction(args)
    else:
        print "Unknown sm loop %s, exiting" % (args.sensorimotor_loop)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ne", "--numepisodes", default=10,  type=int, help="Number of episodes")
    parser.add_argument("-ms", "--maxsteps",    default=100, type=int, help="Maximum number of steps per episodes")
    parser.add_argument("-sm", "--sensorimotor_loop", default="td_0_prediction", type=str, help="Which sm loop (Learner), one of " + ", ".join(sensorimotor_loops))
    
    args = parser.parse_args()
    
    main(args)
    # main_extended(args)

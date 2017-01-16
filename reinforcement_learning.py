"""
Supplement for Embodied AI lecture 20170112

Some Reinforcement Learning examples

2017 Oswald Berthold
"""

# notes
#  use pushback for implementing lambda?

import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# uh oh
from dimstack import dimensional_stacking


sensorimotor_loops = [
    "td_0_prediction",         # TD(0) prediction of v
    "td_0_off_policy_control", # aka Q-Learning
    "td_0_on_policy_control",  # aka SARSA"
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
        # # constant goal
        # self.goal = np.array([[4], [1]])
        # self.goal = np.array([[2], [3]])
        # self.goal = np.array([[0], [0]])
        # self.goal = np.array([[0], [2]])
        # self.goal = np.array([[1], [2]])
        # random fixed goal
        self.goal = np.random.uniform([0, 0], [self.num_x, self.num_y], size=(1, 2)).T.astype(int) # 

        self.reset()
        
    def reset(self):
        # print "%s.reset" % self.__class__.__name__
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
            # print "ev.s", self.s[agent_idx]
            # get agent location as coordinates
            a_pos    = self.decode_state_to_loc(self.s[agent_idx])
            # get agent reward
            a_reward = self.decode_loc_to_reward(a_pos)
            # debug
            # print "a_pos, a_reward", a_pos, a_reward
            # compute agent sensors from location and reward
            sensors = np.array([a_pos.flatten().tolist() + [a_reward]]).T

            # step the agent
            a = agent.step(sensors)

            # check terminal for state a_pos, separate from reward computation
            isterminal = self.decode_loc_to_terminal(a_pos)
            agent.terminal = isterminal
            
            self.s[agent_idx] = self.do_action(agent_idx, a)
            
            # print "%s.step #%04d a_%d = %s, s_%d = %s" % (self.__class__.__name__, self.t, agent_idx, a, agent_idx, self.s[agent_idx])
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
    def __init__(self, ndim_s = 3, ndim_a = 1, ndim_x = 3, ndim_y = 3, alpha = 1e-3, gamma = 0.0):
        Agent.__init__(self, ndim_s, ndim_a)

        # world dims
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y

        # learning rate
        self.alpha = alpha # 5e-3
        
        # discount factor
        self.gamma = gamma # 0.7
        
        # hardcoded gridworld actions
        self.actions = ["nop", "n", "e", "s", "w", "ne", "nw", "se", "sw"]
        self.actions_num = np.arange(len(self.actions), dtype=int).reshape((len(self.actions), 1))
        # action
        self.a = np.zeros((self.actions_num.shape[1], 1))
        self.a_tm1 = self.a.copy()
        # state
        self.s = np.zeros((ndim_s, 1)) # x, y, r
        self.s_tm1 = self.s.copy()
        
        # estimated state value function v
        self.v = np.ones((self.ndim_x, self.ndim_y)) * 0.1
        # estimated state-action value function q
        q_shape = (self.ndim_x, self.ndim_y, len(self.actions))
        self.q = np.ones(q_shape) * 2.0
        # self.q = np.random.uniform(0, 10, q_shape)
        # self.q = np.arange(np.prod(q_shape)).reshape(q_shape)
        self.q_Q     = np.ones(q_shape) * 2.0
        # self.q_Q     = np.random.uniform(0, 0.1, q_shape)
        # self.q_Q[self.goal[0,0], self.goal[1,0]] = 0.0
        self.q_SARSA = np.ones(q_shape) * 2.0

        # self.policy_func = self.policy_random
        self.policy_func = self.policy_epsilon_greedy

    # policies
    def policy(self, q, s):
        return self.policy_func(q, s)
        
    def policy_random(self, q, s):
        return np.random.randint(len(self.actions), size=self.a.shape)

    def policy_epsilon_greedy(self, q, s, epsilon = 0.05):
        if np.random.uniform() < epsilon:
            return self.policy_random(q, s)
        else:
            # get best action according to current q estimate
            q_s = q[s[0,0], s[1,0]]
            # print "%s.policy_epsilon_greedy q_s = %s" % (self.__class__.__name__, q_s)
            a_s = np.argmax(q_s).reshape(self.a.shape)
            # print "%s.policy_epsilon_greedy a_s = %s" % (self.__class__.__name__, a_s)
            return a_s
        
    def step(self, s):
        # stop episode
        if self.terminal:
            self.terminal_ -= 1
            
        # sensory measurement: [x, y, reward].T
        self.s = s.copy()
        # print "%s.step s = %s" % (self.__class__.__name__, self.s)

        # current state
        l_x = int(self.s[0,0])
        l_y = int(self.s[1,0])
        # last state
        l_x_tm1 = int(self.s_tm1[0,0])
        l_y_tm1 = int(self.s_tm1[1,0])
        l_a_tm1   = self.a_tm1[0,0]
        # print "l", l_x, l_y, "l_tm1", l_x_tm1, l_y_tm1
                
        # update v
        # print "v", l_x, l_y, self.v[l_x, l_y]
        
        # back up old state value once
        self.v_s_tm1 = self.v[l_x_tm1, l_y_tm1].copy()
        # perform update, SB2nded pg. ?, eq. ?
        self.v[l_x_tm1, l_y_tm1] = self.v_s_tm1 + self.alpha * (self.s[2,0] + self.gamma * self.v[l_x, l_y] - self.v_s_tm1)

        # back up old state-action value once
        self.q_sa_tm1 = self.q[l_x_tm1, l_y_tm1, l_a_tm1].copy()
        # perform update, SB2nded pg. ?, eq. ?
        self.q[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_sa_tm1 + self.alpha * (self.s[2,0] + self.gamma * self.q[l_x, l_y, l_a_tm1] - self.q_sa_tm1)
                
        # back up old state-action value once Q-Learning
        self.q_Q_tm1 = self.q_Q[l_x_tm1, l_y_tm1, l_a_tm1].copy()
        # perform update, SB2nded pg. ?, eq. ?
        self.q_Q[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_Q_tm1 + self.alpha * (self.s[2,0] + self.gamma * np.max(self.q_Q[l_x, l_y, l_a_tm1]) - self.q_Q_tm1)

        # policy: some functional thing that produces an action
        # self.a = self.policy(self.q_Q, self.s)
        self.a = self.policy(self.q_SARSA, self.s)
        # print self.a
        
        # back up old state-action value once Q-Learning
        self.q_SARSA_tm1 = self.q_SARSA[l_x_tm1, l_y_tm1, l_a_tm1].copy()
        # perform update, SB2nded pg. ?, eq. ?
        self.q_SARSA[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_SARSA_tm1 + self.alpha * (self.s[2,0] + (self.gamma * self.q_SARSA[l_x, l_y, self.a]) - self.q_SARSA_tm1)

        # back up state
        self.s_tm1 = self.s.copy()
        # back up action
        self.a_tm1 = self.a.copy()
        self.t += 1
        return self.a

# class TD0OffPolicyControlAgent(TD0PredictionAgent):
#     def __init__(self, ndim_s = 3, ndim_a = 1, ndim_x = 3, ndim_y = 3, alpha = 1e-3, gamma = 0.0):
#         TD0PredictionAgent.__init__(self, ndim_s = ndim_s, ndim_a = ndim_a, ndim_x = ndim_x, ndim_y = ndim_y, alpha = alpha, gamma = gamma)
#         q_shape = (self.ndim_x, self.ndim_y, len(self.actions))
#         # Q update
#         self.q_Q     = np.ones(q_shape) * 2.0
#         # self.q_Q     = np.random.uniform(0, 0.1, q_shape)
#         # self.q_Q[self.goal[0,0], self.goal[1,0]] = 0.0
#         self.q_SARSA = np.ones(q_shape) * 2.0
#         # self.q_SARSA[self.goal[0,0], self.goal[1,0]] = 0.0

#     def step(self, s):
#         # stop episode
#         if self.terminal:
#             self.terminal_ -= 1
            
#         # sensory measurement: [x, y, reward].T
#         self.s = s.copy()
#         # print "%s.step s = %s" % (self.__class__.__name__, self.s)

#         # current state
#         l_x = int(self.s[0,0])
#         l_y = int(self.s[1,0])
#         # last state
#         l_x_tm1 = int(self.s_tm1[0,0])
#         l_y_tm1 = int(self.s_tm1[1,0])
#         l_a_tm1   = self.a_tm1[0,0]
#         # print "l", l_x, l_y, "l_tm1", l_x_tm1, l_y_tm1
                
#         # update v
#         # print "v", l_x, l_y, self.v[l_x, l_y]
        
#         # back up old state value once
#         self.v_s_tm1 = self.v[l_x_tm1, l_y_tm1].copy()
#         # perform update, SB2nded pg. ?, eq. ?
#         self.v[l_x_tm1, l_y_tm1] = self.v_s_tm1 + self.alpha * (self.s[2,0] + self.gamma * self.v[l_x, l_y] - self.v_s_tm1)

#         # back up old state-action value once
#         self.q_sa_tm1 = self.q[l_x_tm1, l_y_tm1, l_a_tm1].copy()
#         # perform update, SB2nded pg. ?, eq. ?
#         self.q[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_sa_tm1 + self.alpha * (self.s[2,0] + self.gamma * self.q[l_x, l_y, l_a_tm1] - self.q_sa_tm1)

#         # back up old state-action value once Q-Learning
#         self.q_Q_tm1 = self.q_Q[l_x_tm1, l_y_tm1, l_a_tm1].copy()
#         # perform update, SB2nded pg. ?, eq. ?
#         self.q_Q[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_Q_tm1 + self.alpha * (self.s[2,0] + self.gamma * np.max(self.q_Q[l_x, l_y, l_a_tm1]) - self.q_Q_tm1)

#         # policy: some functional thing that produces an action
#         # self.a = self.policy(self.q_Q, self.s)
#         self.a = self.policy(self.q_SARSA, self.s)
#         # print self.a
        
#         # back up old state-action value once Q-Learning
#         self.q_SARSA_tm1 = self.q_SARSA[l_x_tm1, l_y_tm1, l_a_tm1].copy()
#         # perform update, SB2nded pg. ?, eq. ?
#         self.q_SARSA[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_SARSA_tm1 + self.alpha * (self.s[2,0] + (self.gamma * self.q_SARSA[l_x, l_y, self.a]) - self.q_SARSA_tm1)
                                

#         # back up state
#         self.s_tm1 = self.s.copy()
#         # back up action
#         self.a_tm1 = self.a.copy()
#         self.t += 1
#         return self.a
                
        
################################################################################
# operations
    
def plot_init(ev):
    plt.ion()
    fig = plt.figure()
    fig.suptitle("TD(0) prediction learning for v and q")
    gs_numcol = 1 + 1 # 1 + 1 + 4 # 3
    gs = gridspec.GridSpec(len(ev.agents) * 4, gs_numcol)
    axs = []
    for i, a in enumerate(ev.agents):
        axs.append([
            # fig.add_subplot(gs[gs_numcol*i]),
            # fig.add_subplot(gs[gs_numcol*i+1]),
            # fig.add_subplot(gs[gs_numcol*i+2:])
            fig.add_subplot(gs[i*2+3,0]),
            fig.add_subplot(gs[i*2+3,1]),
            fig.add_subplot(gs[i*2,:]),
            fig.add_subplot(gs[i*2+1,:]),
            fig.add_subplot(gs[i*2+2,:]),
            ])
        axs[-1][0].set_title("Agent %d state (position on grid)" % i, fontsize=8)
        axs[-1][0].set_xlabel("x")
        axs[-1][0].set_ylabel("y")
        axs[-1][0].set_aspect(1)
        axs[-1][1].set_title("Agent %d state value v(s)" % i, fontsize = 8)
        axs[-1][1].set_xlabel("x")
        axs[-1][1].set_ylabel("y")
        axs[-1][1].set_aspect(1)
        axs[-1][2].set_title("Agent %d state-action value q(s,a)" % i, fontsize = 8)
        axs[-1][2].set_xlabel("f(a, x)")
        axs[-1][2].set_ylabel("y")
        # axs[-1][2].set_aspect((len(a.actions)*ev.num_x)/float(ev.num_y))
        # axs[-1][2].set_aspect((len(a.actions)*ev.num_x)/float(ev.num_y))
        axs[-1][2].set_aspect(1)
        axs[-1][3].set_aspect(1)
        axs[-1][4].set_aspect(1)
    return fig, gs, axs

def plot_pcolor_coordinates():
    pass

def plot_draw_ev(fig, gs, axs, ev):
    for i, a in enumerate(ev.agents):
        # print "plot_draw_ev s_%d = %s" % (i, ev.s[i])

        # x = 
        
        # plot state
        ax_s = axs[i][0]
        # clean up
        ax_s.clear()
        
        # print "ev.s[i].shape", ev.s[i].shape, a.v.shape, a.q.shape
        ax_s.pcolormesh(ev.s[i].T, cmap=plt.get_cmap("gray"))
        # ax_s.pcolormesh(ev.s[i][::-1], cmap=plt.get_cmap("gray"))
        ax_s.plot([ev.goal[0,0] + 0.5], [ev.goal[1,0] + 0.5], "ro", markersize = 20, alpha= 0.5)

        # plot state value
        ax_v = axs[i][1]
        ax_v.clear()
        # v_img = np.log(ev.agents[i].v + 1.0)
        v_img = ev.agents[i].v.T
        ax_v.pcolormesh(v_img, cmap=plt.get_cmap("gray"), vmin = 0.0) # , vmax = 1.0)

        # plot state-action value
        ax_q = axs[i][2]
        ax_q.clear()
        q_img = dimensional_stacking(ev.agents[i].q, [2, 1], [0])
        # q_img = dimensional_stacking(ev.agents[i].q_Q, [2, 1], [0])
        # q_img = dimensional_stacking(ev.agents[i].q_SARSA, [2, 1], [0])
        # print "q_img.shape", q_img.shape
        ax_q.pcolormesh(q_img, cmap=plt.get_cmap("gray"), vmin = 0.0, vmax = 2.0)

        # plot state-action value
        ax_q_Q = axs[i][3]
        ax_q_Q.clear()
        q_img_Q = dimensional_stacking(ev.agents[i].q_Q, [2, 1], [0])
        # q_img = dimensional_stacking(ev.agents[i].q_SARSA, [2, 1], [0])
        # print "q_img.shape", q_img.shape
        ax_q_Q.pcolormesh(q_img_Q, cmap=plt.get_cmap("gray"), vmin = 0.0, vmax = 2.0)

        # plot state-action value
        ax_q_SARSA = axs[i][4]
        ax_q_SARSA.clear()
        q_img_SARSA = dimensional_stacking(ev.agents[i].q_SARSA, [2, 1], [0])
        # print "q_img.shape", q_img.shape
        ax_q_SARSA.pcolormesh(q_img_SARSA, cmap=plt.get_cmap("gray"), vmin = 0.0, vmax = 2.0)
                        
    plt.draw()
    plt.pause(1e-3)
        

def get_agent(args):
    # if args.sensorimotor_loop == "td_0_prediction":
    return TD0PredictionAgent(ndim_s = 3, ndim_a = 1, ndim_x = args.world_x, ndim_y = args.world_y, alpha = args.alpha, gamma = args.gamma)
    # elif args.sensorimotor_loop == "td_0_off_policy_control":
    # return TD0OffPolicyControlAgent(ndim_s = 3, ndim_a = 1, ndim_x = args.world_x, ndim_y = args.world_y, alpha = args.alpha, gamma = args.gamma)
    # elif args.sensorimotor_loop == "td_0_on_policy_control":
    # else:
    # print "Unknown sm loop %s, exiting" % (args.sensorimotor_loop)
    # sys.exit(1)
    
def rl_experiment(args):
    # numepisodes = args.numepisodes
    # maxsteps    = args.maxsteps
    # plotfreq    = args.plotfreq
    
    setattr(args, "world_x", 5)
    setattr(args, "world_y", 5)
    
    ag = get_agent(args)
    # ag2 = TD0PredictionAgent(ndim_s = 3, ndim_a = 1)
    ev = GridEnvironment(agents = [ag], num_x = args.world_x, num_y = args.world_y)
    # ag.q_Q[ev.goal[0,0], ev.goal[1,0],:] = 0.1
    # ag.q_SARSA[ev.goal[0,0], ev.goal[1,0],:] = 0.0

    s = ag.s
    a = ag.a

    fig, gs, axs = plot_init(ev)
    
    print "environment", ev
    print "      agent", ag

    for i in range(args.numepisodes):
        # reset agent
        ev.reset()
        t = 0
        terminal = False
        while not terminal and t < args.maxsteps:
        # for t in range(maxsteps):
            # print "epi %d, step %d" % (i, t)
            # step the world
            ev.step()
            # print "td_0_prediction a[t = %d] = %s, s[t = %d] = %s" % (t, a, t, s)
            if t % args.plotfreq == 0:
                plot_draw_ev(fig, gs, axs, ev)
            terminal = np.all(np.array([agent.terminal_ < 1 for agent in ev.agents]))
            t += 1
        print "epi %d, final step %d" % (i, t)

    print "ev.steps = %d" % (ev.t)
    print "ag.steps = %d" % (ag.t)

    # save result
    for i, agent in enumerate(ev.agents):
        np.save("td0_ag%d_v.npy" % i, agent.v)
        np.save("td0_ag%d_q.npy" % i, agent.q)

    plt.ioff()
    plt.show()
            
def main(args):
    rl_experiment(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a",  "--alpha", default=1e-2,  type=float, help="Learning rate \alpha")
    parser.add_argument("-g",  "--gamma", default=0.8,  type=float, help="Discount factor \gamma")
    parser.add_argument("-ne", "--numepisodes", default=500,  type=int, help="Number of episodes")
    parser.add_argument("-ms", "--maxsteps",    default=100, type=int, help="Maximum number of steps per episodes")
    parser.add_argument("-sm", "--sensorimotor_loop", default="td_0_prediction", type=str, help="Which sm loop (Learner), one of " + ", ".join(sensorimotor_loops))
    parser.add_argument("-p",  "--plotfreq", default=1000, type=int, help="Plotting interval in steps")
    
    args = parser.parse_args()
    
    main(args)
    # main_extended(args)

"""
Supplement for Embodied AI lecture 20170112

Some Reinforcement Learning examples

Implementing only Temporal Difference methods so far:
 - TD(0) prediction
 - Q-Learning
 - SARSA

Possible additions
 - use function approximation for v,q,q_Q,q_SARSA
 - use state matrix as visual input / compare pg-pong, although that uses policy gradient
 
2017 Oswald Berthold
"""

# notes
#  use pushback for implementing lambda?

import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost

# uh oh
from dimstack import dimensional_stacking

# # from scikit neural networks
# from sknn.mlp import Regressor, Layer

# using keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import RMSprop
from keras import initializations
from keras.engine.topology import Merge

def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

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
        # self.goal = np.array([[4], [4]])
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

        # # include map with walls / real maze
        # # print "ag_pos", ag_pos
        # if ag_pos[0,0] in [2,3,4] and ag_pos[1,0] in [3]:
        #     # ag_vel = np.clip(ag_vel, )
        #     ag_vel[1,0] = np.clip(ag_vel[1,0], -np.inf, 0)
        
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
        elif a[0,0] == 1: # west
            vel = [1, 0]
        elif a[0,0] == 3: # north
            vel = [0, 1]
        elif a[0,0] == 5: # east
            vel = [-1, 0]
        elif a[0,0] == 7: # south
            vel = [0, -1]
        elif a[0,0] == 2: # northwest
            vel = [1, 1]
        elif a[0,0] == 4: # northeast
            vel = [-1, 1]
        elif a[0,0] == 6: # southeast
            vel = [-1, -1]
        elif a[0,0] == 8: # southwest
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
    # def __init__(self, ndim_s = 3, ndim_a = 1, ndim_x = 3, ndim_y = 3, alpha = 1e-3, gamma = 0.0):
    def __init__(self, args=argparse.Namespace(ndim_s = 3, ndim_a = 1)):
        Agent.__init__(self, args.ndim_s, args.ndim_a)

        # world dims
        self.ndim_x = args.ndim_x
        self.ndim_y = args.ndim_y

        # learning rate
        self.alpha = args.alpha # 5e-3

        # policy epsilon
        self.epsilon = args.epsilon # 5e-3
        
        # discount factor
        self.gamma = args.gamma # 0.7

        # type of learner / experiment
        self.sensorimotor_loop = args.sensorimotor_loop

        # type of value functions representation: table, parameterized approximation
        self.repr = args.repr
        
        # hardcoded gridworld actions
        self.actions = ["nop", "w", "nw", "n", "ne", "e", "se", "s", "sw"]
        self.actions_num = np.arange(len(self.actions), dtype=int).reshape((len(self.actions), 1))
        # action
        self.a = np.zeros((self.actions_num.shape[1], 1))
        self.a_tm1 = self.a.copy()
        # state
        self.s = np.zeros((self.ndim_s, 1)) # x, y, r
        self.s_tm1 = self.s.copy()
        
        # estimated state value function v
        self.v_tbl = np.ones((self.ndim_x, self.ndim_y)) * 0.1
        # estimated state-action value function q
        q_shape = (self.ndim_x, self.ndim_y, len(self.actions))
        self.q_tbl = np.ones(q_shape) * 0.0 # 2.0
        # self.q_tbl = np.random.uniform(0, 10, q_shape)
        # self.q_tbl = np.arange(np.prod(q_shape)).reshape(q_shape)
        self.q_Q_tbl     = np.ones(q_shape) * 0.0 # 2.0
        # self.q_Q_tbl     = np.random.uniform(0, 0.1, q_shape)
        # self.q_Q_tbl[self.goal[0,0], self.goal[1,0]] = 0.0
        self.q_SARSA_tbl = np.ones(q_shape) * 0.0 # 2.0

        if self.repr == "table":
            self.v       = self.v_tbl_predict
            self.q       = self.q_tbl_predict
            self.q_Q     = self.q_Q_tbl_predict
            self.q_SARSA = self.q_SARSA_tbl_predict
            self.v_update       = self.v_tbl_update
            self.q_update       = self.q_tbl_update
            self.q_Q_update     = self.q_Q_tbl_update
            self.q_SARSA_update = self.q_SARSA_tbl_update
        elif self.repr == "approximation":
            self.init_fa()
            self.v       = self.v_fa_predict
            self.q       = self.q_fa_predict
            self.q_Q     = self.q_Q_fa_predict
            self.q_SARSA = self.q_SARSA_fa_predict
            self.v_update       = self.v_fa_update
            self.q_update       = self.q_fa_update
            self.q_Q_update     = self.q_Q_fa_update
            self.q_SARSA_update = self.q_SARSA_fa_update

        # set pplicy according to learner
        print "self.sensorimotor_loop", self.sensorimotor_loop
        if self.sensorimotor_loop == "td_0_prediction":
            self.policy_func = self.policy_random
        elif self.sensorimotor_loop == "td_0_off_policy_control" or \
          self.sensorimotor_loop == "td_0_on_policy_control":
            print "epsilon greedy"
            self.policy_func = self.policy_epsilon_greedy
        else:
            # self.policy_func = self.policy_random
            print "Unknown learner %s, exiting" % (self.sensorimotor_loop)
            sys.exit(1)

    def init_fa(self):
        # init_str = "normal"
        init_str = my_init
        layer_1_num_units = 200
        layer_2_num_units = 20
        output_gain = 1.0
        input_gain  = 10.0
                        
        # this returns a tensor
        inputs = Input(shape=(2,))
        inputs_gain = Lambda(lambda x: x * input_gain)(inputs)
        # inputs_squared = Lambda(lambda x: (x ** 2) * 0.1)(inputs)
        # inputs_combined = Merge(mode="concat", concat_axis=1)([inputs_gain, inputs_squared])
        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(layer_1_num_units, activation='tanh', init=init_str)(inputs_gain)
        x = Dense(layer_2_num_units, activation='tanh', init=init_str)(x)
        predictions = Dense(1, activation='linear')(x)
        outputs_gain = Lambda(lambda x: x * output_gain)(predictions)

        # this creates a model that includes
        # the Input layer and three Dense layers
        opt_v_fa = RMSprop(lr = self.alpha)
        self.v_fa = Model(input=inputs, output=outputs_gain)
        self.v_fa.compile(optimizer=opt_v_fa, loss='mse')
        self.v_fa_training_cnt = 0
        self.v_fa_training_loss = 0

        # Q approximation
        # this returns a tensor
        inputs_q_fa = Input(shape=(2 + len(self.actions),))
        # inputs_q_fa = Input(shape=(3,))
        inputs_gain = Lambda(lambda x: x * input_gain)(inputs_q_fa)
        # inputs_squared = Lambda(lambda x: (x ** 2) * 0.1)(inputs_q_fa)
        # inputs_combined = Merge(mode="concat", concat_axis=1)([inputs_gain, inputs_squared])
        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(layer_1_num_units, activation='tanh', init=init_str)(inputs_gain)
        x = Dense(layer_2_num_units, activation='tanh', init=init_str)(x)
        predictions = Dense(1, activation='linear')(x)
        outputs_gain = Lambda(lambda x: x * output_gain)(predictions)

        # this creates a model that includes
        # the Input layer and three Dense layers
        opt_q_fa = RMSprop(lr = self.alpha)
        self.q_fa = Model(input=inputs_q_fa, output=outputs_gain)
        self.q_fa.compile(optimizer=opt_q_fa, loss='mse')
        self.q_fa_training_cnt = 0
        self.q_fa_training_loss = 0
        
        # this returns a tensor
        # inputs_q_Q_fa = Input(shape=(3,))
        inputs_q_Q_fa = Input(shape=(2 + len(self.actions),))
        inputs_gain = Lambda(lambda x: x * input_gain)(inputs_q_Q_fa)
        # inputs_squared = Lambda(lambda x: (x ** 2) * 0.1)(inputs_q_Q_fa)
        # inputs_combined = Merge(mode="concat", concat_axis=1)([inputs_gain, inputs_squared])
        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(layer_1_num_units, activation='tanh')(inputs_gain)
        x = Dense(layer_2_num_units, activation='tanh')(x)
        predictions = Dense(1, activation='linear')(x)
        outputs_gain = Lambda(lambda x: x * output_gain)(predictions)

        # this creates a model that includes
        # the Input layer and three Dense layers
        opt_q_Q_fa = RMSprop(lr = self.alpha)
        self.q_Q_fa = Model(input=inputs_q_Q_fa, output=outputs_gain)
        self.q_Q_fa.compile(optimizer=opt_q_Q_fa, loss='mse')
        self.q_Q_fa_training_cnt = 0
        self.q_Q_fa_training_loss = 0
        
        # this returns a tensor
        inputs_q_SARSA_fa = Input(shape=(2 + len(self.actions),))
        inputs_gain = Lambda(lambda x: x * input_gain)(inputs_q_SARSA_fa)
        # inputs_squared = Lambda(lambda x: (x ** 2) * 0.1)(inputs_q_SARSA_fa)
        # inputs_combined = Merge(mode="concat", concat_axis=1)([inputs_gain, inputs_squared])
        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(layer_1_num_units, activation='tanh')(inputs_gain)
        x = Dense(layer_2_num_units, activation='tanh')(x)
        predictions = Dense(1, activation='linear')(x)
        outputs_gain = Lambda(lambda x: x * output_gain)(predictions)
        
        # this creates a model that includes
        # the Input layer and three Dense layers
        opt_q_SARSA_fa = RMSprop(lr = self.alpha)
        self.q_SARSA_fa = Model(input=inputs_q_SARSA_fa, output=outputs_gain)
        self.q_SARSA_fa.compile(optimizer=opt_q_SARSA_fa, loss='mse')
        self.q_SARSA_fa_training_cnt = 0
        self.q_SARSA_fa_training_loss = 0
        
    def v_fa_predict(self, s):
        return self.v_fa.predict(s[:2,0].reshape((1,2)) * 1.0) * 1.0
        
    def v_fa_update(self, s):
        # print "s", s
        v_fa_tm1 = self.v(self.s_tm1)
        v_fa = self.v(s)
        x = self.s_tm1[:2,0].reshape((1,2))
        y = s[2,0] + self.gamma * v_fa
        if True or self.v_fa_training_cnt > 100 or s[2,0] > 0.0:
            # target_weight = (1.0 + s[2] * 10.0).reshape()
            target_weight = np.ones((1,)) + s[2] * 10.0
            self.v_fa_training_loss = self.v_fa.train_on_batch(x * 1.0, y * 1.0, sample_weight = target_weight)  # starts training
            self.v_fa_training_cnt += 1
        
    def q_fa_predict(self, s, a):
        a_ = np.zeros((len(self.actions),1))
        a_[a[0,0],0] = 1.0
        # x = np.vstack((s[:2,0].reshape((2,1)), a))
        x = np.vstack((s[:2,0].reshape((2,1)), a_))
        return self.q_fa.predict(x.T * 1.0) * 1.0
        
    def q_fa_update(self, s, a):
        # print "s", s
        a_tm1_ = np.zeros((len(self.actions),1))
        a_tm1_[self.a_tm1[0,0],0] = 1.0
        # print "a_tm1_", a_tm1_
        
        # q_fa_tm1 = self.q(self.s_tm1, self.a_tm1)
        q_fa = self.q(s, a)
        # x = np.vstack((self.s_tm1[:2,0].reshape((2,1)), self.a_tm1)).T
        x = np.vstack((self.s_tm1[:2,0].reshape((2,1)), a_tm1_)).T
        # print "x", x
        y = s[2,0] + self.gamma * q_fa
        if True or self.q_fa_training_cnt > 100 or s[2,0] > 0.0:
            target_weight = np.ones((1,)) + s[2] * 10.0
            self.q_fa_training_loss = self.q_fa.train_on_batch(x * 1.0, y * 1.0, sample_weight = target_weight)  # starts training
            self.q_fa_training_cnt += 1
        
    def q_Q_fa_predict(self, s, a):
        a_ = np.zeros((len(self.actions),1))
        a_[a[0,0],0] = 1.0
        x = np.vstack((s[:2,0].reshape((2,1)), a_))
        # x = np.vstack((s[:2,0].reshape((2,1)), a))
        return self.q_Q_fa.predict(x.T)
        
    def q_Q_fa_update(self, s, a):
        # print "s", s
        a_tm1_ = np.zeros((len(self.actions),1))
        a_tm1_[self.a_tm1[0,0],0] = 1.0
        
        # q_Q_fa_tm1 = self.q_Q(self.s_tm1, self.a_tm1)
        q_Q_fa_ = []
        for a_ in range(len(self.actions)):
            q_Q_fa_.append(self.q_Q(self.s, np.array([[a_]])))
        q_Q_fa_ = np.array([q_Q_fa_])
        q_Q_fa_max = np.max(q_Q_fa_)
        q_Q_fa_max = np.array([[q_Q_fa_max]]) # ?
        # print "argmax", q_Q_fa_max
        x = np.vstack((self.s_tm1[:2,0].reshape((2,1)), a_tm1_)).T
        y = s[2,0] + self.gamma * q_Q_fa_max
        # print "x", x, "y", y
        if True or self.q_Q_fa_training_cnt > 100 or s[2,0] > 0.0:
            target_weight = np.ones((1,)) + s[2] * 10.0
            self.q_Q_fa_training_loss = self.q_Q_fa.train_on_batch(x, y, sample_weight = target_weight)  # starts training
            self.q_Q_fa_training_cnt += 1
        
    def q_SARSA_fa_predict(self, s, a):
        a_ = np.zeros((len(self.actions),1))
        a_[a[0,0],0] = 1.0
        x = np.vstack((s[:2,0].reshape((2,1)), a_))
        # x = np.vstack((s[:2,0].reshape((2,1)), a))
        return self.q_SARSA_fa.predict(x.T)
        
    def q_SARSA_fa_update(self, s, a):
        # print "s", s
        a_tm1_ = np.zeros((len(self.actions),1))
        a_tm1_[self.a_tm1[0,0],0] = 1.0
        
        q_SARSA_fa = self.q_SARSA(s, a)
        x = np.vstack((self.s_tm1[:2,0].reshape((2,1)), a_tm1_)).T
        y = s[2,0] + self.gamma * q_SARSA_fa
        if True or self.q_SARSA_fa_training_cnt > 100 or s[2,0] > 0.0:
            target_weight = np.ones((1,)) + s[2] * 10.0
            self.q_SARSA_fa_training_loss = self.q_SARSA_fa.train_on_batch(x, y, sample_weight = target_weight)  # starts training
            self.q_SARSA_fa_training_cnt += 1

    ################################################################################

    def update_get_indices(self, s, s_tm1, a_tm1):
        l_x = int(s[0,0])
        l_y = int(s[1,0])
        l_x_tm1 = int(s_tm1[0,0])
        l_y_tm1 = int(s_tm1[1,0])
        l_a_tm1   = int(a_tm1[0,0])
        return (l_x, l_y, l_x_tm1, l_y_tm1, l_a_tm1)
    
    def v_tbl_predict(self, s):
        l_x = int(s[0,0])
        l_y = int(s[1,0])
        return self.v_tbl[l_x, l_y]

    def q_tbl_predict(self, s, a):
        l_x = int(s[0,0])
        l_y = int(s[1,0])
        l_a = int(a[0,0])
        return self.q_tbl[l_x, l_y, l_a]

    def q_Q_tbl_predict(self, s, a):
        l_x = int(s[0,0])
        l_y = int(s[1,0])
        l_a = int(a[0,0])
        return self.q_Q_tbl[l_x, l_y, l_a]

    def q_SARSA_tbl_predict(self, s, a):
        l_x = int(s[0,0])
        l_y = int(s[1,0])
        l_a = int(a[0,0])
        return self.q_SARSA_tbl[l_x, l_y, l_a]
        
    def v_tbl_update(self, s):
        l_x, l_y, l_x_tm1, l_y_tm1, l_a_tm1 = self.update_get_indices(s, self.s_tm1, self.a_tm1)
        
        # back up old state value once
        # self.v_tbl_s_tm1 = self.v_tbl[l_x_tm1, l_y_tm1].copy()
        self.v_tbl_s_tm1 = self.v(self.s_tm1).copy()
        # perform update, SB2nded pg. ?, eq. ?
        # self.v_tbl[l_x_tm1, l_y_tm1] = self.v_tbl_s_tm1 + self.alpha * 0.1 * (s[2,0] + self.gamma * self.v_tbl[l_x, l_y] - self.v_tbl_s_tm1)
        self.v_tbl[l_x_tm1, l_y_tm1] = self.v_tbl_s_tm1 + self.alpha * 0.1 * (s[2,0] + self.gamma * self.v(s) - self.v_tbl_s_tm1)

    def q_tbl_update(self, s, a):
        l_x, l_y, l_x_tm1, l_y_tm1, l_a_tm1 = self.update_get_indices(s, self.s_tm1, self.a_tm1)
        
        # back up old state-action value once
        # self.q_tbl_sa_tm1 = self.q_tbl[l_x_tm1, l_y_tm1, l_a_tm1].copy()
        self.q_tbl_sa_tm1 = self.q(self.s_tm1, self.a_tm1).copy()
        # perform update, SB2nded pg. ?, eq. ?
        # self.q_tbl[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_tbl_sa_tm1 + self.alpha * (self.s[2,0] + self.gamma * self.q_tbl[l_x, l_y, l_a_tm1] - self.q_tbl_sa_tm1)
        self.q_tbl[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_tbl_sa_tm1 + self.alpha * (self.s[2,0] + self.gamma * self.q(s, self.a_tm1) - self.q_tbl_sa_tm1)

    def q_Q_tbl_update(self, s, a):
        l_x, l_y, l_x_tm1, l_y_tm1, l_a_tm1 = self.update_get_indices(s, self.s_tm1, self.a_tm1)
        
        # back up old state-action value once Q-Learning
        # self.q_Q_tbl_tm1 = self.q_Q_tbl[l_x_tm1, l_y_tm1, l_a_tm1].copy()
        self.q_Q_tbl_tm1 = self.q_Q(self.s_tm1, self.a_tm1).copy()
        # perform update, SB2nded pg. ?, eq. ?
        # print "q_Q update max(Q_q(S, a))", np.max(self.q_Q_tbl[l_x, l_y, l_a_tm1])
        # print "self.q_Q_tbl[l_x, l_y, l_a_tm1]", self.q_Q_tbl[l_x, l_y, :]
        self.q_Q_tbl[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_Q_tbl_tm1 + self.alpha * (self.s[2,0] + self.gamma * np.max(self.q_Q_tbl[l_x, l_y, :]) - self.q_Q_tbl_tm1)
        # self.q_Q_tbl[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_Q_tbl_tm1 + self.alpha * (self.s[2,0] + self.gamma * np.max(self.q_Q_tbl[l_x, l_y, l_a_tm1]) - self.q_Q_tbl_tm1)
    def q_SARSA_tbl_update(self, s, a):
        l_x, l_y, l_x_tm1, l_y_tm1, l_a_tm1 = self.update_get_indices(s, self.s_tm1, self.a_tm1)
        
        # back up old state-action value once Q-Learning
        # self.q_SARSA_tbl_tm1 = self.q_SARSA_tbl[l_x_tm1, l_y_tm1, l_a_tm1].copy()
        self.q_SARSA_tbl_tm1 = self.q_SARSA(self.s_tm1, self.a_tm1).copy()
        # perform update, SB2nded pg. ?, eq. ?
        # self.q_SARSA_tbl[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_SARSA_tbl_tm1 + self.alpha * (self.s[2,0] + (self.gamma * self.q_SARSA_tbl[l_x, l_y, self.a]) - self.q_SARSA_tbl_tm1)
        self.q_SARSA_tbl[l_x_tm1, l_y_tm1, l_a_tm1] = self.q_SARSA_tbl_tm1 + self.alpha * (self.s[2,0] + (self.gamma * self.q_SARSA(s, a)) - self.q_SARSA_tbl_tm1)
                
    # policies
    def policy(self, q, s, epsilon = 0.0):
        return self.policy_func(q, s)
        
    def policy_random(self, q, s):
        return np.random.randint(len(self.actions), size=self.a.shape)

    def policy_epsilon_greedy(self, q, s, epsilon = 0.05):
        if np.random.uniform() < epsilon:
            return self.policy_random(q, s)
        else:
            # get best action according to current q estimate
            q_s = q[int(s[0,0]), int(s[1,0])]
            # print "%s.policy_epsilon_greedy q_s = %s" % (self.__class__.__name__, q_s)
            a_s = np.argmax(q_s).reshape(self.a.shape)
            # print "%s.policy_epsilon_greedy a_s = %s" % (self.__class__.__name__, a_s)
            return a_s
        
    def step(self, s):
        # stop episode
        if self.terminal:
            self.terminal_ -= 1
            if self.repr == "approximation":
                if not hasattr(self, "avg_loss"):
                    self.avg_loss = 0.0
                self.avg_loss = 0.9 * self.avg_loss + 0.1 * np.sum([self.v_fa_training_loss, self.q_fa_training_loss, self.q_Q_fa_training_loss, self.q_SARSA_fa_training_loss])
                print "tc", self.v_fa_training_cnt, self.v_fa_training_loss, self.q_fa_training_cnt, self.q_fa_training_loss, self.q_Q_fa_training_cnt, self.q_Q_fa_training_loss, self.q_SARSA_fa_training_cnt, self.q_SARSA_fa_training_loss
                print "avg loss", self.avg_loss
        # sensory measurement: [x, y, reward].T
        self.s = s.copy()
        # print "%s.step s = %s" % (self.__class__.__name__, self.s)

        # current state
        l_x = int(self.s[0,0])
        l_y = int(self.s[1,0])
        # last state
        l_x_tm1 = int(self.s_tm1[0,0])
        l_y_tm1 = int(self.s_tm1[1,0])
        l_a_tm1   = int(self.a_tm1[0,0])
        # print "l", l_x, l_y, "l_tm1", l_x_tm1, l_y_tm1
                
        # update v
        # print "v", l_x, l_y, self.v_tbl[l_x, l_y]

        # update value functions
        # v
        self.v_update(self.s)

        # q with td0 update
        self.q_update(self.s, self.a)
                
        # q with Q update
        self.q_Q_update(self.s, self.a)
        
        # policy: some functional thing that produces an action
        if self.sensorimotor_loop == "td_0_prediction":
            self.a = self.policy(self.q_tbl, self.s)
        elif self.sensorimotor_loop == "td_0_off_policy_control":
            # back up old q_Q for off policy foo
            self.a = self.policy(self.q_Q_tbl, self.s, epsilon = self.epsilon)
        elif self.sensorimotor_loop == "td_0_on_policy_control":
            self.a = self.policy(self.q_SARSA_tbl, self.s, epsilon = self.epsilon)
        # print self.a
        
        # q with sarsa update
        self.q_SARSA_update(self.s, self.a)
        
        # back up state
        self.s_tm1 = self.s.copy()
        # back up action
        self.a_tm1 = self.a.copy()
        self.t += 1
        return self.a

################################################################################
# operations
    
def plot_init(ev):
    plt.ion()
    fig = plt.figure()
    # sensorimotor_loop
    smls = []
    for a in ev.agents:
        smls.append(a.sensorimotor_loop)
    
    fig.suptitle("TD(0) learning of v and q, %d agents using %s" % (len(ev.agents), ", ".join(smls)))
    gs_numcol = 1 + 1 # 1 + 1 + 4 # 3
    gs = gridspec.GridSpec(len(ev.agents) * 4, gs_numcol)
    axs = []
    # plt.subplots_adjust(left=0.2)
    # plt.subplots_adjust(bottom=-0.2)


    for i, a in enumerate(ev.agents):
        # # subplothost foo double labels
        # ax_s       = SubplotHost(fig, gs[i*2+3,0])
        # ax_v       = SubplotHost(fig, gs[i*2+3,1])
        # ax_q       = SubplotHost(fig, gs[i*2,:])
        # ax_q_Q     = SubplotHost(fig, gs[i*2+1,:])
        # ax_q_SARSA = SubplotHost(fig, gs[i*2+2,:])
        
        axs.append([
            # fig.add_subplot(gs[gs_numcol*i]),
            # fig.add_subplot(gs[gs_numcol*i+1]),
            # fig.add_subplot(gs[gs_numcol*i+2:])

            # # subplothost foo double labels
            # fig.add_subplot(ax_s),
            # fig.add_subplot(ax_v),
            # fig.add_subplot(ax_q),
            # fig.add_subplot(ax_q_Q),
            # fig.add_subplot(ax_q_SARSA),
            
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
        ax_q = axs[-1][2]
        ax_q.set_title("Agent %d state-action value q(s,a)" % i, fontsize = 8)
        ax_q.set_xlabel("f(a, x)")
        ax_q.set_ylabel("y")
        ax_q.set_aspect(1)
        # ax_q.set_aspect((len(a.actions)*ev.num_x)/float(ev.num_y))
        # ax_q.set_aspect((len(a.actions)*ev.num_x)/float(ev.num_y))
        axs[-1][3].set_aspect(1)
        axs[-1][4].set_aspect(1)
        
    return fig, gs, axs

def plot_pcolor_coordinates():
    pass

def plot_draw_ev(fig, gs, axs, ev):
    for i, a in enumerate(ev.agents):
        # print "plot_draw_ev s_%d = %s" % (i, ev.s[i])
        
        # plot state
        ax_s = axs[i][0]
        # clean up
        ax_s.clear()

        # plot state
        # print "ev.s[i].shape", ev.s[i].shape, a.v_tbl.shape, a.q_tbl.shape
        ax_s.pcolormesh(ev.s[i].T, cmap=plt.get_cmap("gray"))
        # ax_s.pcolormesh(ev.s[i][::-1], cmap=plt.get_cmap("gray"))
        ax_s.plot([ev.goal[0,0] + 0.5], [ev.goal[1,0] + 0.5], "ro", markersize = 20, alpha= 0.5)
        ax_s.set_title("Agent %d state (position on grid)" % i, fontsize=8)
        ax_s.set_xlabel("x")
        ax_s.set_ylabel("y")
        ax_s.set_aspect(1)

        # meshgrid
        # v
        v_img = np.zeros((ev.num_x, ev.num_y))
        for k in range(ev.num_x):
            for l in range(ev.num_y):
                v_img[k,l] = a.v(np.array([[k, l, 0]]).T)
        ev.agents[i].v_tbl = v_img.T

        # q
        q_img = np.zeros((ev.num_x, ev.num_y, 9))
        for k in range(ev.num_x):
            for l in range(ev.num_y):
                for m in range(9):
                    q_img[k,l,m] = a.q(np.array([[k, l]]).T, np.array([[m]]).T)
        # q_img_full = ev.agents[i].q_tbl
        ev.agents[i].q_tbl = q_img.copy().transpose([0, 1, 2])

        # q_Q
        q_Q_img = np.zeros((ev.num_x, ev.num_y, 9))
        for k in range(ev.num_x):
            for l in range(ev.num_y):
                for m in range(9):
                    q_Q_img[k,l,m] = a.q_Q(np.array([[k, l]]).T, np.array([[m]]).T)
        ev.agents[i].q_Q_tbl = q_Q_img.copy().transpose([0, 1, 2])

        # q_SARSA
        q_SARSA_img = np.zeros((ev.num_x, ev.num_y, 9))
        for k in range(ev.num_x):
            for l in range(ev.num_y):
                for m in range(9):
                    q_SARSA_img[k,l,m] = a.q_SARSA(np.array([[k, l]]).T, np.array([[m]]).T)
        ev.agents[i].q_SARSA_tbl = q_SARSA_img.copy().transpose([0, 1, 2])
        
        # plot state value
        ax_v = axs[i][1]
        ax_v.clear()
        # v_img = np.log(ev.agents[i].v_tbl + 1.0)
        v_img = ev.agents[i].v_tbl
        ax_v.pcolormesh(v_img, cmap=plt.get_cmap("gray"))#, vmin = 0.0) # , vmax = 1.0)
        ax_v.set_title("Agent %d state value v(s)" % i, fontsize = 8)
        ax_v.set_xlabel("x")
        ax_v.set_ylabel("y")
        ax_v.set_aspect(1)

        # plot state-action value
        ax_q = axs[i][2]
        ax_q.clear()
        ax_q.set_title("Q_{TD(0)", fontsize=8)
        q_img = ev.agents[i].q_tbl
        print "q_img", np.min(q_img), np.max(q_img)
        q_img = dimensional_stacking(np.transpose(q_img, [1, 0, 2]), [2, 1], [0])
        # print "q_img.shape", q_img.shape
        ax_q.pcolormesh(q_img, cmap=plt.get_cmap("gray"))#, vmin = 0.0)#, vmax = 2.0)
        ax_q.set_title("Agent %d state-action value q(s,a)" % i, fontsize = 8)
        # ax_q.set_xlabel("f(a, x)")
        ax_q.set_ylabel("y")
        # ax_q.set_aspect((len(a.actions)*ev.num_x)/float(ev.num_y))
        # ax_q.set_aspect((len(a.actions)*ev.num_x)/float(ev.num_y))
        ax_q.set_xticks([])
        
        # ax_q_x = ax_q.twiny()
        # # ax_q_x.set_xlim((0, 3))
        # offset = 0.0, -25
        # new_axisline = ax_q_x.get_grid_helper().new_fixed_axis
        # ax_q_x.axis["bottom"] = new_axisline(loc="bottom", axes=ax_q_x, offset=offset)
        # ax_q_x.axis["top"].set_visible(False)

        # ax_q.set_xticks(np.arange(5+1))# + 0.5)
        # # ax_q.set_xticklabels(np.tile(measures.values(), 3))

        # ax_q_x.set_xticks(np.arange(9+1))# + 0.5)
        # # ax_q_x.set_xticklabels(robots.values())
        # ax_q_x.set_aspect(1)
        
        # plot state-action value
        ax_q_Q = axs[i][3]
        ax_q_Q.clear()
        ax_q_Q.set_title("Q_{Q}, min = %f, max = %f" % (np.min(ev.agents[i].q_Q_tbl), np.max(ev.agents[i].q_Q_tbl)), fontsize=8)
        q_Q_img = ev.agents[i].q_Q_tbl
        print "q_Q_img", np.min(q_Q_img), np.max(q_Q_img)
        q_img_Q = dimensional_stacking(np.transpose(q_Q_img, [1, 0, 2]), [2, 1], [0])
        # q_img = dimensional_stacking(ev.agents[i].q_SARSA_tbl, [2, 1], [0])
        # print "q_img.shape", q_img.shape
        ax_q_Q.pcolormesh(q_img_Q, cmap=plt.get_cmap("gray"))#, vmin = 0.0) #, vmax = 2.0)
        ax_q_Q.set_aspect(1)
        ax_q_Q.set_xticks([])

        # plot state-action value
        ax_q_SARSA = axs[i][4]
        ax_q_SARSA.clear()
        ax_q_SARSA.set_title("Q_{SARSA} min = %f, max = %f" % (np.min(ev.agents[i].q_SARSA_tbl), np.max(ev.agents[i].q_SARSA_tbl)), fontsize=8)
        q_SARSA_img = ev.agents[i].q_SARSA_tbl
        print "q_SARSA_img", np.min(q_SARSA_img), np.max(q_SARSA_img)
        q_img_SARSA = dimensional_stacking(np.transpose(q_SARSA_img, [1, 0, 2]), [2, 1], [0])
        # print "q_img.shape", q_img.shape
        mpabl = ax_q_SARSA.pcolormesh(q_img_SARSA, cmap=plt.get_cmap("gray"))#, vmin = 0.0, vmax = 5.0)
        ax_q_SARSA.set_aspect(1)
        # plt.colorbar(mpabl, ax=ax_q_SARSA, orientation="horizontal")
        
        ax_q_SARSA.set_xticks(np.arange(0, 5*9, 2.5))
        ticklabels = ["x=x,a=nop", "x=x,a=w", "x=x,a=nw", "x=x,a=n", "x=x,a=ne", "x=x,a=e", "x=x,a=se", "x=x,a=s", "x=x,a=sw"]
        # ticklabels.insert(0, "")
        ticklabels2 = []

        for i_q_tl, q_tl in enumerate(ticklabels):
            ticklabels2.append("")
            ticklabels2.append(q_tl)
        ticklabels2.append("")
        ax_q_SARSA.set_xticklabels(ticklabels2, fontsize=8)
                        
    plt.draw()
    plt.pause(1e-3)
        

def get_agent(args):
    # if args.sensorimotor_loop == "td_0_prediction":
    # return TD0PredictionAgent(ndim_s = 3, ndim_a = 1, ndim_x = args.world_x, ndim_y = args.world_y, alpha = args.alpha, gamma = args.gamma)
    return TD0PredictionAgent(args)
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
    
    setattr(args, "ndim_s", 3)
    setattr(args, "ndim_a", 1)
    setattr(args, "ndim_x", args.world_x)
    setattr(args, "ndim_y", args.world_y)

    if args.sensorimotor_loop == "td0":
        args.sensorimotor_loop = "td_0_prediction"
    elif args.sensorimotor_loop in ["q", "Q"]:
        args.sensorimotor_loop = "td_0_off_policy_control"
    elif args.sensorimotor_loop in ["sarsa", "SARSA"]:
        args.sensorimotor_loop = "td_0_on_policy_control"
    
    ag = get_agent(args)
    # ag2 = TD0PredictionAgent(ndim_s = 3, ndim_a = 1)
    ev = GridEnvironment(agents = [ag], num_x = args.world_x, num_y = args.world_y)
    # ag.q_Q_tbl[ev.goal[0,0], ev.goal[1,0],:] = 0.1
    # ag.q_SARSA_tbl[ev.goal[0,0], ev.goal[1,0],:] = 0.0

    # s = ag.s
    # a = ag.a

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
            if (i * args.maxsteps + t) % args.plotfreq == 0:
                print "plotting at step %d" % (i * args.maxsteps + t)
                plot_draw_ev(fig, gs, axs, ev)
            terminal = np.all(np.array([agent.terminal_ < 1 for agent in ev.agents]))
            t += 1
        print "epi %d, final step %d" % (i, t)

    print "ev.steps = %d" % (ev.t)
    print "ag.steps = %d" % (ag.t)

    # save result
    for i, agent in enumerate(ev.agents):
        np.save("td0_ag%d_v.npy" % i, agent.v_tbl)
        np.save("td0_ag%d_q.npy" % i, agent.q_tbl)

    plt.ioff()
    plt.show()
            
def main(args):
    rl_experiment(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a",  "--alpha",   default=1e-2,  type=float, help="Learning rate \alpha [0.01]")
    parser.add_argument("-e",  "--epsilon", default=0.1,   type=float, help="\epsilon-greedy \epsilon [0.1]")
    parser.add_argument("-g",  "--gamma",   default=0.8,  type=float, help="Discount factor \gamma [0.8]")
    parser.add_argument("-ne", "--numepisodes", default=500,  type=int, help="Number of episodes [500]")
    parser.add_argument("-ms", "--maxsteps",    default=100, type=int, help="Maximum number of steps per episodes [100]")
    parser.add_argument("-sm", "--sensorimotor_loop", default="td_0_prediction", type=str, help="Which sm loop (Learner), one of " + ", ".join(sensorimotor_loops) + " [td_0_prediction]")
    parser.add_argument("-p",  "--plotfreq", default=1000,    type=int, help="Plotting interval in steps [1000]")
    parser.add_argument("-r",  "--repr",     default="table", type=str, help="Value function representation [table]")
    parser.add_argument("-wx",  "--world_x", default=5, type=int, help="Size of world along x [5]")
    parser.add_argument("-wy",  "--world_y", default=5, type=int, help="Size of world along y [5]")
    
    args = parser.parse_args()
    
    main(args)
    # main_extended(args)

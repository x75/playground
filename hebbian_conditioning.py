"""
Simplified version of DAC example from lecture slides Lecture 6 (part ii)

2017 Oswald Berthold 

Environment: 1-dimensional arena enclosed by two boundaries

Robot: Velocity controlled robot moving back and forth in the arena

Sensors: Collision, Proximity, unused: Velocity, Velocity prediction

Goal: Stay away from boundaries

Basic reflexes: Jump back reflex hardwired to collision

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def sm_loop_1(i, x, X, idx, conf):
    """The network architecture is slightly different than the DAC example, here we don't
    learn the weights from proximity to collision layer but from proximity to motor. Can
    of course be modified to comply with DAC example.
    """

    pos, vel, vel_, col, prox, col_rfr, col_, weight, col__rfr = idx
    
    # measurement
    x[prox,0] = np.abs(x[pos,0]) + np.random.normal(0, 5e-2)

    # predict collision: proximity sensor times weight
    x[col_,0] = x[prox,0] * x[weight,0]
        
    # predicted collision threshold?
    if np.abs(x[col_,0]) > 0.6:
        x[col__rfr,0] = 1.0
                
    # real collision sensor
    if np.abs(x[pos,0]) > conf["limit"]: #or x[col_,0] > :
        x[col,0] = 1.0

    # Hebbian learning, the weight change = learning rate x pre-synaptic with fixed delay x post-synaptic
    dw = conf["eta"] * X[i-conf["tau"],prox] * x[col,0]

    # Stabilization for Hebbian learning
    # weight bounding
    # x[weight,0] += dw * (1.0 - min(max(0, x[weight,0] - 3.0), 1.0))
    
    # normalization
    x[weight,0] += dw # update
    x[weight,0] /= max(1, x[weight,0] * 0.125) # normalize
    
    # collision reflex window to suppress random walk
    x[col_rfr,0] = x[col,0] > 5e-2
    
    # motor circuit
    # random velocity gated by reflex activity
    v_rnd  =  np.random.uniform(-0.7, 0.7) * (1 - x[col_rfr,0]) * (1 - x[col__rfr,0])
    # velocity component from reflex triggered by collision
    v_col  = -np.sign(x[pos,0]) * x[col_rfr,0] * x[col,0]
    # velocity component from reflex triggered by predicted collision
    v_col_ = -np.sign(x[pos,0]) * x[col__rfr,0] * 1 # x[col_,0]
    # sum them up for final velocity command
    x[vel_,0] = v_rnd + v_col + v_col_

    return x

def do_plot_1(X, idx, conf):
    pos, vel, vel_, col, prox, col_rfr, col_, weight, col__rfr = idx
    
    # plotting
    plt.suptitle("Hebbian learning quasi-DAC / fixed delay ISO")
    plt.subplot(411)
    plt.title("Position, proximity sensor, boundaries")
    plt.plot(X[:,pos], "k-", label="pos")
    plt.plot([0, conf["numsteps"]], [conf["limit"],   conf["limit"]], "r-")
    plt.plot([0, conf["numsteps"]], [-conf["limit"], -conf["limit"]], "r-")
    plt.plot(X[:,prox], "g-", label="prox")
    plt.legend()
    
    plt.subplot(412)
    plt.title("Velocity and velocity prediction for t + 1")
    plt.plot(X[:,vel], "k-", label="vel")
    plt.plot(X[:,vel_], "k-", label="vel_", alpha = 0.5)
    plt.legend()
    
    plt.subplot(413)
    plt.title("Collision, collision reflex window, collision predicted")
    plt.plot(X[:,col], "k-", label = "col")
    plt.plot(X[:,col_], "r-", label = "col_", alpha = 0.5)
    plt.plot(X[:,col_rfr], "k-", label = "col_rfr", alpha = 0.5)
    plt.legend()
    
    plt.subplot(414)
    plt.plot(X[:,col_], "k-", label = "col_")
    plt.plot(X[:,weight], "b-", label = "weight", alpha = 0.5)
    plt.plot(X[:,col__rfr], "g-", label = "col__rfr", alpha = 0.5)
    plt.legend()
    plt.show()


def g(x, thresh):
    return x > thresh
    
def sm_loop_2(i, x, X, idx, conf):
    """DAC style conditioned reflex"""
    # need h_1, h_2, c_1, c_2, p_1, p_2, w_11, w_12, w_21, w_22, a_1, a_2, a_bar
    pos, vel, h_1, h_2, c_1, c_2, p_1, p_2, w_11, w_12, w_21, w_22, c_1_, c_2_, a_1, a_2, a_bar = idx
    weight_idx = [w_11, w_12, w_21, w_22]
    
    # proximity sensor p
    x[p_1,0] = max(0, x[pos,0]) + np.random.normal(0, 1e-2) #
    x[p_2,0] = min(0, x[pos,0]) + np.random.normal(0, 1e-2) #

    # collision sensor prediction: weight x proximity sensor p
    x[[c_1_, c_2_],0] = np.dot(x[[w_11, w_12, w_21, w_22],0].reshape((2,2)), x[[p_1,p_2],0])
        
    # collision sensor c
    x[[c_1,c_2],0] += (x[[pos, pos],0] * np.array([1.0, -1.0])) > np.array([conf["limit"], conf["limit"]])

    # collision layer, h_i, combined activity collision sensor and collision sensor prediction
    x[[h_1, h_2],0] = x[[c_1_, c_2_],0] + x[[c_1,c_2],0]
        
    # motor circuit a, thresholded activity h
    x[[a_1, a_2],0] = g(x[[h_1, h_2],0], conf["thresh"]) # v_rnd + v_col # + v_col_
    
    # Hebbian learning DAC
    # the weight change = learning rate x pre-synaptic with fixed delay x post-synaptic - forgetting rate * mean motor activity a x w
    dw = ((conf["eta"] * np.outer(X[i-conf["tau"],[p_1,p_2]], x[[a_1,a_2],0]))) - (conf["epsilon"] * np.mean(x[[a_1,a_2],0]) * x[weight_idx,0].reshape((2, 2)))
    # # squelch small weight changes
    # dw = 0.5 * (np.abs(dw) > 1e-2) * dw
    # update
    x[weight_idx,0] += dw.flatten()

    # Stabilization for Hebbian learning
    # weight bounding
    # x[weight,0] += dw * (1.0 - min(max(0, x[weight,0] - 3.0), 1.0))
    
    # normalization
    # x[weight_idx,0] /= max(1, np.linalg.norm(x[weight_idx,0] * 0.125, 2))# np.clip(x[weight_idx,0] * 0.125, 1, np.inf) # normalize

    # some noise if no one else is talking on the line    
    if np.sum(x[[a_1,a_2],0]) <= 1e-3:
        v_rnd  =  np.random.uniform(0.0, 0.5, (2, 1)) # * (1 - x[col_rfr,0]) * (1 - x[col__rfr,0])
        x[[a_1,a_2],0] = v_rnd[[0,1],0]
    
    return x

def do_plot_2(X, idx, conf):
    pos, vel, h_1, h_2, c_1, c_2, p_1, p_2, w_11, w_12, w_21, w_22, c_1_, c_2_, a_1, a_2, a_bar = idx
    
    # plotting
    plt.suptitle("Hebbian learning DAC")
    plt.subplot(411)
    plt.title("Position, boundaries, and proximity sensors p")
    plt.plot(X[:,pos], "k-", label="pos")
    plt.plot([0, conf["numsteps"]], [conf["limit"],   conf["limit"]], "r-")
    plt.plot([0, conf["numsteps"]], [-conf["limit"], -conf["limit"]], "r-")
    plt.plot(X[:,p_1], "g-", label="p_1", alpha=0.5)
    plt.plot(X[:,p_2], "b-", label="p_2", alpha=0.5)
    plt.legend()
    
    plt.subplot(412)
    plt.title("Velocity and velocity prediction a")
    plt.plot(X[:,vel], "k-", label="vel")
    plt.plot(X[:,a_1], "r-", label="a_1", alpha = 0.5)
    plt.plot(X[:,a_2], "r-", label="a_2", alpha = 0.5)
    plt.legend()
    
    plt.subplot(413)
    plt.title("Collision sensors c, collision prediction c_, combined activity h")
    plt.plot(X[:,c_1], "k-", label = "c_1")
    plt.plot(X[:,c_2], "b-", label = "c_2")
    plt.plot(X[:,c_1_], "g-", label = "c_1_", alpha = 0.5)
    plt.plot(X[:,c_2_], "c-", label = "c_2_", alpha = 0.5)
    plt.plot(X[:,h_1], "r-", label = "h_1")
    plt.plot(X[:,h_2], "y-", label = "h_2")
    plt.legend()
    
    plt.subplot(414)
    plt.title("Weights w")
    # print weight norm
    # weight = np.sum(np.square(X[:,[w_11,w_12,w_21,w_22]]), axis=1)
    # plt.plot(weight, "b-", label = "weight", alpha = 0.5)
    # print single weights
    weight = X[:,[w_11,w_12,w_21,w_22]]
    plt.plot(weight, label = "w", alpha = 0.5)
    plt.legend()
    
    plt.show()


def main(args):
    """Simple 1D system performing random walk, on contact with the boundaries a reflex
    to move away is triggered by a collision sensor. Hebbian learning is used to
    predict the trigger condition and avoid an actual collision.

    The reflexes can be seen in the plot as spikes coming above the noise threshold.
    """
    
    # initialization
    conf = {
        "numsteps": args.numsteps, # number of steps
        "dt": 0.1, # time interval
        "limit": 0.5, # arena boundaries
        "tau": 3, # prediction horizon, avg proximity/collision delay
        "eta": 1e-1,     # learning rate
    }
    
    # state variables: pos, vel, vel pred, collision sensor, proximity sensor, collision reflex window, collision prediction, weight, collision prediction thresholded
        
    if args.sensorimotor_loop == 1:
        sm_loop = sm_loop_1
        do_plot = do_plot_1
        dim = 9
        # indices
        idx = range(dim)
        pos, vel, vel_, col, prox, col_rfr, col_, weight, col__rfr = idx
        
        # world state transition
        A = np.eye(dim)
        A[pos,vel]   = conf["dt"]
        A[vel,vel]   = 0
        A[vel,vel_]  = 1.0
        A[col,col] = 0.8
        A[col__rfr,col__rfr] = 0.8

    elif args.sensorimotor_loop == 2:
        sm_loop = sm_loop_2
        do_plot = do_plot_2
        dim = 17
        # indices
        idx = range(dim)
        pos, vel, h_1, h_2, c_1, c_2, p_1, p_2, w_11, w_12, w_21, w_22, c_1_, c_2_, a_1, a_2, a_bar = idx

        conf["thresh"] = 0.5
        conf["eta"]     = 1e-0
        conf["epsilon"] = 1e-1
        
        # world state transition
        A = np.zeros((dim, dim))
        A[pos,pos]   = 1.0
        A[pos,vel]   = 0.0 # conf["dt"]
        A[pos,a_1]   = -3.0 * conf["dt"]
        A[pos,a_2]   =  3.0 * conf["dt"]
        A[vel,a_1]   = -3.0
        A[vel,a_2]   = 3.0
        A[w_11,w_11]   = 1.0
        A[w_12,w_12]   = 1.0
        A[w_21,w_21]   = 1.0
        A[w_22,w_22]   = 1.0
        # A[vel,vel]   = 0
        # A[vel,vel_]  = 1.0
        # A[col,col] = 0.8
        # A[col__rfr,col__rfr] = 0.8
        

    x = np.zeros((dim, 1))

    # logging
    X = np.zeros((conf["numsteps"], dim))
    
    # sensorimotor loop
    for i in range(conf["numsteps"]):
        x = sm_loop(i, x, X, idx, conf)

        # print "|W| outer", np.linalg.norm(x[[w_11, w_12, w_21, w_22],0], 2)
                
        # logging
        X[i] = x.T

        # update one timestep
        x = np.dot(A, x)
    
    do_plot(X, idx, conf)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numsteps", default=10000, type=int, help="Number of timesteps")
    parser.add_argument("-sm", "--sensorimotor_loop", default=1, type=int, help="Which sm loop?")
    
    args = parser.parse_args()
    
    main(args)
    # main_extended(args)

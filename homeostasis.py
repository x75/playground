"""homeostasis example according to playfulmachines, ca. pgs. 67"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# fig = plt.figure()
# ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
# line, = ax.plot([], [], lw=2)


# def init():
#     line.set_data([], [])
#     return line,

def main(args):
    ndim_s = 2
    ndim_m = 1
    numsteps = args.numsteps

    # system
    angle = np.random.uniform(-1.0 * np.pi/4.0, -3.0 * np.pi/4.0, size=(1,1)) # np.ones((1, 1)) * 10.0
    angleSpeed = np.ones_like(angle) * 0.0
    l = 1
    g = -0.01
    friction = 0.99
    motorTorque = 0.009

    # brain
    x = np.zeros((ndim_s, 1))
    xPred = np.zeros_like(x)
    xError = np.zeros_like(x)
    y = np.zeros((ndim_m, 1))

    A = np.zeros([ndim_s, ndim_m])
    b = np.zeros_like(x)

    C = np.random.uniform(-1e-1, 1e-1, size=(ndim_m, ndim_s))
    h = np.random.uniform(-1e-3, 1e-3, size=y.shape) # ones_like(y) * 0.1

    epsA = 0.01
    epsC = 0.1
    # global angle, angleSpeed, A, C, h, b, y

    # gd m'fin logging
    x_ = np.zeros((numsteps,) + x.shape)
    xPred_ = np.zeros((numsteps,) + xPred.shape)
    xError_ = np.zeros((numsteps,) + xError.shape)
    y_ = np.zeros((numsteps,) + y.shape)
    Anorm_ = np.zeros((numsteps,1))
    Cnorm_ = np.zeros((numsteps,1))
    angle_ = np.zeros((numsteps,) + angle.shape)
    angleSpeed_ = np.zeros((numsteps,) + angleSpeed.shape)
    
    # Feed forward model

    for i in range(numsteps):
        # new measurement
        # print("angle", angle)
        x = np.array([[np.sin(angle[0,0])], [np.cos(angle[0,0])]]) # ,[2,1])
        # print("x:", x)

        xError = x - xPred
        # print("xError: ", xError)

        # # xError = np.dot(xError.T, xError)

        # # print("xError: ", xError)

        # Train Model
        dA = epsA * xError * y
        A += dA
        db = epsA * xError
        b += db

        Anorm = np.linalg.norm(A, 2)
        
        # print("|A| = %f, |dA| = %f" % (Anorm, np.linalg.norm(dA, 2)))
        # print("|b| = %f, |db| = %f" % (np.linalg.norm(b, 2), np.linalg.norm(db, 2)))

        # Train Controller
        z = np.dot(C, x) + h
        # print("z:", z, z.shape)
        g_z = 1 - np.power(np.tanh(z),2)
        # print("g_z:", g_z, g_z.shape)

        # eta = np.zeros((ndim_m, 1))
        # for row in range(A.shape[0]):
        #     eta += A[row,0] * g_z * xError[row,0]
        eta = np.dot(A.T, xError)
        # y
        # print("eta.shape", eta.shape)
        assert eta.shape == (1,1)
        
        dC = epsC * np.dot(eta * g_z, x.T) # np.zeros_like(C)
        dh = epsC * eta * g_z
        # print("dC.shape", dC.shape)
        # dC = np.zeros_like(C)
        # dh = np.zeros_like(h)
        C += dC
        h += dh

        Cnorm = np.linalg.norm(C, 2)
        
        # # TODO: hacky?
        # n = (np.dot(A, g_z) * xError).T
        # print("n:", n)

        # C += epsC * n * 1.0

        # C += epsC * A * (1 - np.power(np.tanh(C * x + h), 2)) * xError * x
        # h += epsC * A * (1 - np.power(np.tanh(C * x + h), 2)) * xError


        # #    print("A b C h:", A, b, C, h)

        # Control ##

        # K(x) = tanh(Cx + h)
        # print(C.shape, x.shape, h.shape)
        y = np.tanh(np.dot(C, x) + h)
        # print("y:", y)
    
        # predict next sensor state
        xPred = np.dot(A, y) + b

        # Dynamics model ##

        # motor Torque
        #if (int(i / 200) % 2 == 0):
        #    angleSpeed += 0.1
        # else:
        #	y = 0

        # angleSpeed += motorTorque * y[0][0]
        angleSpeed = motorTorque * y #[0][0]
        # friction
        angleSpeed *= friction

        # # gravity
        # angleSpeed += np.cos(angle) * g
        
        # calculate new position
        # angle += angleSpeed
        angle = angleSpeed

        # logging
        x_[i] = x
        xPred_[i] = xPred
        xError_[i] = xError
        Anorm_[i] = Anorm
        Cnorm_[i] = Cnorm
        y_[i] = y
        angle_[i] = angle
        angleSpeed_[i] = angleSpeed

    # print("x_.shape", x_.shape)
        
    plt.subplot(511)
    plt.plot(x_.reshape((numsteps, -1)), "k-", alpha=0.5, label="x")
    plt.plot(xPred_.reshape((numsteps, -1)) + 2, "b-", alpha=0.5, label="xP")
    plt.plot(xError_.reshape((numsteps, -1)) + 4, "r-", alpha=0.5, label="xE")
    plt.legend()
    plt.subplot(512)
    plt.plot(y_.reshape((numsteps, -1)), "k-", label="y")
    plt.legend()
    plt.subplot(513)
    plt.plot(angle_.reshape((numsteps, -1)), "k-", label="angle")
    plt.legend()
    plt.subplot(514)
    plt.plot(angleSpeed_.reshape((numsteps, -1)), "k-", label="angledot")
    plt.legend()
    plt.subplot(515)
    plt.plot(Anorm_.reshape((numsteps, -1)), "k-", label="|A|")
    plt.plot(Cnorm_.reshape((numsteps, -1)), "k-", label="|C|")
    plt.legend()
    plt.show()
        
# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=400, interval=20, blit=True)

# plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", "--numsteps", type=int, default=100)
    args = parser.parse_args()
    main(args)

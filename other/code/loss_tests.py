import matplotlib.pylab as plt
import numpy as np

vals = np.arange(-10, 10, 0.05)


def hubert():
    plt.plot(vals, vals**2, label='L2 loss')

    def hubert(x):
        if np.abs(x) < 1:
            return 0.5 * x**2
        else:
            return np.abs(x) - 0.5

    plt.plot(vals, [hubert(x) for x in vals], label='Smoothed L1 loss')
    plt.xlabel('$|\hat{y}_i - y_i |$')
    plt.ylabel('$z_i$')
    plt.legend(loc='upper right')

    plt.show()

def relu_comp():
    def relu(x):
        return x if x > 0 else 0
    def leaky_relu(x):
        return x if x > 0 else 0.01*x
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    def tanh(x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    plt.subplot(1,4, 1)
    plt.plot(vals, [relu(val) for val in vals], label='ReLU')
    plt.title('Rectified Linear')
    plt.subplot(1,4, 2)
    plt.plot(vals, [leaky_relu(val) for val in vals], label='Leaky ReLU')
    plt.title('Leaky Rectified Linear')
    plt.subplot(1,4, 3)
    plt.plot(vals, [sigmoid(val) for val in vals], label='$\sigma$')
    plt.title('Sigmoid function')
    plt.subplot(1,4, 4)
    plt.plot(vals, [tanh(val) for val in vals], label='tanh')
    plt.title('Hyperbolic tangent')

    #plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    relu_comp()
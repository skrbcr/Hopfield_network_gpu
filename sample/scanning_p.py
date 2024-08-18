
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
sys.path.append('.')
from Hopfield import Hopfield

if __name__ == '__main__':
    N = 1000
    M0 = 1

    ps = np.arange(100, 300, 30)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.set_title(f'Scanning the number of patterns (N = {N})')
    ax.set_xlabel('Steps')
    ax.set_ylabel('The number of patterns $p$')
    colormap = colormaps['jet'](np.linspace(0, 1, len(ps)))

    for i, p in enumerate(ps):
        hopfield = Hopfield(N, p)
        hopfield.memorize()
        hopfield.recall(M0, delta_m=1e-5)
        ax.plot(np.arange(len(hopfield.m)), hopfield.m, color=colormap[i], label=f'$p / N = {p / N:.2f}$')

    ax.legend()
    plt.show()


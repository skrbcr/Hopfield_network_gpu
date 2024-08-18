
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
sys.path.append('.')
from Hopfield import Hopfield

if __name__ == '__main__':
    N = 1000
    P = 100

    hopfield = Hopfield(N, P)
    m0s = np.arange(0, 1.0, 0.1)
    hopfield.memorize()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.set_title(f'Scanning initial overlap $m_0$ (N = {N}, P = {P})')
    ax.set_xlabel('Steps')
    ax.set_ylabel('$m$')
    colormap = colormaps['jet'](np.linspace(0, 1, len(m0s)))

    for i, m0 in enumerate(m0s):
        hopfield.recall(m0, delta_m=1e-3)
        ax.plot(np.arange(len(hopfield.m)), hopfield.m, color=colormap[i], label=f'$m_0 = {m0:.1f}$')

    ax.legend()
    plt.show()


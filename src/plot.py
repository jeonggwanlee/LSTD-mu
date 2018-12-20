import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

class Plot:
    def plot_rewad(self, x, y1, y2):

        plt.ylim(-220, 100)

        plt.plot(x, y1, 'bo-', linewidth=2.5, linestyle="-", label="LSPI-model-based LSTDQ")
        plt.plot(x, y2, 'ro-', linewidth=2.5, linestyle="-", label="LSPI-IS")
        plt.legend(loc='upper left')
        plt.show()

    

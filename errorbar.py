import numpy as np
import matplotlib.pyplot as plt

a = 0
b = 10
n = 11

x = np.linspace(a, b, n)
y = x[::-1]
xerr = 0.5
yerr = 0.5

'''
matplotlib.pyplot.errorbar(x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, 
                            barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, 
                            errorevery=1, capthick=None, hold=None, data=None, **kwargs)
'''

plt.errorbar(x, y,
             xerr=xerr,
             yerr=yerr,
             label='trend',
             fmt='-',
             color='g',
             ecolor='xkcd:salmon', elinewidth=1.5,
             capsize=5,
             #capthick=2
             )
plt.xlabel("features")
plt.ylabel("error")
plt.title("Plot of some functions")
plt.legend(loc="lower right")
plt.grid()
plt.show()

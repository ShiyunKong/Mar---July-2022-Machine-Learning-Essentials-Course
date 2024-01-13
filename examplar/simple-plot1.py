#start the code with the 3 lines: 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#define a style
plt.style.use('classic')
# create an array of 100 numbers from 0 to 10
x = np.linspace(0, 10, 100)
#prepare a figure and plot
fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');
#plot it
plt.show()
#save the figure into a file
fig.savefig('my_figure.png')

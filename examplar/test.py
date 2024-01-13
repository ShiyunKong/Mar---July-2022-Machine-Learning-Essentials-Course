#start the code with the 3 lines:
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np




# create an array of 100 numbers from 0 to 10
x = np.linspace(0, 10, 100)

# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)
# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x));
#plot it
plt.show()
#save the figure into a file
fig.savefig('my_figure.png')
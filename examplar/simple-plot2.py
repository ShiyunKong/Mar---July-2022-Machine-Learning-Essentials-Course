#start the code with the 3 lines: 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#define a style
plt.style.use('classic')

# create an array of 100 numbers from 0 to 10
x = np.linspace(0, 10, 100)
#prepare a figure and plot
#plt.figure()  # create a plot figure

# create the first of two panels and set current axis
plt.subplot(3, 2, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))

# create the second panel and set current axis
plt.subplot(3, 2, 4)
plt.plot(x, np.cos(x));
#show it
plt.show()


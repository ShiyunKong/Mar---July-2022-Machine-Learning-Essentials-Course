
#start the code with the 3 lines: 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# create an array of 100 numbers from 0 to 10
x = np.linspace(0, 10, 100)

plt.plot(x, np.sin (x- 0), color='blue') # specify color by name
plt.plot(x, np.sin (x -1), color='g') # short color code rgbcmyk
plt.plot(x, np.sin (x- 2), color='0.75 ') # Grayscale between 0 and 1
plt.plot(x, np.sin (x -3), color='#FFDD44') # Hex code (RRGGBB from 00 to
plt.plot(x, np.sin (x -4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 to 1
plt.plot(x, np.sin (x -5), color='chartreuse') # all HTML color names supported
plt.show()
         

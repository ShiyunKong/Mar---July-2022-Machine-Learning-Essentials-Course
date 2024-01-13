import numpy as np


x = np.linspace(0,5,5)
y = np.linspace(0,4,4)
X,Y = np.meshgrid(x,y)

print("X\n",X)
print("Y\n", Y)

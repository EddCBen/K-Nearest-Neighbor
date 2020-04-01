#4D plotter for iris features taking 4th feature as color on graph
#By: Charaf Eddine BENARAB

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_data(data_x):
    fig = plt.figure() 
    ax = fig.add_subplot(111,projection = '3d')
    x=[]
    y=[]
    z=[]
    c=[]
    for row in range(len(data_x)):
        x.append(data_x[row][0])
        y.append(data_x[row][1])
        z.append(data_x[row][2])
        c.append(data_x[row][3])
    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()
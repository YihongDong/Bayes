import numpy as np
import matplotlib.pyplot as plt
from math import pow,pi,sqrt,exp
from pylab import *

def parzen(label,feature,h,int):
    for i in range(len(label)):
        x = np.arange(float(feature[i])-h, float(feature[i])+h, 0.01)
        y = [float(1/(sqrt(2*pi))*exp(-1/2*pow(j-float(feature[i]),2))) for j in x]
        plot(x, y)

    y_total=[0 for i in np.arange(150,190,0.01)]
    x_total=np.arange(150,190,0.01)
    k=0
    for i in np.arange(150,190,0.01):
        for j in range(len(label)):
            if((float(feature[j])-h<=i) and (float(feature[j])+h>i)):
                y_total[k]+=float(1/(sqrt(2*pi))*exp(-1/2*pow(i-float(feature[j]),2)))
        k=k+1
    s='h='+str(h)
    plot(x_total,y_total)
    title(s)
    figure(int).show()
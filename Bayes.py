from math import pow,pi,sqrt,exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from readdata import *
from Normal_distribution import *
from parzen_windows import *
from roc_auc import *

theta_boy = input("please input the probability of the boys:")
theta_girl = 1-float(theta_boy)

def main():
    path_boy ="F:\\study in school\\machine learning\\forstudent\\实验数据\\boynew.txt"
    path_girl ="F:\\study in school\\machine learning\\forstudent\\实验数据\\girlnew.txt"
    height = []
    weight = []
    feetsize = []
    label = []  # 1表示男，0表示女
    readdata(path_boy,height,weight,feetsize,label,1)
    readdata(path_girl,height,weight,feetsize,label,0)
    #正态分布+极大似然估计
    boy_height_mean,boy_height_variance=onefeature(height,label,1)
    boy_weight_mean,boy_weight_variance=onefeature(weight,label,1)
    boy_feetsize_mean,boy_feetsize_variance=onefeature(feetsize,label,1)
    girl_height_mean,girl_height_variance=onefeature(height,label,0)
    girl_weight_mean,girl_weight_variance=onefeature(weight,label,0)
    girl_feetsize_mean,girl_feetsize_variance=onefeature(feetsize,label,0)

    path_boy_test = "F:\\study in school\\machine learning\\forstudent\\实验数据\\boy.txt"
    path_girl_test ="F:\\study in school\\machine learning\\forstudent\\实验数据\\girl.txt"
    height_test = []
    weight_test = []
    feetsize_test = []
    label_test = []  # 1表示男，0表示女
    label_result= []
    readdata(path_boy_test,height_test,weight_test,feetsize_test,label_test,1)
    readdata(path_girl_test, height_test, weight_test, feetsize_test, label_test, 0)
    label_result=get_result(boy_height_mean,boy_height_variance,girl_height_mean,girl_height_variance,
                            height_test,label_test,theta_boy)
    e1=get_error_percent(label_test,label_result)
    print("以身高为特征的错误率：%f" % e1)
    label_result = get_result(boy_feetsize_mean, boy_feetsize_variance, girl_feetsize_mean, girl_feetsize_variance,
                              feetsize_test, label_test, theta_boy)
    e11 = get_error_percent(label_test, label_result)
    print("以脚的大小为特征的错误率：%f" % e11)
    #双特征
    boy_mean=[boy_height_mean,boy_feetsize_mean]
    boy_variance=[boy_height_variance,boy_feetsize_variance]
    girl_mean=[girl_height_mean,girl_feetsize_mean]
    girl_variance=[girl_height_mean,girl_feetsize_variance]
    test=[height_test,feetsize_test]
    label_result = get_result_two(boy_mean, boy_variance, girl_mean, girl_variance,test,
                              label_test, theta_boy)
    e2 = get_error_percent(label_test, label_result)
    print("以身高和脚的大小为特征的最小损失决策错误率为：%f" % e2)

    risk=array([[0,6],[1,0]])
    label_result = get_result_two_risk(boy_mean, boy_variance, girl_mean, girl_variance, test,
                                  label_test, theta_boy,risk)
    e3 = get_error_percent(label_test, label_result)
    print("以身高和脚的大小为特征的最小风险决策错误率为：%f" % e3)

    # #parzen
    # figure(1)
    # parzen(label, height, 1,1)
    # figure(2)
    # parzen(label, height, 4,2)

    #roc
    figure(3)
    FPR,TPR=get_roc(boy_height_mean, boy_height_variance, girl_height_mean, girl_height_variance,
                                  height_test, label_test, 0.5)
    plot(FPR,TPR,label='0.5_height')

    FPR, TPR = get_roc(boy_height_mean, boy_height_variance, girl_height_mean, girl_height_variance,
                       height_test, label_test, 0.75)
    plot(FPR,TPR,label='0.75_height')

    FPR, TPR = get_roc(boy_height_mean, boy_height_variance, girl_height_mean, girl_height_variance,
                       height_test, label_test, 0.9)
    plot(FPR, TPR, label='0.9_height')

    FPR, TPR = get_roc(boy_weight_mean, boy_weight_variance, girl_weight_mean, girl_weight_variance,
                       weight_test, label_test, 0.5)
    plot(FPR, TPR, label='0.5_weight')

    FPR, TPR = get_roc(boy_feetsize_mean, boy_feetsize_variance, girl_feetsize_mean, girl_feetsize_variance,
                       feetsize_test, label_test, 0.5)
    plot(FPR, TPR, label='0.5_feetsize')

    FPR, TPR = get_roc(boy_mean, boy_variance, girl_mean, girl_variance,
                                  test, label_test, 0.5)
    plot(FPR, TPR,label='0.5_two')
    FPR, TPR = get_roc_risk(boy_mean, boy_variance, girl_mean, girl_variance,
                                  test, label_test, 0.5,risk)
    plot(FPR, TPR,label='0.5_two_risk')

    plot([0,1],[1,0])
    legend(loc='lower right')
    figure(3).show()

    #决策面
    figure(4)
    x=np.arange(130,190,0.01)
    y=[]
    for k in range(len(x)):
        if (probability_density(boy_height_mean,boy_height_variance,x,k)*float(theta_boy)) >=\
                (probability_density(girl_height_mean,girl_height_variance,x,k)*float(theta_girl)):
            y.append(probability_density(boy_height_mean,boy_height_variance,x,k)*float(theta_boy))
            if(abs(probability_density(boy_height_mean,boy_height_variance,x,k)*float(theta_boy)) - \
                    (probability_density(girl_height_mean,girl_height_variance,x,k)*float(theta_girl))<0.0001):
                    decision=x[k]
        else:
            y.append(probability_density(girl_height_mean,girl_height_variance,x,k)*float(theta_girl))
    plot(x,y)
    vlines(decision,min(y),max(y))
    title('身高')
    figure(4).show()

    fig=figure(5)
    ax = Axes3D(fig)
    x=np.arange(140,190,1)
    y=np.arange(35,45,0.2)
    x, y = np.meshgrid(x, y)
    p1 = 1 / sqrt(2 * pi * boy_height_variance) * np.exp(-((x - boy_height_mean) ** 2) / (2 * boy_height_variance))
    p2 = 1 / sqrt(2 * pi * boy_feetsize_variance) * np.exp(-((y - boy_feetsize_mean) ** 2) / (2 * boy_feetsize_variance))
    z1=p1*p2*float(theta_boy)
    p1 = 1 / sqrt(2 * pi * girl_height_variance) * np.exp(-((x - girl_height_mean) ** 2) / (2 * girl_height_variance))
    p2 = 1 / sqrt(2 * pi * girl_feetsize_variance) * np.exp(
        -((y - girl_feetsize_mean) ** 2) / (2 * girl_feetsize_variance))
    z2 = p1 * p2 * float(theta_girl)
    z=[]
    for i in range(len(z1)):
        z0=[]
        for j in range(len(z1[0])):
            z0.append(max(z1[i][j],z2[i][j]))
        z.append(z0)
    ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap='rainbow')
    figure(5).show()

    figure(6)
    plt.contour(x, y, z)
    f=(x-boy_height_mean)**2/boy_height_variance+(y-boy_feetsize_mean)**2/boy_feetsize_variance- \
      (x - girl_height_mean) ** 2 / girl_height_variance-(y-girl_feetsize_mean)**2/girl_feetsize_variance- \
      2 * log(sqrt(girl_height_variance * girl_feetsize_variance / (boy_feetsize_variance * boy_height_variance)))
    plt.contour(x, y, f,0)
    title('身高，脚的大小的决策面')
    show()

main()
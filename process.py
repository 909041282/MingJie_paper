# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import mpl
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
names = ['一等奖数', '二等奖数', '三等奖数', '总获奖']
years = ['2013', '2014', '2015', '2016', '2017', '2018', '2019']

#利用pandas读取文件
def read_data(filename):
    data = pd.read_csv(filename, header=0)
    my_group = data.groupby(data['年份'])[names]
    sum_res = my_group.sum()
    for name in names:
        data[name + '比例'] = [data[name][i] / sum_res[name][data['年份'][i]] for i in range(140)]
    return data

# 绘折线图
def plot_line(data, type):
    for i in range(20):
        plt.subplot(4, 5, 1 + i)
        plt.plot(data['年份'][i * 7:i * 7 + 7], data[type][i * 7:i * 7 + 7])
        plt.title(data['学校'][i * 7])
        plt.ylim(-5, 40)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.subplots_adjust(top=0.962, bottom=0.033, left=0.018, right=0.992, hspace=0.293, wspace=0.190)
    plt.savefig('pic/' + type + '.png')
    plt.show()

# 绘制 饼状图
def plot_pine(data, type):
    for i in range(len(years)):
        x = data[type][data['年份'] == years[i]]
        labels = data['学校'][data['年份'] == years[i]]
        labels = labels[x != 0]
        x = x[x != 0]
        explode = [0] * len(x)
        plt.pie(x, labels=labels, explode=explode)
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.2, wspace=0.2)
        plt.savefig('pic/' + str(years[i]) + type + '.png')
        plt.ion()
        plt.pause(1)
        plt.close()

# 按照学校对数据进行归类
def get_dataset(data, type):
    train_data = [np.array(data[type][i * 7:i * 7 + 7]) for i in range(20)]
    return np.array(train_data)

#霍尔特(Holt)线性趋势法
def get_holt(x,y):
    pred = []
    for data in x:
        r2 = Holt(data).fit()
        pred.append(r2.predict(start=len(data), end=len(data))[0])
    save_res('./res/res_rate_holt.txt', np.array(pred), y)

#简单平均法
def simple_mean(x,y):
    pred = np.array([data.mean() for data in x])
    save_res('./res/res_rate_mean.txt', pred, y)

#简单指数平滑法
def simpleExpSmoothing(x,y):
    pred = []
    for data in x:
        fit = SimpleExpSmoothing(data).fit(smoothing_level=0.6, optimized=False)
    pred.append(fit.forecast(1))
    save_res('./res/res_rate_simpleExpSmoothing.txt', pred, y)

#Holt-Winters季节性预测模型
def exponentialSmoothing(x,y):
    pred = []
    for data in x:
        fit = ExponentialSmoothing((data), seasonal_periods=3, trend='add', seasonal='add', ).fit()
    pred.append(fit.forecast(1))
    save_res('./res/res_rate_exponentialSmoothing.txt', pred, y)

#逻辑回归
def linearRegression(x,y):
    lr =LinearRegression().fit(x,y)
    pred=lr.predict(x)
    save_res('./res/res_rate_linearRegression.txt', pred, y)

# 计算均方差并保存文件
def save_res(name, pred, y):
    fout = open(name, 'a')
    acc = np.sqrt(np.square(pred - y).sum())
    fout.write('acc:{}\n\n'.format(acc))
    fout.close()


data = read_data('1.csv')
for name in names:
    train_data = get_dataset(data, name + '比例')
    x = train_data[:, :-1]
    y = train_data[:, -1]
    linearRegression(x,y)

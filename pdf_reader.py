# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

group_names = ['本科_1', '本科_2', '专科_1', '专科_2']
def read_all():
    data={}
    for group in group_names:
        data[group]=[]
    years = sorted(os.listdir('./data'))
    for year in years:
        for group in group_names:
            filename = './data/' + year + '/' + year +'_'+ group+'.txt'
            with open(filename, 'r', encoding='utf-8') as f:
                data[group].append([line.split() for line in f.readlines()])
    return data

data=read_all()
provice={}
for group in sorted(data):
    provice[group]={}
    for year_data in data[group]:
        for line in year_data:
            if line[1] not in provice[group]:
                provice[group][line[1]]=[]
    for year_data in data[group]:
        year_province = {}
        for line in year_data:
            if line[1] not in year_province:
                year_province[line[1]]=1
            else:
                year_province[line[1]]+=1
        for pro in provice[group]:
            if pro in year_province:
                provice[group][pro].append(year_province[pro])
            else:
                provice[group][pro].append(0)


for group in group_names:
    x=range(2009,2019)
    for pro in provice[group]:
        plt.plot(x,provice[group][pro],label=pro)
        plt.xlabel('年份')
        plt.ylabel('数目')

        plt.title(group)
        plt.legend(loc='upper right')
    plt.show()

    # for i in range(10):
    #     labels=provice[group]
    #     show_label=[]
    #     y=[]
    #     for label in labels:
    #         if provice[group][label][i]!=0:
    #             show_label.append(label)
    #             y.append(provice[group][label][i])
    #     plt.pie(y,labels=show_label,autopct='%1.1f%%',shadow=False,startangle=150)
    #     plt.title('第'+str(x[i])+'年各省人数饼状图')
    #     plt.show()



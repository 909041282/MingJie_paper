# -*- coding: utf-8 -*-

import os
from school import school_to_province
group_names = ['_本科_1.txt', '_本科_2.txt', '_专科_1.txt', '_专科_2.txt']

def read_all():
    years = sorted(os.listdir('./data'), reverse=True)
    datas = {}
    for year in years:
        data = []
        for group in group_names:
            filename = './data/' + year + '/' + year + group
            with open(filename, 'r', encoding='utf-8') as f:
                data.append([line.split() for line in f.readlines()])

        datas[year] = data
    return datas


datas = read_all()
for year in datas.keys() - ['2009', '2010']:
    value = datas[year]
    for i in range(4):
        for line in value[i]:
            if line[2] not in school_to_province:
                school_to_province[line[2]] = line[1]
i=2
filename = './data/2010/2010—'+group_names[i]
value = datas['2010'][i]
for line in value:
    line.insert(1, school_to_province[line[1]])

with open(filename, 'w',encoding='utf-8') as f:
    for line in value:
        for item in line:
            f.write(item + ' ')
        f.write('\n')

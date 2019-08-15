import sys
import math
import os
from dictionary import Journal

dictionary = {}


summer15 = Journal()
F = open('input/summer15.txt', 'r')
title = ''
for line in F:
    line = line.strip('\n')
    is_date = False
    
    if len(line) < 20:
        title = line
        summer15.dates[title] = {}
        is_date = True
    else:
        words = line.split(' ')
        for word in words:
            word = word.lower()
            if word in summer15.dates[title]:
                summer15.dates[title][word] += 1
            else:
                summer15.dates[title][word] = 1


for key in summer15.dates:
    for word in summer15.dates[key]:
        print(summer15.dates[key][word])

#this seems to correctly dictionaryize everything 
     

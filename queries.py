import sys
import matplotlib.pyplot as plt
import numpy as np 
import heapq
from collections import Counter

def names(Journal):
    name_frequences = {}
    for day in Journal.dates:
        for word in Journal.dates[day]:
            if '\'s' in word:
                word = word.replace('\'s', '')
            if word in Journal.common_names:

                if word not in name_frequences:
                    name_frequences[word] = 1
                else:
                    name_frequences[word] += 1
    # items = heapq.nlargest(5, name_frequences, key=name_frequences.get)

    c = dict(Counter(name_frequences).most_common(15))
    print(c)
    



    
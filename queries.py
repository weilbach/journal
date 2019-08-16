import sys
import matplotlib.pyplot as plt
import numpy as np 
import heapq
from collections import Counter

#functions that take the journal as an argument

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

    c = Counter(name_frequences).most_common(15)
    c = dict(c)

    print(c)

    #this all needs to be way better but it's a start
    plt.bar(c.keys(), c.values(), align='edge', width=.3)
    plt.xticks(fontsize= 8, rotation = 90)
    plt.show()

def foods(Journal):

    food_words = {'eaten', 'ate', 'breakfast', 'lunch', 'dinner', 'brunch', 'munched', 'grubbed', 'food', 'consumed'}
    eat_frequencies = {}
    for day in Journal.dates:
        eat_count = 0
        for word in Journal.dates[day]:
            if word in food_words:
                eat_count += 1
        
        eat_frequencies[day] = eat_count
    

    c = dict(Counter(eat_frequencies).most_common(10))
    print(c)








    
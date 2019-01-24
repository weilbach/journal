#!/usr/bin/env python3

#reduce2
import sys

mydict = {}
for line in sys.stdin:

    line = line.strip('\n')
    words = line.split('\t')
    key = words[0]

    if key not in mydict:
        mydict[key] = [words[1]]
    else:
        mydict[key].append(words[1])

for item in mydict:
    for thing in mydict[item]:
        print('clean.csv' + '\t' + thing)
#!/usr/bin/env python3

#map2
import sys


for line in sys.stdin:
    line = line.strip('\n')
    words = line.split('\t')
    values = words[1].split(',')
    key = values[3]
    print(key + '\t' + words[1])

    
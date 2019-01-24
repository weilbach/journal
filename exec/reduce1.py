#!/usr/bin/env python3
"""Reduce 1."""

import sys
import collections
import re

# testdoc2 = open('reduce1.txt', 'w')


WORDDICT = {}
for line in sys.stdin:
    line = line.strip('\n')
    word = line.split('\t')
    full_key = word[0] + ',' + word[1]
    if full_key in WORDDICT:
        WORDDICT[full_key] += 1
    else:
        WORDDICT[full_key] = 1

SORTEDDICT = collections.OrderedDict(sorted(WORDDICT.items()))
for key in SORTEDDICT:
    # testdoc2.write(key + '\t' + str(SORTEDDICT[key]))
    # testdoc2.write('\n')
    print(key + '\t' + str(SORTEDDICT[key]))

# testdoc2.close()

    

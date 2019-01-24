#!/usr/bin/env python3
"""Map 0."""

import sys
#counts the number of documents in the input file

count = 0

for line in sys.stdin:
    print(str(count) + "\t1")
    count += 1
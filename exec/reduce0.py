#!/usr/bin/env python3
"""Reduce 0."""

import sys
import collections

#creates and writes to document that is used in flask or something
doc_count = open('total_document_count.txt', 'w')

total = 0
counter = 0

for line in sys.stdin:
    total += 1
    if total % 2 == 0:
        counter += 1
    

# total /= 3
# doc_count.write(str(total))
doc_count.write(str(counter))

doc_count.close()
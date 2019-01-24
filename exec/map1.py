#!/usr/bin/env python3
"""Map 1."""

import sys
import re
import math

#this is the executable that will calculate all the occurrences of each word
#for each date
#update it works

#leaving this here for reference
#word -> idf -> doc_id -> number of occurrences in doc_id 
#-> doc_id's normalization factor (before sqrt)

current_date = ''
count = 0
# test_doc = open('map1.txt', 'a')

for line in sys.stdin:
    line = line.strip('\n')
    if count % 2 == 0:
            current_date = str(line).strip()
    if count % 2 != 0:
        words = line.split(' ')
        for word in words:
            word = word.lower()
            word = re.sub(r'[^a-zA-Z0-9]+', '', word)
            # if word not in word_array and word != '':
            print(current_date + '\t' + word)
                # test_doc.write(current_doc + '\t' + word)
                # test_doc.write('\n')
    count += 1

# test_doc.close()
    
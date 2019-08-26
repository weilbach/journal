import sys
import math
import os
from dictionary import Journal
from queries import names, foods
from explicit import weed


#this is serving as the main file 
dictionary = {}


# summer15 = Journal()
summer18 = Journal()

# summer15.assemble_dictionary('summer15.txt')
# summer18.assemble_dictionary('summer2018.txt')
summer18.assemble_fragments('summer2018.txt')

# summer15.assemble_names()
# summer18.assemble_names()

# names(summer15)
# foods(summer18)
# weed(summer18)

#this seems to correctly dictionaryize everything 
     

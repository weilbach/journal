import sys
import math
import os
from dictionary import Journal
from queries import names


#this is serving as the main file 
dictionary = {}


summer15 = Journal()

summer15.assemble_dictionary('summer15.txt')

summer15.assemble_names()

names(summer15)

#this seems to correctly dictionaryize everything 
     

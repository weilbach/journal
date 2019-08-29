import sys
import math
import os
from dictionary import Journal
from queries import names, foods
from explicit import weed
from checkpoints import load_checkpoint, save_checkpoint
from dayrank import create_csv


#this is serving as the main file 

#this file now has the ability to load and save checkpoints 
dictionary = {}


# summer15 = Journal()
summer18 = Journal()

# summer15.assemble_dictionary('summer15.txt')
summer18.assemble_dictionary('summer2018.txt')
summer18.assemble_fragments('summer2018.txt')

# summer15.assemble_names()
# summer18.assemble_names()

create_csv(summer18.fragments, summer18.name)

# names(summer15)
# foods(summer18)
# weed(summer18)
# save_checkpoint(summer18.fragments, summer18.dates, summer18.common_names, summer18.name)

#this seems to correctly dictionaryize everything 
     

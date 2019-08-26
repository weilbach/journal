import sys
import os
import re


class Journal():

    # dates = {}
    def __init__(self):
        self.dates = {} #a more comprehensive dictionary of all words for each day
        self.common_names = set() #just holds all names seen in a journal
        self.fragments = {} #will hold sentence fragments
    
    def assemble_dictionary(self, filename):
        filepath = 'input/' + filename
        F = open(filepath, 'r')
        title = ''
        for line in F:
            line = line.strip('\n')
            is_date = False
            
            if len(line) < 20:
                title = line
                self.dates[title] = {}
                is_date = True
            else:
                words = line.split(' ')
                for word in words:
                    word = word.lower()
                    if word in self.dates[title]:
                        self.dates[title][word] += 1
                    else:
                        self.dates[title][word] = 1
        
        #in case you wanna check prints
        # for key in self.dates:
        #     for word in self.dates[key]:
        #         print(self.dates[key][word])
    
    def assemble_names(self):
        #ok this seems to work
        F = open('first-names.txt', 'r')
        for line in F:
            line = line.strip('\n')
            line = line.lower()
            self.common_names.add(line)
    
    def assemble_fragments(self, filename):
        filepath = 'input/' + filename
        F = open(filepath, 'r')
        title = ''
        for line in F:
            line = line.strip('\n')
            if len(line) < 20:
                title = line
                self.fragments[title] = []
            else:
                #I'd like to thank Jacob Kilby for this line of code
                words = re.split(r'\.|\bthen\b', line)
                self.fragments[title] = words
        
        #these are just checks
        # print(self.fragments['6/23'])
        # print(self.fragments['6/30'])


        

    
    



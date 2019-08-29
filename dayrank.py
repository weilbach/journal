#gonna be doing some ML stuff in this file yudig 

def create_csv(fragments, name):

    f = open(name, 'w+')
    for lists in fragments.values():
        for frag in lists:
            if ',' in frag:
                frag.replace(',', '')
            frag.strip()
            if len(frag) < 5: #this is sort of an imperfect fix 
                continue #never mind it's just not fixed yet lol
            f.write(frag + ',' +'\n')
    
    f.close()




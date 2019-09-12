import pickle

def load_checkpoint(name):

    checkpoints = []
    filename = '%s_journal_%d.pkl' % ('justin', name)
    full_path = 'checkpoints' + filename
    with open(full_path, 'rb') as cp_file:
        cp = pickle.load(cp_file)
        checkpoints.append(cp)
    
    return checkpoints

def save_checkpoint(fragments, dictionary, common_names, name):
    checkpoint = {
        'fragments': fragments,
        'dictionary': dictionary,
        'common_names': common_names
    }
    filename = '%s_journal_%s.pkl' % ('justin', name)
    full_path = 'checkpoints/' + filename
    with open(full_path, 'wb') as f:
        pickle.dump(checkpoint, f)
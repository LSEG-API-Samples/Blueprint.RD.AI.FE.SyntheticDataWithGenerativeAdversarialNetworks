import pickle


def load_pickled_object(pickle_file):
    with open(pickle_file, 'rb') as f:
        p = pickle.load(f)
    return p

import pandas as pd
import numpy as np
import string


def load_data(fname):
    """
    Reads in a csv file and return a dataframe. A dataframe df is similar to dictionary.
    You can access the label by calling df['label'], the content by df['content']
    the rating by df['rating']
    """
    return pd.read_csv(fname)


# def get_split_binary_data():
#     """
#     Reads in the data from data/dataset.csv and returns it using
#     extract_dictionary and generate_feature_matrix split into training and test sets.
#     The binary labels take two values:
#         -1: poor/average
#          1: good
#     Also returns the dictionary used to create the feature matrices.
#     """
#     fname = "summer2018.csv"
#     dataframe = load_data(fname)
#     dataframe = dataframe[dataframe['label'] != 0]
#     positiveDF = dataframe[dataframe['label'] == 1].copy()
#     negativeDF = dataframe[dataframe['label'] == -1].copy()
#     X_train = pd.concat([positiveDF[:150], negativeDF[:30]]).reset_index(drop=True).copy()
#     dictionary = project1.extract_dictionary(X_train)
#     X_test = pd.concat([positiveDF[150:], negativeDF[30:]]).reset_index(drop=True).copy()
#     Y_train = X_train['label'].values.copy()
#     Y_test = X_test['label'].values.copy()
#     X_train = project1.generate_feature_matrix(X_train, dictionary)
#     X_test = project1.generate_feature_matrix(X_test, dictionary)

#     return (X_train, Y_train, X_test, Y_test, dictionary)

def extract_dictionary(df):
    """
    Reads a panda dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    # count = 0
    text_list = df['text']
    word_dict = {}
    count = 0

    for line in text_list:

        # line = remove_punctuation(line) #don't need this anymore 
        line = line.lower()
        words = line.split(' ')
        for word in words:
            if word != '':
                if word not in word_dict:
                    word_dict[word] = count
                    count += 1
    
    # print(word_dict)

    return word_dict


def generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (number of reviews, number of words).
    Input:
        df: dataframe that has the ratings and labels
        word_list: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (number of reviews, number of words)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    word_list = df['text']
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    count = 0

    for line in word_list:
        line = remove_punctuation(line)
        line = line.lower()
        words = line.split(' ')
        for word in words:
            if word in word_dict:
                feature_matrix[count][word_dict[word]] = 1
        count += 1


    return feature_matrix

def remove_punctuation(value):
    #this is a remnant from an old project, it works but is likely slow
    result = ''
    for c in value:
        # If char is not punctuation, add it to the result.
        if c not in string.punctuation:
            result += c
        else:
            c = ' '
            result += c
    return result



def get_imbalanced_data(dictionary):
    """
    Reads in the data from data/imbalanced.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are binary as follows
        -1: poor/average
        1: good
    Input:
        dictionary: the dictionary created via get_split_binary_data
    """
    fname = "data/imbalanced.csv"
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe['label'] != 0]
    positiveDF = dataframe[dataframe['label'] == 1].copy()
    negativeDF = dataframe[dataframe['label'] == -1].copy()
    dataframe = pd.concat([positiveDF[:300], negativeDF[:700]]).reset_index(drop=True).copy()
    X_train = project1.generate_feature_matrix(dataframe, dictionary)
    Y_train = dataframe['label'].values.copy()

    return (X_train, Y_train)


def get_imbalanced_test(dictionary):
    """
    Reads in the data from data/dataset.csv and returns a subset of it
    reflecting an imbalanced test dataset
        -1: poor/average
        1: good
    Input:
        dictionary: the dictionary created via get_split_binary_data
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe['label'] != 0]
    positiveDF = dataframe[dataframe['label'] == 1].copy()
    negativeDF = dataframe[dataframe['label'] == -1].copy()
    X_test = pd.concat([positiveDF[:400], negativeDF[:100]]).reset_index(drop=True).copy()
    Y_test = X_test['label'].values.copy()
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_test, Y_test)


def get_multiclass_training_data():
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are multiclass as follows
        -1: poor
         0: average
         1: good
    Also returns the dictionary used to create X_train.
    """
    fname = "summer2018labels.csv"
    dataframe = load_data(fname)
    dictionary = extract_dictionary(dataframe)
    X_train = generate_feature_matrix(dataframe, dictionary)
    Y_train = dataframe['label'].values.copy()

    return (X_train, Y_train, dictionary)



def get_heldout_reviews(dictionary):
    """
    Reads in the data from data/heldout.csv and returns it as a feature
    matrix based on the functions extract_dictionary and generate_feature_matrix
    Input:
        dictionary: the dictionary created by get_multiclass_training_data
    """
    fname = "summer15.csv"
    dataframe = load_data(fname)
    X = generate_feature_matrix(dataframe, dictionary)
    return X


def generate_challenge_labels(y, uniqname):
    """
    Takes in a numpy array that stores the prediction of your multiclass
    classifier and output the prediction to held_out_result.csv. Please make sure that
    you do not change the order of the ratings in the heldout dataset since we will
    this file to evaluate your classifier.
    """
    pd.Series(np.array(y)).to_csv(uniqname+'.csv', header=['label'], index=False)
    return

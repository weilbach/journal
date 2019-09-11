#gonna be doing some ML stuff in this file yudig 
import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt
from string import *
from mlhelper import get_multiclass_training_data

def create_csv(fragments, name):

    #okay i think this fully works now

    f = open(name, 'w+')
    for key, lists in fragments.items():
        for frag in lists:
            frag = remove_punctuation(frag)
            frag = frag.strip()
            if len(frag) < 5: #this is sort of an imperfect fix 
                continue
            f.write(key + ',' + frag + ',' + '1' + '\n')
    
    f.close()


def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """
    if penalty == 'l2':
        return SVC(C=c, kernel='linear', degree=degree, class_weight=class_weight)
    elif penalty == 'l1':
        return LinearSVC(C=c, penalty=penalty, class_weight=class_weight, dual=False)
    
    # Optionally implement this helper function if you would like to
    # instantiate your SVM classifiers in a single function. You will need
    # to use the above parameters throughout the assignment.


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



def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    #HINT: You may find the StratifiedKFold from sklearn.model_selection
    #to be useful

    skf = StratifiedKFold(n_splits=k)
    scores = []

    for train, test in skf.split(X,y):
        y_pred = []
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        clf.fit(X_train, y_train)
        
        if metric == 'auroc':
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        score = performance(y_test, y_pred, metric)
        scores.append(score)

    return np.array(scores).mean()


def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """

    print('Linear SVM Hyperparameter Selection based on ' + metric)
    scores = []
    
    #use all given hyperparameters (that's the right term, right?)
    for C in C_range:
        # calculate average performance
        clf = select_classifier(penalty=penalty, c=C)
        score = cv_performance(clf, X, y, k, metric)
        scores.append(score)

    
    best = np.max(scores)

    #gonna need some print statements in here to get values
    to_return = C_range[scores.index(best)]
    print('the best score for ' + metric + ' is ' + str(best))
    return to_return
    


def plot_weight(X,y,penalty,metric,C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []

    # TODO: Implement this part of the function
    #Here, for each value of c in C_range, you should
    #append to norm0 the L0-norm of the theta vector that is learned
    #when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)

    for C in C_range:
        clf = select_classifier(c=C, penalty=penalty)
        clf.fit(X, y)
        w = clf.coef_
        w = np.asarray(w)
        w = np.squeeze(w)
        to_append = np.linalg.norm((w), ord=0)
        norm0.append(to_append)


    #This code will plot your L0-norm as a function of c
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-'+penalty+'_penalty.png')
    plt.savefig('Norm-'+penalty+'_penalty.png')
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
            X: (n,d) array of feature vectors, where n is the number of examples
               and d is the number of features
            y: (n,) array of binary labels {1,-1}
            k: an int specifying the number of folds (default=5)
            metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
            parameter_values: a (num_param, 2)-sized array containing the
                parameter values to search over. The first column should
                represent the values for C, and the second column should
                represent the values for r. Each row of this array thus
                represents a pair of parameters to be tried together.
        Returns:
            The parameter value(s) for a quadratic-kernel SVM that maximize
            the average 5-fold CV performance
    """
    
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    scores = []
    c_r = []
    # print(param_range.shape)
    rows, cols = param_range.shape

    if rows == 7:
        for i in range(0, rows):
            for j in range(0, rows):
                c = param_range[i][0]
                r = param_range[j][1]
                clf = SVC(kernel='poly', degree=2, C=c, coef0=r, class_weight='balanced', gamma='auto')
                score = cv_performance(clf, X, y, k=k, metric='auroc')
                scores.append(score)
                c_r.append((c, r))
                print('grid ' + str(score) + ' ' + str(c) + ',' + str(r))
    else:
        for i in range(0, rows):
            c = param_range[i][0]
            r = param_range[i][1]
            clf = SVC(kernel='poly', degree=2, C=c, coef0=r, class_weight='balanced', gamma='auto')
            score = cv_performance(clf, X, y, k=k, metric='auroc')
            scores.append(score)
            c_r.append((c, r))
            print('random ' + str(score) + ' ' + str(c) + ',' + str(r))

    best = np.max(scores)
    if rows == 7:
        print('the best score for grid search for AUROC is ' + str(best))
    else:
        print('the best score for random search for AUROC is ' + str(best))

    # #gonna need some print statements in here to get values
    return_index = scores.index(best)
    c_return = c_r[return_index][0]
    r_return = c_r[return_index][1]
    
    return r_return, c_return


def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
   
    if metric == 'accuracy':
        score = metrics.accuracy_score(y_true, y_pred)
        return score
    elif metric == 'f1_score':
        score = metrics.f1_score(y_true, y_pred)
        return score
    elif metric == 'auroc':
        score = metrics.roc_auc_score(y_true, y_pred)
        return score
    elif metric == 'precision':
        score = metrics.precision_score(y_true, y_pred)
        return score
    elif metric == 'sensitivity':
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[-1,1]).ravel()
        score = tp / (tp + fn)
        return score
    elif metric == 'specificity':
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[-1,1]).ravel()
        score = tn / (tn + fp)
        return score
   
# def question2():

#      #start question 2
#     X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()

#     nonzero_count = 0
#     total_count = 0
#     for i in X_train:
#         for j in i:
#             if j == 1:
#                 nonzero_count += 1
#         total_count += 1
    
#     print('amount of nonzeros is ' + str(nonzero_count / total_count))

#     total_size = len(X_train[0])
#     print('size of feature matrix is ' + str(total_size))

def question3():
    
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()

    #start question 3 

    c_list = [.001, .01, .1, 1, 10, 100, 1000]
    metric_list = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']
    best_cs = []
    for metric in metric_list:
        best_C = select_param_linear(X_train, Y_train, metric=metric, C_range=c_list)
        print('the best C for ' + str(metric) + ' is ' +  str(best_C))
        best_cs.append(best_C)
    
    print('\n')
    print('question 3.1d')
    for metric in metric_list:
        clf = select_classifier(penalty='l2', c=.1)
        clf.fit(X_train, Y_train)
        if metric == 'auroc':
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        test_perf = performance(Y_test, y_pred, metric=metric)

        print('The test performance for ' + str(metric) + ' is ' + str(test_perf))

def question3e():

    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    c_list = [.001, .01, .1, 1, 10, 100, 1000]
    
    print('3.1.e')

    plot_weight(X_train, Y_train, 'l2', 'specificity', C_range=c_list)

def question3f():

    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    c = .1
    clf = select_classifier(c=c)
    clf.fit(X_train, Y_train)
    
    theta = clf.coef_
    theta = np.asarray(theta)
    theta = np.squeeze(theta)
    
    print(theta.shape)
    small_thetas = []
    big_thetas = []

    #sort in order for small values
    new_theta = theta
    new_theta = np.sort(new_theta)

    #find the 4 smallest coefficients
    for i in range(0, 4):
        small_thetas.append(new_theta[i])
        # print(new_theta[i])
    
    #sort in reverse order for large values
    new_theta = np.sort(new_theta)[::-1]
    
    #find the 4 largest coefficients
    for i in range(0, 4):
        big_thetas.append(new_theta[i])
        # print(new_theta[i])
    
    index = 0
    #loop only to check for small ones
    for word in dictionary_binary:
        if theta[index] in small_thetas:
            print('the coefficient ' + str(theta[index]) + ' maps to ' + word)
        index += 1
    
    index = 0
    #loop to check for big ones
    for word in dictionary_binary:
        if theta[index] in big_thetas:
            print('the coefficient ' + str(theta[index]) + ' maps to ' + word)
        index += 1

def question32():

   #this is for the quadratic model instead of the linear model

    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()


    #start question 3.2 everything really

    params = np.array([[.001, .001],
                        [.01, .01],
                        [.1, .1],
                        [1, 1],
                        [10, 10],
                        [100, 100],
                        [1000, 1000]
                        ])
    
    best_cs = []
    best_rs = []

    best_R, best_C = select_param_quadratic(X_train, Y_train, param_range=params)
    print('the best C for grid search is ' +  str(best_C))
    print('the best R for grid search is ' +  str(best_R))
    best_cs.append(best_C)
    best_rs.append(best_R)
    

    random_c = np.random.uniform(3, -3, 25)
    random_r = np.random.uniform(3, -3, 25)
    
    shape = (25,2)
    random_mat = np.zeros(shape)

    for i in range(0, len(random_c)):
        random_mat[i][0] = 10 ** random_c[i]
        random_mat[i][1] = 10 ** random_r[i]
    
    best_R, best_C = select_param_quadratic(X_train, Y_train, param_range=random_mat)
    print('the best C for random search is ' +  str(best_C))
    print('the best R for random search is ' +  str(best_R))


def question34():

    print('this is question 3.4')

    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()

    c_list = [.001, .01, .1, 1, 10, 100, 1000]
    
    best_C = select_param_linear(X_train, Y_train, metric='auroc', C_range=c_list, penalty='l1')

    print('the best C value for l1 optimization is ' + str(best_C))

    plot_weight(X_train, Y_train, penalty='l1', metric='auroc', C_range=c_list)


def question4():

    print('this is question 4')

    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    metric_list = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']
    
    clf = select_classifier(penalty='l2', c=.01, class_weight={-1:10, 1:1})
    clf.fit(X_train, Y_train)

    for metric in metric_list:
        if metric == 'auroc':
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        score = performance(Y_test, y_pred, metric)
        print('the score for ' + str(metric) + ' is ' + str(score))


def question42():

    print('this is question 4.2')

    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)
    metric_list = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']

    clf = select_classifier(penalty='l2', c=.01, class_weight={-1:1, 1:1})
    clf.fit(IMB_features, IMB_labels)

    for metric in metric_list:
        if metric == 'auroc':
            y_pred = clf.decision_function(IMB_test_features)
        else:
            y_pred = clf.predict(IMB_test_features)
        score = performance(IMB_test_labels, y_pred, metric)
        print('the score for ' + str(metric) + ' is ' + str(score))


def question43():

    print('this is question 4.3')

    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)

    metric_list = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']
    
    #this is the code used to find the best weights using cross validation
    best = 0
    index_i = -1
    index_j = -1
    for i in range(1, 5):
        for j in range(1, 10):
            if j != 1 or (j == 1 and i == 1):
                clf = select_classifier(penalty='l2', c=.01, class_weight={-1:i, 1:j})
                score = cv_performance(clf, IMB_features, IMB_labels, k=5, metric='f1_score')
                print('the cv performance for weights ' + str(i) + ',' + str(j) + ' is ' + str(score))
                if score > best:
                    best = score
                    index_i = i
                    index_j = j
    
    # this gives me 4,7
    # when i ran the loop from 1-6 and 1-11 i actually got 5,9 but 
    # 4,7 has the exact same performance so I'm going with that 
    # 2,8 actually has better performance on most metrics but it's CV was lower
    

    #this is the code for assessing performance on all metrics using the new weights
    for metric in metric_list:
        clf = select_classifier(penalty='l2', c=.01, class_weight={-1:4, 1:7})
        clf.fit(IMB_features, IMB_labels)
        if metric == 'auroc':
            y_pred = clf.decision_function(IMB_test_features)
        else:
            y_pred = clf.predict(IMB_test_features)
        score = performance(IMB_test_labels, y_pred, metric)
        print('the score for ' + str(metric) + ' using weights ' + str(4) + ',' + str(7) + ' is ' + str(score))


def question44():

    #something about an ROC curve idk man 
    print('start question 4.4')

    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)


    #old class weights that are bad
    clf = select_classifier(c=.01, class_weight={-1: 1, 1: 1})
    clf.fit(IMB_features, IMB_labels)
    y_pred = clf.decision_function(IMB_test_features)
    fpr, tpr, temp = metrics.roc_curve(IMB_test_labels, y_pred)


    #new class weights after optimization
    clf = select_classifier(c=.01, class_weight={-1:4, 1: 7})
    clf.fit(IMB_features, IMB_labels)
    y_pred = clf.decision_function(IMB_test_features)
    fpr2, tpr2, temp = metrics.roc_curve(IMB_test_labels, y_pred)


    # plot everything
    plt.figure()
    plt.plot(fpr, tpr, color='yellow', linewidth=2.0, 
    label='$W_n = 1$, $W_p = 1$, score = 0.910975')
    plt.plot(fpr2, tpr2, color='blue', linewidth=2.0, 
    label='$W_n = 4$, $W_p = 7$, score = 0.932425')
    plt.plot([0, 1], [0, 1], color='red', linewidth=2.0, linestyle='dashed')
    plt.xlim([0.0, 1.2])
    plt.ylim([0.0, 1.2])
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.title('4.4(a) - ROC curve with different class weights')
    plt.legend(loc="lower right")
    plt.savefig('q44a.png')


def question5():

    print('start question 5')
    #here I will be attempting to do one vs all classification 

    uniqname = 'weilbach'

    multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data()
    
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    c_list = [.001, .01, .1, 1, 10, 100, 1000]
    
    best_c = None  
    score = 0  

    for c in c_list:
        clf = LinearSVC(penalty='l2', C=c, multi_class='ovr', dual=True)
        new_score = cv_performance(clf, multiclass_features, multiclass_labels, k=5, metric='accuracy')
        print('the score for c =' + str(c) + ' is ' + str(score))
        if new_score > score:
            best_c = c
            score = new_score
        

    
    clf = LinearSVC(penalty='l2', C=best_c, multi_class='ovr', dual=True)

    clf.fit(multiclass_features, multiclass_labels)

    y_pred = clf.predict(heldout_features)
    print(y_pred)
    # #given the words in the tweet, what's good with the amount of retweets 

    # generate_challenge_labels(y_pred, uniqname)



def main():

    df = pd.read_csv('summer2018labels.csv', usecols=['date','text','label'])
    df = pd.DataFrame(df)
    my_dict = extract_dictionary(df)
    generate_feature_matrix(df, my_dict)
    
    #question 2
    # question2()

    #question 3...this one is huge
    # question3()

    # question3e()

    # question3f()

    # #start question 3.2
    # question32()

    # question34()

    # question4()

    # question42()

    # question43()

    # question44()

    question5()


if __name__ == '__main__':
    main()




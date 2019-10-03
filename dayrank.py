#gonna be doing some ML stuff in this file yudig 
import pandas as pd
import numpy as np
import itertools
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt
from mlhelper import *

def create_csv(fragments, name, labels=None):
    """
    creates a csv file out of fragments of a part of a day
    the file is created with the name that is passed in
    as of right now it creates .txt files instead of .csv
    """

    #okay i think this fully works now
    f = open(name, 'w+')
    f.write('date,text,label' + '\n')
    for key, lists in fragments.items():
        for frag in lists:
            frag = remove_punctuation(frag)
            frag = frag.strip()
            if len(frag) < 5: #this is sort of an imperfect fix 
                continue
            f.write(key + ',' + frag + ',' + '\n')
    
    f.close()

def add_labels(file, labels):
    """
    this function adds labels to files that aren't labeled
    the labels aren't necessarily correct, they are just what
    the function has been passed
    """
    #for reference, this function works

    f = open(file, 'r+')
    lines = []
    for i, line in enumerate(f):
        if i != 0:
            lines.append(line)
    f.close()

    f = open(file, 'w+')
    f.write('date,text,label' + '\n')
    for i in range(0, len(lines)):
        f.write(lines[i].strip('\n') + str(labels[i]) + '\n')
    
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
    return y_pred



def main():

    # df = pd.read_csv('summer2018labels.csv', usecols=['date','text','label'])
    # df = pd.DataFrame(df)
    # my_dict = extract_dictionary(df)
    # generate_feature_matrix(df, my_dict)

    summer15_pred = question5()
    add_labels('summer15.txt', summer15_pred)




if __name__ == '__main__':
    main()




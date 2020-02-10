import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
import numpy as np 
import random
from sklearn.model_selection import KFold
from collections import defaultdict

CLASSIFIERS_NAME = {0 : 'SGDClassifier', 1 : 'GaussianNB', 2 : 'RandomForestClassifier', 3 : 'MLPClassifier', 4 : 'AdaBoostClassifier'}
CLASSIFIERS = {}

np.random.seed(401)

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.sum(np.diag(C)) / np.sum(C)


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.diag(C) / np.sum(C, axis=1)


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.diag(C) / np.sum(C, axis=0)
    

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    print('Running Section 3.1')
    best_accuracy = 0
    iBest = -1
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
    # For each classifier, compute results and write the following output:
        for i in range(5):
            #SGD Classifier Training
            if i == 0:
                classifier = SGDClassifier(max_iter=1000)
            #GaussianNB
            elif i == 1:
                classifier = GaussianNB()
            #RandomForestClassifier
            elif i == 2:
                classifier = RandomForestClassifier(max_depth=5, n_estimators=10)
            #MLPClassifier
            elif i == 3:
                classifier = MLPClassifier(alpha=0.05)
            #AdaBoostClassifier
            elif i == 4:
                classifier = AdaBoostClassifier()
            CLASSIFIERS[i] = classifier
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            accuracy_value = accuracy(conf_matrix)
            recall_values = recall(conf_matrix)
            precision_values = precision(conf_matrix)
            classifier_name = CLASSIFIERS_NAME[i]
            if accuracy_value > best_accuracy:
                best_accuracy = accuracy_value
                iBest = i
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {accuracy_value:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall_values]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision_values]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
            pass
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('Running Section 3.2 with Best Classifier: ' + CLASSIFIERS_NAME[iBest])
    training_examples = [1000, 5000, 10000, 15000, 20000]
    classifier = CLASSIFIERS[iBest]
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        for num_train in training_examples:
            random_indices = list(range(num_train))
            random.shuffle([random_indices])
            
            newX_train = np.take(X_train, random_indices, 0)
            newY_train = np.take(y_train, random_indices)
            if num_train == 1000:
                X_1k = newX_train
                y_1k = newY_train
            classifier.fit(newX_train, newY_train)
            y_pred = classifier.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            accuracy_value = accuracy(conf_matrix)
            outf.write(f'{num_train}: {accuracy_value:.4f}\n')
            pass
    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('Running Section 3.3')
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        k_feats = [5, 10, 15, 20, 25, 30, 35, 40, 50]
        # for each number of features k_feat, write the p-values for
        # that number of features:
        # For the 32k training set and each number of features k = {5, 50}, find the best k features according to this approach
        for k in k_feats:
            selector = SelectKBest(f_classif, k)
            X_new = selector.fit_transform(X_train, y_train)
            pp_values = np.array(selector.pvalues_)
            if k == 5:
                # Extract the indices of the top k = 5 features using the 32K dataset
                best_5_features_in_full = [i for i, x in enumerate(selector.get_support()) if x]
                print('Best 5 features in 32k...')
                print(best_5_features_in_full)
                Xfull_new = X_new
            outf.write(f'{k} p-values: {[round(pval, 4) for pval in pp_values]}\n')
        
        print('Training the best Classifier with k = 5 features')
        classifier = CLASSIFIERS[i]
        #Train on the 5 best features from the 1K dataset 
        selector = SelectKBest(f_classif, 5)
        X1k_new = selector.fit_transform(X_1k, y_1k)
        # Extract the indices of the top k = 5 features using the 1K dataset
        best_5_features_in_1k = [i for i, x in enumerate(selector.get_support()) if x]
        print('Best 5 features in 1k...')
        print(best_5_features_in_1k)
        
        #Train the classifier with 1K 
        classifier.fit(X1k_new, y_1k)
        y1k_pred = classifier.predict(selector.transform(X_test))
        conf_matrix = confusion_matrix(y_test, y1k_pred)
        accuracy_1k = accuracy(conf_matrix)
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        
        #Train the classifier with full dataset
        classifier.fit(Xfull_new, y_train)
        y_pred = classifier.predict(selector.transform(X_test))
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy_full = accuracy(conf_matrix)
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        
        #Find the k = 5 features intersection of 1k and 32k dataset
        feature_intersection = list(set(best_5_features_in_1k) & set(best_5_features_in_full))
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        #Top k = 5 features intersection of 32k dataset
        outf.write(f'Top-5 at higher: {best_5_features_in_full}\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('Running Section 3.4')
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        kf = KFold(n_splits = 5, shuffle = True)
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))
        all_accuracies_dict = defaultdict(list)
        bestClassifier_accuracies = []
        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            kfold_accuracies = []
            #Train each classifier for this fold
            for index, (key, classifier) in enumerate(CLASSIFIERS.items()):
                print("Training: " + CLASSIFIERS_NAME[key])
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                conf_matrix = confusion_matrix(y_test, y_pred)
                accuracy_value = accuracy(conf_matrix)
                kfold_accuracies.append(accuracy_value)
                if key == i:
                    bestClassifier_accuracies.append(accuracy_value)
                all_accuracies_dict[key].append(accuracy_value)
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        #Cross-Validation with the best classifier from 3.1
        p_values = []
        for index, (key, accuracies) in enumerate(all_accuracies_dict.items()):
            if key != i:
                S = ttest_rel(accuracies, bestClassifier_accuracies)
                p_values.append(S.pvalue)
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    # TODO: load data and split into train and test.
    # TODO : complete each classification experiment, in sequence.
    data = np.load(args.input)
    sample_dataset = data["arr_0"]
    X = sample_dataset[:,:-1]
    y = sample_dataset[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8, random_state = 33)
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    

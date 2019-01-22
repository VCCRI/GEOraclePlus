# Benchmark classifiers

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tablib

from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import density
from time import time


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %I:%M:%S %p")
SEED = 0
MAX_ITER = 1000
TOL = 1e-3
random.seed(SEED)


def main(train_files, output_dir, print_report, select_chi2, print_cm, print_top10, all_categories, use_hashing,
         n_features, filtered, **kwargs):
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    for train_file in train_files:
        run_benchmark(print_report, select_chi2, print_cm, print_top10, all_categories, use_hashing, n_features,
                      filtered, train_file, output_dir)


def run_benchmark(print_report, select_chi2, print_cm, print_top10, all_categories, use_hashing,
                  n_features, filtered, train_file, output_dir):
    print("Loading data set")
    data = tablib.Dataset().load(open(train_file).read(), "csv")
    random.shuffle(data)
    texts, labels = zip(*data)
    split_index = int(len(data) * 0.8)
    train_data = texts[:split_index]
    train_target = labels[:split_index]
    test_data = texts[split_index:]
    test_target = labels[split_index:]
    print('data loaded')

    # split a training set and a test set
    y_train, y_test = train_target, test_target

    print("Extracting features from the training data using a sparse vectorizer")
    if use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                       n_features=n_features)
        x_train = vectorizer.transform(train_data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        x_train = vectorizer.fit_transform(train_data)

    print("Extracting features from the test data using the same vectorizer")
    x_test = vectorizer.transform(test_data)

    # mapping from integer feature name to original token string
    if use_hashing:
        feature_names = None
    else:
        feature_names = vectorizer.get_feature_names()

    if select_chi2:
        print("Extracting %d best features by a chi-squared test" %
              select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=select_chi2)
        x_train = ch2.fit_transform(x_train, y_train)
        x_test = ch2.transform(x_test)
        if feature_names:
            # keep selected feature names
            feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))
        print()

    if feature_names:
        feature_names = np.asarray(feature_names)

    # #############################################################################
    # Benchmark classifiers
    def benchmark(clf, clf_descr=None):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(x_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(x_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(y_test, pred, average="weighted")
        print("f1 score:   %0.3f" % fscore)
        print("precision:  %0.3f" % precision)
        print("recall:     %0.3f" % recall)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            if print_top10 and feature_names is not None:
                print("top 10 keywords per class:")
                for i, label in enumerate(train_target):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
            print()

        # if parser_result.print_report:
        #     print("classification report:")
        #     print(metrics.classification_report(y_test, pred,
        #                                         target_names=target_names))

        if print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))

        print()
        if clf_descr is None:
            clf_descr = str(clf).split('(')[0]
        return clf_descr, fscore, precision, recall, train_time, test_time

    results = []
    for clf, name in (
            (RidgeClassifier(random_state=SEED), "Ridge Classifier"),
            (Perceptron(random_state=SEED, max_iter=MAX_ITER, tol=TOL), "Perceptron"),
            (PassiveAggressiveClassifier(random_state=SEED, max_iter=MAX_ITER, tol=TOL), "Passive-Aggressive"),
            (RandomForestClassifier(random_state=SEED), "Random forest"),
            (AdaBoostClassifier(random_state=SEED), "Ada Boost"),
            (BaggingClassifier(random_state=SEED), "Bagging"),
            (GradientBoostingClassifier(random_state=SEED), "Gradient Boosting"),
            (LinearSVC(random_state=SEED, max_iter=MAX_ITER, tol=TOL), "Linear SVC"),
            (SGDClassifier(random_state=SEED, max_iter=MAX_ITER, tol=TOL), "SGD Classifier"),
            (DecisionTreeClassifier(random_state=SEED), "Decision Tree"),
            (MLPClassifier(random_state=SEED), "Neural Networks"),
            (KNeighborsClassifier(), "kNN"),
            (NearestCentroid(), "Nearest Centroid"),
            (MultinomialNB(), "Multinomial NB"),
            (BernoulliNB(), "Bernoulli NB")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf))

    # print('=' * 80)
    # print("LinearSVC with L1-based feature selection")
    # # The smaller C, the stronger the regularization.
    # # The more regularization, the more sparsity.
    # results.append(benchmark(Pipeline([
    #     ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
    #                                                     tol=1e-3))),
    #     ('classification', LinearSVC(penalty="l2"))])))

    # make some plots
    indices = np.arange(len(results))
    results = [[x[i] for x in results] for i in range(len(results[0]))]

    clf_names, f1_scores, precisions, recalls, training_times, test_times = results
    clf_names = [x for _, x in sorted(zip(f1_scores, clf_names))]
    precisions = [x for _, x in sorted(zip(f1_scores, precisions))]
    recalls = [x for _, x in sorted(zip(f1_scores, recalls))]
    training_times = [x for _, x in sorted(zip(f1_scores, training_times))]
    test_times = [x for _, x in sorted(zip(f1_scores, test_times))]
    f1_scores = sorted(f1_scores)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    width = 0.15

    plt.barh(indices + width * 2, f1_scores, width, label="f1 score")
    plt.barh(indices + width * 1, recalls, width, label="recall")
    plt.barh(indices + width * 0, precisions, width, label="precision")
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.savefig(output_dir + "/" + os.path.splitext(os.path.basename(train_file))[0] + "_clf_score.pdf", format="pdf",
                dpi=600, bbox_inches='tight')
    plt.close()

    df = pd.DataFrame({"training": training_times, "testing": test_times}, index=clf_names)
    df.to_csv(output_dir + "/" + os.path.splitext(os.path.basename(train_file))[0] + "_clf_time.csv")

    precisions = [x for _, x in sorted(zip(clf_names, precisions))]
    recalls = [x for _, x in sorted(zip(clf_names, recalls))]
    f1_scores = [x for _, x in sorted(zip(clf_names, f1_scores))]
    clf_names = sorted(clf_names)
    df = pd.DataFrame({"f1_scores": f1_scores, "precisions": precisions, "recalls": recalls}, index=clf_names)
    df.index.name = "clf"
    df.to_csv(output_dir + "/" + os.path.splitext(os.path.basename(train_file))[0] + "_clf_score.csv")


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


if __name__ == '__main__':
    # parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_files", nargs="+", help="The CSV training files")
    parser.add_argument("output_dir", help="The output directory for the benchmark results")
    parser.add_argument("--report",
                        action="store_true", dest="print_report",
                        help="Print a detailed classification report.")
    parser.add_argument("--chi2_select",
                        action="store", type=int, dest="select_chi2",
                        help="Select some number of features using a chi-squared test")
    parser.add_argument("--confusion_matrix",
                        action="store_true", dest="print_cm",
                        help="Print the confusion matrix.")
    parser.add_argument("--top10",
                        action="store_true", dest="print_top10",
                        help="Print ten most discriminative terms per class"
                             " for every classifier.")
    parser.add_argument("--all_categories",
                        action="store_true", dest="all_categories",
                        help="Whether to use all categories or not.")
    parser.add_argument("--use_hashing",
                        action="store_true",
                        help="Use a hashing vectorizer.")
    parser.add_argument("--n_features",
                        action="store", type=int, default=2 ** 16,
                        help="n_features when using the hashing vectorizer.")
    parser.add_argument("--filtered",
                        action="store_true",
                        help="Remove newsgroup information that is easily overfit: "
                             "headers, signatures, and quoting.")
    parser.set_defaults(method=main)

    args = parser.parse_args()
    args.method(**vars(args))

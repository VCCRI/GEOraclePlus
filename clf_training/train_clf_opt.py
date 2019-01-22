# Train classifiers and identify optimised parameters

import matplotlib
matplotlib.use('Agg')

import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import tablib

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


SEED = 0
MAX_ITER = 10000
TOL = 1e-3
random.seed(SEED)


def main(train_file, n_threads, svc, sgd, pa, ridge, bnb, **kwargs):
    data = tablib.Dataset().load(open(train_file).read(), "csv")
    random.shuffle(data)
    texts, labels = zip(*data)
    split_index = int(len(data) * 0.8)
    train_data = texts[:split_index]
    train_target = labels[:split_index]
    test_data = texts[split_index:]
    test_target = labels[split_index:]

    results = []
    if svc:
        results.append(
            train_svc((train_data, train_target), (test_data, test_target), get_default_parameters(), n_threads)
        )

    if sgd:
        results.append(
            train_sgd((train_data, train_target), (test_data, test_target), get_default_parameters(), n_threads)
        )

    if pa:
        results.append(
            train_pa((train_data, train_target), (test_data, test_target), get_default_parameters(), n_threads)
        )

    if ridge:
        results.append(
            train_ridge((train_data, train_target), (test_data, test_target), get_default_parameters(), n_threads)
        )

    if bnb:
        results.append(
            train_bnb((train_data, train_target), (test_data, test_target), get_default_parameters(), n_threads)
        )

    indices = np.arange(len(results))
    results = [[x[i] for x in results] for i in range(len(results[0]))]
    clf_names, ori_precision, ori_recall, ori_fscore, opt_precision, opt_recall, opt_fscore = results
    clf_names = [x for _, x in sorted(zip(opt_fscore, clf_names))]
    ori_precision = [x for _, x in sorted(zip(opt_fscore, ori_precision))]
    ori_recall = [x for _, x in sorted(zip(opt_fscore, ori_recall))]
    ori_fscore = [x for _, x in sorted(zip(opt_fscore, ori_fscore))]
    opt_precision = [x for _, x in sorted(zip(opt_fscore, opt_precision))]
    opt_recall = [x for _, x in sorted(zip(opt_fscore, opt_recall))]
    opt_fscore = sorted(opt_fscore)

    plt.figure()
    plt.title("Scores")
    width = 0.1

    plt.barh(indices + width * 6, opt_fscore, width, label="optimal f1 score")
    plt.barh(indices + width * 5, ori_fscore, width, label="original f1 score")
    plt.barh(indices + width * 3.5, opt_recall, width, label="optimal recall")
    plt.barh(indices + width * 2.5, ori_recall, width, label="original recall")
    plt.barh(indices + width, opt_precision, width, label="optimal precision")
    plt.barh(indices, ori_precision, width, label="original precision")
    plt.yticks(())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.savefig(train_file.split(".")[0] + "_fs_results.pdf", format="pdf", dpi=600, bbox_inches='tight')
    plt.close()


def get_default_parameters():
    parameters = {
        "vect__ngram_range": [(1, 1), (1, 2), (1, 3), (2, 2), (3, 3)],
        "tfidf__norm": ("l1", "l2", None),
        "tfidf__use_idf": (True, False),
        "tfidf__sublinear_tf": (True, False)
    }

    return parameters


def train_sgd(training_set, testing_set, parameters, n_threads):
    clf = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("fs", SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False))),
        ("clf", SGDClassifier(max_iter=MAX_ITER, tol=TOL, random_state=SEED))
    ])

    parameters.update({
        "clf__loss": ("hinge", "log", "modified_huber", "squared_hinge", "perceptron"),
        "clf__penalty": ("none", "l2", "l1", "elasticnet"),
        "clf__alpha": [10 ** x for x in range(-4, 4)]
    })

    return ("SGD",) + get_scores(clf, training_set, testing_set, parameters, n_threads, "sgd")


def train_svc(training_set, testing_set, parameters, n_threads):
    clf = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("fs", SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False))),
        ("clf", LinearSVC(max_iter=MAX_ITER, tol=TOL, random_state=SEED))
    ])

    parameters.update({
        "clf__loss": ("hinge", "squared_hinge"),
        "clf__C": [10 ** x for x in range(-4, 4)],
        "clf__fit_intercept": (True, False)
    })

    return ("SVC",) + get_scores(clf, training_set, testing_set, parameters, n_threads, "svc")


def train_pa(training_set, testing_set, parameters, n_threads):
    clf = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("fs", SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False))),
        ("clf", PassiveAggressiveClassifier(max_iter=MAX_ITER, tol=TOL, random_state=SEED))
    ])

    parameters.update({
        "clf__C": [10 ** x for x in range(-4, 4)],
        "clf__fit_intercept": (True, False),
        "clf__loss": ("hinge", "squared_hinge")
    })

    return ("Passive Aggressive",) + get_scores(clf, training_set, testing_set, parameters, n_threads, "pa")


def train_ridge(training_set, testing_set, parameters, n_threads):
    clf = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("fs", SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False))),
        ("clf", RidgeClassifier(max_iter=MAX_ITER, tol=TOL, random_state=SEED))
    ])

    parameters.update({
        "clf__alpha": [10 ** x for x in range(-4, 4)],
        "clf__fit_intercept": (True, False),
        "clf__solver": ("auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga")
    })

    return ("Ridge",) + get_scores(clf, training_set, testing_set, parameters, n_threads, "ridge")


def train_bnb(training_set, testing_set, parameters, n_threads):
    clf = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("fs", SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False))),
        ("clf", BernoulliNB())
    ])

    parameters.update({
        "clf__alpha": [10 ** x for x in range(-4, 4)],
        "clf__fit_prior": (True, False)
    })

    return ("BernoulliNB",) + get_scores(clf, training_set, testing_set, parameters, n_threads, "bnb")


def get_scores(clf, training_set, testing_set, parameters, n_threads, clf_name):
    train_data, train_target = training_set
    test_data, test_target = testing_set

    clf.fit(train_data, train_target)
    predicted = clf.predict(test_data)
    ori_scores = metrics.precision_recall_fscore_support(test_target, predicted, average="weighted")
    print(metrics.classification_report(test_target, predicted))

    gs_clf = GridSearchCV(clf, parameters, scoring="f1_weighted", n_jobs=n_threads, error_score=0.0)
    gs_clf = gs_clf.fit(train_data, train_target)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    predicted = gs_clf.predict(test_data)
    opt_scores = metrics.precision_recall_fscore_support(test_target, predicted, average="weighted")
    print(metrics.classification_report(test_target, predicted))
    pickle.dump(gs_clf, open("pert_{}_clf.pickle".format(clf_name), "wb"))

    return ori_scores[:3] + opt_scores[:3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", help="The training file")
    parser.add_argument("n_threads", type=int, help="Number of threads")
    parser.add_argument("--svc", action="store_true", help="Train Support Vector Classifier")
    parser.add_argument("--sgd", action="store_true", help="Train Stochastic Gradient Descent Classifier")
    parser.add_argument("--pa", action="store_true", help="Train Passive Aggressive Classifier")
    parser.add_argument("--ridge", action="store_true", help="Train Ridge Classifier")
    parser.add_argument("--bnb", action="store_true", help="Train Bernoulli Naive Bayes Classifier")
    parser.set_defaults(method=main)

    args = parser.parse_args()
    args.method(**vars(args))

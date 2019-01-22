# Train classifiers using default parameters

import argparse
import pickle
import random
import tablib

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 0
MAX_ITER = 10000
TOL = 1e-3
random.seed(SEED)


def main(train_file, output_file, svc, sgd, pa, ridge, bnb, **kwargs):
    data = tablib.Dataset().load(open(train_file).read(), "csv")
    random.shuffle(data)
    texts, labels = zip(*data)
    split_index = int(len(data) * 0.8)
    train_data = texts[:split_index]
    train_target = labels[:split_index]
    test_data = texts[split_index:]
    test_target = labels[split_index:]

    pipeline = [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("fs", SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False)))
    ]

    if svc:
        pipeline.append(("clf", LinearSVC(max_iter=MAX_ITER, tol=TOL, random_state=SEED)))
    elif sgd:
        pipeline.append(("clf", SGDClassifier(max_iter=MAX_ITER, tol=TOL, random_state=SEED)))
    elif pa:
        pipeline.append(("clf", PassiveAggressiveClassifier(max_iter=MAX_ITER, tol=TOL, random_state=SEED)))
    elif ridge:
        pipeline.append(("clf", RidgeClassifier(max_iter=MAX_ITER, tol=TOL, random_state=SEED)))
    elif bnb:
        pipeline.append(("clf", BernoulliNB()))

    clf = Pipeline(pipeline)
    clf.fit(train_data, train_target)

    predicted = clf.predict(test_data)
    print(metrics.classification_report(test_target, predicted))
    pickle.dump(clf, open(output_file, "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", help="The training file")
    parser.add_argument("output_file", help="The pickled file of the trained classifier")
    parser.add_argument("--svc", action="store_true", help="Train Support Vector Classifier")
    parser.add_argument("--sgd", action="store_true", help="Train Stochastic Gradient Descent Classifier")
    parser.add_argument("--pa", action="store_true", help="Train Passive Aggressive Classifier")
    parser.add_argument("--ridge", action="store_true", help="Train Ridge Classifier")
    parser.add_argument("--bnb", action="store_true", help="Train Bernoulli Naive Bayes Classifier")
    parser.set_defaults(method=main)

    args = parser.parse_args()
    args.method(**vars(args))

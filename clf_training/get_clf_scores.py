import argparse
import pickle
import random
import tablib

from sklearn import metrics

SEED = 0
random.seed(SEED)


def main(train_file, clf_pickle, **kwargs):
    data = tablib.Dataset().load(open(train_file).read(), "csv")
    random.shuffle(data)
    texts, labels = zip(*data)
    split_index = int(len(data) * 0.8)
    test_data = texts[split_index:]
    test_target = labels[split_index:]

    clf = pickle.load(open(clf_pickle, "rb"))
    predicted = clf.predict(test_data)
    print(metrics.classification_report(test_target, predicted, digits=3))
    # opt_scores = metrics.precision_recall_fscore_support(test_target, predicted, average="weighted")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", help="The training file")
    parser.add_argument("clf_pickle", help="The pickled classfier")
    parser.set_defaults(method=main)

    args = parser.parse_args()
    args.method(**vars(args))

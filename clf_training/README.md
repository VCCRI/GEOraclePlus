# Text Classifiers Training

This section contains instructions on training text classifiers for classifying GSEs and GSMs.

## Usage

### Benchmark classifiers

Benchmark all available classifiers on scikit-learn

`python benchmark_clf.py <train_files> <output_dir>`

Detailed script usage can be access by using `-h` option

### Train classifiers

Training classifiers with default parameters

`python train_clf_ori.py <train_file> <output_file>`

Training classifiers and identify optimised parameters

`python train_clf_opt.py <train_file> <output_file>`

Detailed script usage can be access by using `-h` option

### Retrieve classifiers scores

`python get_clf_scores.py <train_file> <clf_file>`

### Extract F1 scores from benchmark results

`python get_f1_scores.py <scores_dir> <output_file>`
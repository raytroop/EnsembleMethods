"""
StackingCVClassifier
An ensemble-learning meta-classifier for stacking using cross-validation to prepare the inputs for the level-2 classifier to prevent overfitting.

https://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/
"""
import warnings
warnings.filterwarnings('ignore')

from sklearn import datasets

iris = datasets.load_iris()
# pylint: disable=E1101
X, y = iris.data[:, 1:3], iris.target
# pylint: enable=E1101

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
import numpy as np

RANDOM_SEED = 42

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = GaussianNB()
lr = LogisticRegression()

# The StackingCVClassifier uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(RANDOM_SEED)
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=lr, verbose=1, cv=3)

# print('3-fold cross validation:\n')

# for clf, label in zip([clf1, clf2, clf3, sclf],
#                       ['KNN',
#                        'Random Forest',
#                        'Naive Bayes',
#                        'StackingClassifier']):

#     scores = model_selection.cross_val_score(clf, X, y,
#                                              cv=3, scoring='accuracy')
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]"
#           % (scores.mean(), scores.std(), label))


import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10, 8))

for clf, lab, grd in zip([clf1, clf2, clf3, sclf],
                         ['KNN',
                          'Random Forest',
                          'Naive Bayes',
                          'StackingCVClassifier'],
                         itertools.product([0, 1], repeat=2)):

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(lab)
plt.show()

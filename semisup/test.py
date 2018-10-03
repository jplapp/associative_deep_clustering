import unittest

import numpy as np

#from backend import calc_correct_logit_score
from scipy.optimize import linear_sum_assignment


def confusion_matrix(labels, predictions, num_labels):
    """Compute the confusion matrix."""
    rows = []
    for i in range(num_labels):
        row = np.bincount(predictions[labels == i], minlength=num_labels)
        rows.append(row)
    return np.vstack(rows)

def calc_correct_logit_score(preds, lbls, num_labels):
    # for the correct cluster score, a one-to-one mapping of clusters to
    # classes is necessary
    # this can be done using the hungarian algorithm
    # (it is not allowed to assign two clusters to the same class label,
    # even if that would benefit overall score

    conf_mtx = confusion_matrix(lbls, preds, num_labels)

    assi = linear_sum_assignment(-conf_mtx)

    acc = conf_mtx[assi].sum() / conf_mtx.sum()

    return conf_mtx[:, assi[1]], acc


# check calc_correct_logit_score
class TestEvalulationMethods(unittest.TestCase):
    def test_simple(self):
        labels = np.array([1, 1, 0])
        preds = np.array([1, 1, 0])

        conf_mtx, acc = calc_correct_logit_score(preds, labels, 2)

        target = np.array([[1, 0], [0, 2]])

        self.assertEqual(conf_mtx.tolist(), target.tolist())
        self.assertAlmostEqual(acc, 1.0)

    def test_inversed(self):
        labels = np.array([0, 0, 1])
        preds = np.array([1, 1, 0])

        conf_mtx, acc = calc_correct_logit_score(preds, labels, 2)

        target = np.array([[2, 0], [0, 1]])

        self.assertEqual(conf_mtx.tolist(), target.tolist())
        self.assertAlmostEqual(acc, 1.0)

    def test_complex(self):
        labels = np.array([2, 2, 0, 0, 1])
        preds = np.array([1, 1, 0, 2, 2])

        conf_mtx, acc = calc_correct_logit_score(preds, labels, 3)

        target = np.array([[1,1,0],[0,1,0],[0,0,2]])

        self.assertEqual(conf_mtx.tolist(), target.tolist())
        self.assertAlmostEqual(acc, 0.80)


if __name__ == '__main__':
    unittest.main()

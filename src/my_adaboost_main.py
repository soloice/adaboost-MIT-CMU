import gl
import numpy as np
# import os
import copy
import time
from extract_haar_feature import get_int_image, calc_feature


class WeakLearner:
    def __init__(self, threshold_=0.0, less_=1.0, haar_feature_id_=-1):
        # parameter less indicates the direction of inequality: less (+1.0) or greater(-1.0)
        # decision boundary:
        self.threshold = threshold_
        self.less = less_
        self.err_rate = 1.0
        self.haar_feature_id = haar_feature_id_

    def fit(self, ordered_features, original_ids, weights_list, weights_total):
        # Train the WeakClassifier by finding the best decision boundary
        # feature should be an np array of shape (n,)
        assert type(ordered_features) == np.ndarray
        n_sample = len(original_ids)
        weight_pos_all, weight_neg_all = weights_total
        weight_pos_before, weight_neg_before = 0.0, 0.0
        sorted_weights_list = weights_list[original_ids]
        # n_pos = n_sample / 2
        n_pos = gl.n_pos
        for i in xrange(n_sample - 1):
            # Pay attention to the case in which no gaps between 2 consequent elements
            original_id = original_ids[i]  # positive examples
            if original_id < n_pos:
                weight_pos_before += sorted_weights_list[i]
            else:
                weight_neg_before += sorted_weights_list[i]

            if ordered_features[i] == ordered_features[i + 1]:
                continue
            tmp_threshold = (ordered_features[i] + ordered_features[i + 1]) / 2.0
            # if sample < threshold, positive, else negative
            tmp_err = weight_neg_before + weight_pos_all - weight_pos_before
            if tmp_err < self.err_rate:
                self.threshold, self.err_rate, self.less = tmp_threshold, tmp_err, 1.0
            # if sample > threshold, positive, else negative
            tmp_err = weight_pos_before + weight_neg_all - weight_neg_before
            if tmp_err < self.err_rate:
                self.threshold, self.err_rate, self.less = tmp_threshold, tmp_err, -1.0
        # print weight_pos_before, weight_pos_all
        # print weight_neg_before, weight_neg_all
        return self.err_rate

    def show(self):
        print 'For current weak classifier:'
        print '\tthreshold = ', self.threshold
        print '\tdirection = ', '<' if self.less > 0 else '>'
        print '\tchosen feature_id = ', self.haar_feature_id
        print '\terror_rate = ', self.err_rate
        # print self.threshold, self.less, self.haar_feature_id, self.err_rate

    def predict_one_sample(self, sample):
        # predict only one sample (given in the form of its feature)
        return np.sign(self.less * (self.threshold - sample))

    def predict(self, new_samples):
        # predict a collection of samples (given in the form of features in a 1-d np.ndarray)
        return np.array([np.sign(self.less * (self.threshold - sample)) for sample in new_samples])
        # return np.sign(self.less * (self.threshold - new_samples))


class StrongLearner:
    def __init__(self, weak_learners_, alphas_):
        # Use a list of weighted weak_learners to act as a strong learner
        self.weak_learners = weak_learners_
        self.alphas = alphas_

    def predict_one_image_score(self, int_img):
        required_features = [self.weak_learners[i].haar_feature_id for i in xrange(len(self.alphas))]
        features = [calc_feature(feature_id, int_img) for feature_id in required_features]
        return sum([self.alphas[i] * self.weak_learners[i].predict_one_sample(features[i])
                    for i in xrange(len(self.alphas))])

    def predict_one_image(self, int_img):
        return np.sign(self.predict_one_image_score(int_img))

    def predict(self, images):
        return np.array([self.predict_one_image(get_int_image(image)) for image in images])


def my_adaboost(n_pos, n_neg, raw_feature_data, n_iter=20, n_toss=300,
                feature_array=None, index_array=None):
    # assert type(labels) == np.ndarray, 'labels is not an np array'
    # assert type(data) == np.ndarray, 'data is not an np array'
    assert type(feature_array) == type(index_array)
    n_samples = n_pos + n_neg
    labels = np.array([1.0] * n_pos + [-1.0] * n_neg)
    weights = np.array([1.0 / n_samples] * n_samples)
    alphas, all_weak_learners = [0.0] * n_iter, [WeakLearner() for i in xrange(n_iter)]
    boosted_labels = [0.0] * n_samples
    n_features = feature_array.shape[0]
    # n_features = gl.n_features
    # weight_pos, weight_neg = float(n_pos) / n_samples, float(n_neg) / n_samples
    # print weight_pos, weight_neg
    t_last = time.time()
    for i in xrange(n_iter):
        print "\nIn iteration", i
        # print "weights = ", weights
        # learn a new classifier Ci
        err_i, chosen_clf = 0.5, WeakLearner()
        weight_pos, weight_neg = np.sum(weights[:n_pos]), np.sum(weights[n_pos + 1:])
        print weight_pos, weight_neg
        # randomly pick n_toss features to train
        chosen_features = np.random.permutation(xrange(n_features))[:n_toss]
        for feature_id in chosen_features:
            # for feature_id in xrange(n_features):  # it takes too long to run over all features
            clf = WeakLearner(haar_feature_id_=feature_id)
            x, y = feature_array[feature_id], index_array[feature_id]
            tmp_error_rate = clf.fit(x, y, weights, (weight_pos, weight_neg))
            if tmp_error_rate < err_i:
                err_i = tmp_error_rate
                chosen_clf = copy.deepcopy(clf)
                # print 'feature_id = ', feature_id
                # print 'tmp_error_rate = ', err_i
                if err_i < 0 or err_i > 0.49999:
                    print "eps_" + str(i) + "=" + str(err_i)
                    print "feature_id =", feature_id
                    np.save('../data/debug/w' + str(feature_id) + '.npy', weights)
                    exit(2)
        # calculate current classifier's importance
        alpha_i = 0.5 * np.log((1 - err_i) / (err_i + 1e-16))  # add 1e-16 to prevent overflow
        alphas[i], all_weak_learners[i] = alpha_i, copy.deepcopy(chosen_clf)
        print "err_i = " + str(err_i)
        print "alpha_i = " + str(alpha_i)
        chosen_clf.show()

        chosen_feature_id = chosen_clf.haar_feature_id
        predicted_labels = chosen_clf.predict(raw_feature_data[chosen_feature_id])

        # update sample weights
        weights = weights * np.exp((-alpha_i * (labels * predicted_labels)))
        weights = weights / np.sum(weights)
        assert all(weights >= 0)
        boosted_labels += alpha_i * predicted_labels

        print 'overall # of correct:', np.count_nonzero(labels * np.sign(boosted_labels) + 1.0)  # correct: 2; wrong: 0
        # print 'count it another way:', len(filter(lambda i: np.sign(boosted_labels[i]) == np.sign(labels[i]), xrange(n_samples)))
        print 'overall accuracy:', np.count_nonzero(labels * np.sign(boosted_labels) + 1.0) / (n_samples + 0.0)
        t_new = time.time()
        print 'Time for current iteration:', t_new - t_last
        t_last = t_new
    # print '\n\n alpha:', alphas
    # print 'weights:', weights
    # print 'ensemble result:', boosted_labels
    return StrongLearner(all_weak_learners, alphas)

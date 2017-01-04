from read_data import read_all_images
from extract_haar_feature import extract_haar_feature, get_int_image
from my_adaboost_main import my_adaboost
import numpy as np
import pickle
import gl
import time
from sklearn import metrics

if __name__ == '__main__':
    total_time1 = time.time()
    load_trained_model = True
    if load_trained_model:
        fh = open(gl.detector_root + 'detector1.pkl', 'rb')
        my_face_detector = pickle.load(fh)
        fh.close()
        total_time2 = time.time()
    else:
        t1 = time.time()
        read_all_images()
        load_previous_calculated_feature = False
        if load_previous_calculated_feature:
            sorted_feature = np.load('../data/haar_feature/sorted-features-train.npy')
            sorted_indices = np.load('../data/haar_feature/sorted-indices-train.npy')
            print sorted_indices.shape, sorted_indices.shape
            t2 = time.time()
            print 'Loading time:', t2 - t1
        else:
            sorted_feature, sorted_indices = extract_haar_feature('train')
        raw_feature = np.load('../data/haar_feature/raw-feature-train.npy')
        print raw_feature.shape
        t2 = time.time()
        print 'time = ', t2 - t1

        t1 = time.time()
        my_face_detector = my_adaboost(gl.n_pos, gl.n_neg, raw_feature, n_iter=80, n_toss=300,
                                       feature_array=sorted_feature, index_array=sorted_indices)
        t2 = time.time()
        print 'Training time:', t2 - t1
        # save model
        fh = open(gl.detector_root + 'detector1.pkl', 'wb')
        pickle.dump(my_face_detector, fh)
        fh.close()

        total_time2 = time.time()
        print 'total time for training phase:', total_time2 - total_time1

    # predict new examples
    test_root = '../data/faces/test'
    images = read_all_images(test_root)
    predicted_on_test_set = np.zeros(len(images))
    for img_id, img in images.items():
        predicted_on_test_set[img_id] = my_face_detector.predict_one_image(get_int_image(img))
    ground_truth = np.array([1.0] * gl.n_pos + [-1.0] * gl.n_neg)

    mat = metrics.confusion_matrix(ground_truth, predicted_on_test_set)
    print 'confusion matrix:', mat
    assert gl.n_pos == mat[1, 0] + mat[1, 1]
    precision, recall = mat[1, 1] / float(mat[0, 1] + mat[1, 1]), mat[1, 1] / float(gl.n_pos)
    print 'precision:', precision, 'recall:', recall
    print 'accuracy:', float(mat[0, 0] + mat[1, 1]) / np.sum(mat)
    print 'F1-score:', 2 * precision * recall / (precision + recall)
    total_time3 = time.time()
    print 'total time for predicting phase:', total_time3 - total_time2
    print 'total running time:', total_time3 - total_time1

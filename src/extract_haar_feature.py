import gl
import numpy as np
# import cv2
# import random
# import time


def get_int_image(img):
    # print img.shape
    # print type(img)
    # print type(img[0][0])
    height, width = img.shape[:2]
    int_img = np.array(img, dtype='int32')  # use int32 to prevent overflow
    # print int_img.shape
    # print type(int_img)
    # print type(int_img[0][0])
    for i in xrange(1, height):
        s = int_img[i, 0]
        for j in xrange(1, width):
            s += int_img[i, j]
            if i > 0:
                int_img[i, j] = int_img[i-1, j] + s
            else:
                int_img[i, j] = s
    return np.array(int_img)


def get_sum_area(int_image, x, y, w, h):
    # calculate the pixel-wise sum of a sub matrix whose top-left corner is (x, y) and with width w and height h
    if x == 0:
        if y == 0:
            return int_image[x + w - 1, y + h - 1]
        else:
            return int_image[x + w - 1, y + h - 1] - int_image[x + w - 1, y - 1]
    else:
        if y == 0:
            return int_image[x + w - 1, y + h - 1] - int_image[x - 1, y + h - 1]
        else:
            return int_image[x+w-1, y+h-1] + int_image[x-1, y-1] - int_image[x-1, y+h-1] - int_image[x+w-1, y-1]
    # if int_image < n_pos:
    #     return random.randint(0, 200)
    # else:
    #     return random.randint(50, 255)


def calc_feature(feature_id, int_img):
    # Given feature_id and integral image, calculate corresponding Haar feature
    haar_type, w, h, x, y = gl.feature_dct[feature_id]
    sf = float(int_img.shape[0]) / gl.haar_image_size  # scale factor
    w, h, x, y = int(w * sf), int(h * sf), int(x * sf), int(y * sf)
    if haar_type == 1:
        return get_sum_area(int_img, x, y, w, h) - get_sum_area(int_img, x, y+h, w, h)
    elif haar_type == 2:
        return get_sum_area(int_img, x, y, w, h) - get_sum_area(int_img, x+w, y, w, h)
    elif haar_type == 3:
        return get_sum_area(int_img, x, y, w, h) - 2 * get_sum_area(int_img, x, y+h, w, h) \
               + get_sum_area(int_img, x, y+2*h, w, h)
    elif haar_type == 4:
        return get_sum_area(int_img, x, y, w, h) - 2 * get_sum_area(int_img, x+w, y, w, h) \
               + get_sum_area(int_img, x+2*w, y, w, h)
    else:
        return get_sum_area(int_img, x, y, w, h) + get_sum_area(int_img, x+w, y+h, w, h) \
               - get_sum_area(int_img, x+w, y, w, h) - get_sum_area(int_img, x, y+h, w, h)


def extract_haar_feature(npy_name, image_dct=gl.image_dct):
    # Enumerate all feature
    # then calculate each feature in the integral image and save to file.
    # Input parameters: npy_name: name of .npy file to be saved, 'test' or 'train' recommended.
    n_samples = len(image_dct)
    n_features = gl.n_features  # Use only a small subset of features to accelerate computation in debug phase
    print '# of features:', n_features
    # feature_array[i, :]: all features of the i-th image
    # feature array[:, j]: j-th Haar feature run over all images

    feature_array = np.zeros((n_samples, n_features))
    # for file_name in image_dct.keys():
    for img_id in xrange(n_samples):  # to make sure img_id in order
        face = image_dct[img_id]
        int_face = get_int_image(face)
        for feature_id in xrange(n_features):
            feature_array[img_id, feature_id] = calc_feature(feature_id, int_face)
        img_id += 1
        if img_id % 100 == 0:
            print img_id, 'images processed...'

    # Transpose feature array so that each row represents a feature run over all images,
    # and a column contains all features of a certain image
    feature_array = feature_array.T
    feature_root = '../data/haar_feature/'
    np.save(feature_root+'raw-feature-' + npy_name + '.npy', feature_array)
    print 'feature_array shape:', feature_array.shape
    sorted_id_array = np.zeros(feature_array.shape, dtype='int32')
    for feature_id in xrange(n_features):
        current_feature = feature_array[feature_id]
        sorted_ids = np.argsort(current_feature)
        sorted_id_array[feature_id] = sorted_ids
        feature_array[feature_id] = current_feature[sorted_ids]
        # np.save(feature_root + str(feature_id) + '-ids.npy', sorted_ids)
        # np.save(feature_root + str(feature_id) + '-features.npy', feature_array[feature_id])
    np.save(feature_root + 'sorted-features-' + npy_name + '.npy', feature_array)
    np.save(feature_root + 'sorted-indices-' + npy_name + '.npy', sorted_id_array)
    return feature_array, sorted_id_array

# t1 = time.time()
# sorted_feature, sorted_indices = extract_haar_feature('train')
# t2 = time.time()
# print "Time for extracting haar feature:", t2 - t1

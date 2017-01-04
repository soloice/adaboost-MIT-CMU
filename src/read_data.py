import os
import os.path
import cv2
import gl

__author__ = 'soloice'


def read_all_images(img_root='../data/faces/train'):
    # Read all the images into two dictionary, one for image itself, one for face labels.
    # Used by almost all the other functions.
    print 'reading all images from ', img_root, '...'
    if len(gl.image_dct) > 0:
        gl.image_dct.clear()
    # if len(gl.info_dct) > 0:
    #     gl.info_dct.clear()
    img_id, gl.n_pos, gl.n_neg = 0, 0, 0
    # read faces
    for parent, dir_names, file_names in os.walk(img_root + '/face'):
        for file_name in file_names:
            full_name = os.path.join(parent, file_name)
            img = cv2.imread(full_name, cv2.IMREAD_GRAYSCALE)
            gl.image_dct[img_id] = img
            gl.n_pos += 1
            img_id += 1
            # assert type(img) == np.ndarray
            # print type(img), img.shape
            # print full_name, label
            # cv2.imshow(file_name, img)
            # cv2.waitKey(0)
            # cv2.destroyWindow(file_name)
            # input('*******')
    # read non-faces
    for parent, dir_names, file_names in os.walk(img_root + '/non-face'):
        for file_name in file_names:
            full_name = os.path.join(parent, file_name)
            img = cv2.imread(full_name, cv2.IMREAD_GRAYSCALE)
            gl.image_dct[img_id] = img
            gl.n_neg += 1
            img_id += 1
    print 'read', len(gl.image_dct), 'images!\n'
    print 'positive:', gl.n_pos, 'negative:', gl.n_neg
    return gl.image_dct

# generate_training_data()
# read_all_images()
# print len(gl.info_dct.keys()), len(gl.image_dct.keys())

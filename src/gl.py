# global constants

image_dct, feature_dct = {}, {}
n_pos, n_neg = 0, 0
haar_image_size = 19


def build_feature_dct():
    # map a feature id to a Haar feature: its type (1~5), and parameters (width, height, left, top)
    # use all Haar features of 5 five shapes, except for those with width 1.
    if len(feature_dct.keys()) > 0:
        feature_dct.clear()
    feature_id = 0
    margin = 2  # exclude Haar features which locate at the margin of the image
    # type 1: top -> down, [w; b]
    # enumerate image size: w-by-2h
    for w in xrange(2, haar_image_size+1):
        for h in xrange(1, haar_image_size/2+1):
            for left in xrange(0+margin, haar_image_size+1-w-margin):
                for top in xrange(0+margin, haar_image_size+1-2*h-margin):
                    feature_dct[feature_id] = (1, w, h, left, top)
                    feature_id += 1
    # type 2: left -> right, [w b]
    # enumerate image size: 2w-by-h
    for w in xrange(1, haar_image_size/2+1):
        for h in xrange(2, haar_image_size+1):
            for left in xrange(0, haar_image_size+1-2*w-margin):
                for top in xrange(0, haar_image_size+1-h-margin):
                    feature_dct[feature_id] = (2, w, h, left, top)
                    feature_id += 1
    # type 3: top -> down, [w; b; w]
    # enumerate image size: w-by-3h
    for w in xrange(2, haar_image_size+1):
        for h in xrange(1, haar_image_size/3+1):
            for left in xrange(0+margin, haar_image_size+1-w-margin):
                for top in xrange(0+margin, haar_image_size+1-3*h-margin):
                    feature_dct[feature_id] = (3, w, h, left, top)
                    feature_id += 1
    # type 4: left -> right, [w b w]
    # enumerate image size: 3w-by-h
    for w in xrange(1, haar_image_size/3+1):
        for h in xrange(2, haar_image_size+1):
            for left in xrange(0+margin, haar_image_size+1-3*w-margin):
                for top in xrange(0+margin, haar_image_size+1-h-margin):
                    feature_dct[feature_id] = (4, w, h, left, top)
                    feature_id += 1
    # type 5: grid-like: [w b; b w]
    # enumerate image size: 2w-by-2h
    for w in xrange(1, haar_image_size/2+1):
        for h in xrange(1, haar_image_size/2+1):
            for left in xrange(0+margin, haar_image_size+1-2*w-margin):
                for top in xrange(0+margin, haar_image_size+1-2*h-margin):
                    feature_dct[feature_id] = (5, w, h, left, top)
                    feature_id += 1
    print 'n_features = ', feature_id
    return feature_id


n_features = build_feature_dct()
detector_root = '../data/detector/'

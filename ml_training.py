# Code was written primarily by Matt Zucker with help from David Ranshous

from sklearn import svm
import cv2
import numpy as np
import glob, os
import sys
import re
import cPickle as pickle

def make_input(data):
    assert data.shape[1] == 3
    # make data into 0's/1's
    data = data.astype(np.float32) / 255.0

    # columns 0 1 2 0*0 0*1 0*2 1*1 1*2 2*2
    d0 = data[:,0].reshape(-1,1)
    d1 = data[:,1].reshape(-1,1)
    d2 = data[:,2].reshape(-1,1)

    #return data
    result = np.hstack((data, d0*d0, d0*d1, d0*d2, d1*d1, d1*d2, d2*d2))

    print 'data shape is: ', data.shape
    print 'result shape is: ', result.shape

    print data[:5]
    print result[:5]

    return result




all_train_data = []
all_train_labels = []

match_number = re.compile(r'^[0-9]+')

for filename in glob.glob('ml_data/target/*.[jJ][pP][gG]'):
    basename = os.path.basename(filename)
    root, ext = os.path.splitext(basename)
    prefix = root.rsplit('_', 1)[0]

    matches = glob.glob('ml_data/test/' + prefix + '*.[jJ][pP][gG]')

    labels = (cv2.imread(filename, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    rgb = cv2.imread(matches[0])

    # appending images to a larger numpy data structrure
    nrows = len(labels)
    assert len(rgb) == nrows

    labels = labels[nrows/2:]
    rgb = rgb[nrows/2:]

    # tall nx3 arrays
    rgb = rgb.reshape(-1, 3)

    # wide flat array of length n
    labels = labels.flatten()

    all_train_data.append(rgb)
    all_train_labels.append(labels)

# horizontal and vertical stacks of data
all_train_data = np.vstack(tuple(all_train_data))
all_train_labels = np.hstack(tuple(all_train_labels))

print all_train_data.shape
print all_train_labels.shape

train_samples = 100000

if len(all_train_labels) > train_samples:

    index = np.arange(len(all_train_labels))
    np.random.shuffle(index)
    index = index[:train_samples]

    all_train_data = all_train_data[index]
    all_train_labels = all_train_labels[index]

print all_train_data.shape
print all_train_labels.shape

#poly_svm = svm.SVC(kernel='poly', degree=2, C=1.0)
#poly_svm = svm.SVC(kernel='rbf')
poly_svm = svm.LinearSVC()

print("TRAINING ON MANY IMAGES")
poly_svm.fit(make_input(all_train_data),
             all_train_labels)


print('HERE ARE RESULTS:')
for filename in glob.glob('ml_data/test_set/*.[jJ][pP][gG]'):
    test_rgb = cv2.imread(filename)
    result = poly_svm.predict(make_input(test_rgb.reshape(-1, 3)))

    img = result.reshape(test_rgb.shape[:2]) * 255

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img_color[:len(img_color)/4] = 0
    cv2.imshow('win', np.hstack((test_rgb, img_color)))

    while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass

for filename in glob.glob('ml_data/test/*.[jJ][pP][gG]'):
    test_rgb = cv2.imread(filename)
    result = poly_svm.predict(make_input(test_rgb.reshape(-1, 3)))

    img = result.reshape(test_rgb.shape[:2]) * 255

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img_color[:len(img_color)/4] = 0
    cv2.imshow('win', np.hstack((test_rgb, img_color)))

    while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass


output = open("picklejar.pkl","wb")
pickle.dump(poly_svm,output)
output.close()

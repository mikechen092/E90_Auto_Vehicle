# Code was written primarily by Matt Zucker with help from David Ranshous

from sklearn import svm
import cv2
import numpy as np
import glob, os
import sys
import re
import pickle as pickle

def make_input(data):
    assert data.shape[1] == 5
    # make data into 0's/1's
    data = data.astype(np.float32) / 255.0

    # columns 0 1 2 3 4 0*0 0*1 0*2 0*3 0*4 1*1 1*2 1*3 1*4 2*2 2*3 2*4 3*3 3*4 4*4
    d0 = data[:,0].reshape(-1,1)
    d1 = data[:,1].reshape(-1,1)
    d2 = data[:,2].reshape(-1,1)

    d3 = data[:,3].reshape(-1,1)
    d4 = data[:,4].reshape(-1,1)

    #return data
    result = np.hstack((data, d0*d0, d0*d1, d0*d2, d0*d3, d0*d4, d1*d1, d1*d2, d1*d3, d1*d4, d2*d2, d2*d3, d2*d4, d3*d3, d3*d4, d4*d4))

    print('data shape is: ', data.shape)
    print('result shape is: ', result.shape)

    return result




all_train_data = []
all_train_labels = []

match_number = re.compile(r'^[0-9]+')

for filename in glob.glob('ml_data/target/*.[jJ][pP][gG]'):
    basename = os.path.basename(filename)
    root, ext = os.path.splitext(basename)
    prefix = root.rsplit('_', 1)[0]

    matches = glob.glob('ml_data/test/' + prefix + '*.[jJ][pP][gG]')

    # labels are reading in target -> what the pixel is classified as
    labels = (cv2.imread(filename, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

    # matches reading in test -> actual rgb values of image
    rgb = cv2.imread(matches[0])


    # appending images to a larger numpy data structrure
    nrows = len(labels)
    assert len(rgb) == nrows

    # add additional classification using the location of the pixel
    rgb_location = np.zeros((len(rgb),len(rgb[0]),5))
    for i in range(len(rgb_location)):
        for j in range(len(rgb_location[i])):
            rgb_location[i][j][3] = i
            rgb_location[i][j][4] = j
            rgb_location[i][j][:3] = rgb[i][j]

    rgb = rgb_location.reshape(-1,5)

    # wide flat array of length n
    labels = labels.flatten()

    # aggregate all the information
    all_train_data.append(rgb)
    all_train_labels.append(labels)

# horizontal and vertical stacks of data
all_train_data = np.vstack(tuple(all_train_data))
all_train_labels = np.hstack(tuple(all_train_labels))

print(all_train_data.shape)
print(all_train_labels.shape)

train_samples = 500000

# if the number of defined samples is less than the number available
# randomly choose the sample
if len(all_train_labels) > train_samples:

    index = np.arange(len(all_train_labels))
    np.random.shuffle(index)
    index = index[:train_samples]

    all_train_data = all_train_data[index]
    all_train_labels = all_train_labels[index]

print(all_train_data.shape)
print(all_train_labels.shape)

poly_svm = svm.LinearSVC()

print("TRAINING ON MANY IMAGES")
poly_svm.fit(make_input(all_train_data),
             all_train_labels)


print('HERE ARE RESULTS:')
for filename in glob.glob('ml_data/test_set/*.[jJ][pP][gG]'):
    test_rgb = cv2.imread(filename)
    test_rgb_location = np.zeros((len(test_rgb),len(test_rgb[0]),5))
    for i in range(len(test_rgb_location)):
        for j in range(len(test_rgb_location[i])):
            test_rgb_location[i][j][3] = i
            test_rgb_location[i][j][4] = j
            test_rgb_location[i][j][:3] = test_rgb[i][j]
    result = poly_svm.predict(make_input(test_rgb_location.reshape(-1, 5)))

    img = result.reshape(test_rgb.shape[:2]) * 255

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img_color[:len(img_color)/4] = 0
    cv2.imshow('win', np.hstack((test_rgb, img_color)))

    while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass

for filename in glob.glob('ml_data/test/*.[jJ][pP][gG]'):
    test_rgb = cv2.imread(filename)
    test_rgb_location = np.zeros((len(test_rgb),len(test_rgb[0]),5))
    for i in range(len(test_rgb_location)):
        for j in range(len(test_rgb_location[i])):
            test_rgb_location[i][j][3] = i
            test_rgb_location[i][j][4] = j
            test_rgb_location[i][j][:3] = test_rgb[i][j]
    result = poly_svm.predict(make_input(test_rgb_location.reshape(-1, 5)))

    img = result.reshape(test_rgb_location.shape[:2]) * 255

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img_color[:len(img_color)/4] = 0
    cv2.imshow('win', np.hstack((test_rgb, img_color)))

    while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass


output = open("picklejar.pkl","wb")
pickle.dump(poly_svm,output)
output.close()

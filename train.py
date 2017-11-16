import time
import pickle
import glob
import numpy as np
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from feature_extraction import *
from random import shuffle


# Feature extraction parameters
color_space = 'YCrCb'       # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
conv = 'RGB2YCrCb'
orient = 9                  # HOG orientations
pix_per_cell = 8            # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel = 'ALL'             # Can be 0, 1, 2, or 'ALL'
spatial_size = (48, 48)     # Spatial binning dimensions
hist_bins = 48              # Number of histogram bins
spatial_feat = True         # Spatial features on or off
hist_feat = True            # Histogram features on or off
hog_feat = True             # HOG features on or off


def train():

    cars = glob.glob('training_images/vehicles/GTI*/*.png')#+glob.glob('training_images/vehicles/GTI_Far/*.png')
    notcars = glob.glob('training_images/non-vehicles/*/*.png')

    shuffle(cars)
    shuffle(notcars)

    sample_size = 5000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]


    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)


    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    ## Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a pipeline
    scaler = StandardScaler()
    #pca = PCA(n_components=len(X[0])//2, whiten=False, svd_solver='randomized')
    #ica = FastICA(whiten=False)
    clf = LinearSVC(C=0.5, class_weight=None, dual=True, fit_intercept=True,
                    intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
                    penalty='l2', random_state=None, tol=0.0001, verbose=0)
    #clf = KNeighborsClassifier()
    #clf = SVC(kernel='rbf')
    #param_grid = {'C': [0.5, 0.8, 1.0, 1.2, 1.5],
    #              'fit_intercept': [True, False],
    #              'loss': ['hinge', 'squared_hinge'],
    #              }
    #grid_search = GridSearchCV(clf, param_grid)
    #pipeline = Pipeline([('Scaler', scaler), ('PCA', pca), ('clf', clf)])
    #pipeline = Pipeline([('Scaler', scaler), ('clf', grid_search)])
    pipeline = Pipeline([('Scaler', scaler), ('clf', clf)])
    #pipeline = Pipeline([('Scaler', scaler), ('ICA', ica), ('clf', clf)])

    # Check the training time for the pipeline
    t=time.time()
    pipeline.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train classifier...')
    #print(grid_search.best_estimator_)

    # Check the score of the classifer
    print('Test Accuracy of classifier = ', round(pipeline.score(X_test, y_test), 4))

    # Save the pipeline
    pickle.dump(pipeline, open('pipeline.pkl', 'wb'))

    return pipeline

if __name__=='__main__':
    train()

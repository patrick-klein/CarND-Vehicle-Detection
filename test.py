import sys
import cv2
import glob
import os.path
import pickle
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage.measurements import label
from train import *
from search_classify import *


def test(image, save_name):

    if os.path.isfile('pipeline.pkl'):
        clf = pickle.load(open('pipeline.pkl', 'rb'))
    else:
        clf = train()
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    scales = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    stop_vals = 3*[528] + 3*[592] + 3*[656]

    for scale, stop_val in zip(scales,stop_vals):
        out_img, box_list = find_cars(image, y_start_stop=[400, stop_val], scale=scale, clf=clf, orient=orient,
                                      pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      conv=conv, hog_channel=hog_channel)
        heat = add_heat(heat, box_list)
        cv2.imwrite('output_images/{}_1_{}.jpg'.format(save_name,scale), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    heat = cv2.GaussianBlur(heat, (7, 7), 0)
    cv2.imwrite('output_images/{}_2a.jpg'.format(save_name), cv2.applyColorMap(heat, cv2.COLORMAP_HOT))
    heat = apply_threshold(heat, 4)
    heatmap = np.clip(heat, 0, 255)
    cv2.imwrite('output_images/{}_2b.jpg'.format(save_name), cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT))

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    cv2.imwrite('output_images/{}_3.jpg'.format(save_name), cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR))



if __name__ == '__main__':

    if len(sys.argv)>1:
        img_file = sys.argv[1]
        image = mpimg.imread(img_file)
        save_name = img_file.split('/')[1].split('.')[0]
        test(image, save_name)
    else:
        test_set = sorted(glob.glob('test_images/*.jpg'))
        for img_file in test_set:
            image = mpimg.imread(img_file)
            save_name = img_file.split('/')[1].split('.')[0]
            test(image, save_name)

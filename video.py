import sys
import cv2
import os.path
import pickle
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from train import *
from search_classify import *

def video_pipeline(image):

    global heat
    heat *= 0.5

    scales = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    stop_vals = 3*[528] + 3*[592] + 3*[656]

    for scale, stop_val in zip(scales,stop_vals):
        box_list = find_cars_video(image, y_start_stop=[400, stop_val], scale=scale, clf=clf, orient=orient,
                                      pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      conv=conv, hog_channel=hog_channel)
        heat = add_heat(heat, box_list)

    heat = cv2.GaussianBlur(heat, (7, 7), 0)
    heatmap = apply_threshold(heat, 4)
    heatmap = np.clip(heatmap, 0, 255)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img



if __name__ == '__main__':

    if os.path.isfile('pipeline.pkl'):
        clf = pickle.load(open('pipeline.pkl', 'rb'))
    else:
        clf = train()

    # specify video file, else default is used
    if len(sys.argv)>1:
        video_file = sys.argv[1]
    else:
        video_file = 'project_video.mp4'

    heat = np.zeros((720,1280),dtype=np.float)

    video_output = video_file[:-4] + '_output' + video_file[-4:]
    clip = VideoFileClip(video_file)
    #clip = VideoFileClip(video_file).subclip(46,50)
    #clip = VideoFileClip(video_file).subclip(20,33)
    video_clip = clip.fl_image(video_pipeline)
    video_clip.write_videofile(video_output, audio=False)

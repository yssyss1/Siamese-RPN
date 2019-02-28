from __future__ import unicode_literals
from subprocess import check_call
from concurrent import futures
import youtube_dl
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

d_sets = ['yt_bb_detection_train', 'yt_bb_detection_validation']

col_names = ['youtube_id', 'timestamp_ms','class_id','class_name',
             'object_id','object_presence','xmin','xmax','ymin','ymax']

web_host = 'https://research.google.com/youtube-bb/'


def find_nearest_idx(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def download_and_cut(vid, data, d_set_dir):
    video_save_path = os.path.join(d_set_dir, '{}_temp.mp4'.format(vid))

    ydl_opts = {'quiet': True, 'ignoreerrors': True, 'no_warnings': True,
                'format': 'best[ext=mp4]',
                'outtmpl': video_save_path}

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['youtu.be/'+vid])

    if os.path.exists(video_save_path):
        capture = cv2.VideoCapture(video_save_path)
        fps, total_f = capture.get(5), capture.get(7)

        timestamps = [i/float(fps) for i in range(int(total_f))]
        labeled_timestamps = data['timestamp_ms'].values / 1000

        indexes = []
        for label in labeled_timestamps:
            frame_index = find_nearest_idx(timestamps, label)
            indexes.append(frame_index)

        i = 0
        for index, row in data.iterrows():
            capture.set(1,indexes[i])
            ret, image = capture.read()

            # Uncomment lines below to print bounding boxes on downloaded images
            # w, h = capture.get(3),capture.get(4)
            # x1, x2, y1, y2 = row.values[6:10]
            # x1 = int(x1*w)
            # x2 = int(x2*w)
            # y1 = int(y1*h)
            # y2 = int(y2*h)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
            i += 1

            # Make the class directory if it doesn't exist yet
            # class_dir = d_set_dir+str(row.values[2])
            class_dir = os.path.join(d_set_dir, vid)
            os.makedirs(class_dir, exist_ok=True)

            frame_path = class_dir+'/'+row.values[0]+'_'+str(row.values[1])+\
                '_'+str(row.values[2])+'_'+str(row.values[4])+'.jpg'
            cv2.imwrite(frame_path, image)

        capture.release()

    os.remove(video_save_path)
    print("{} complete!".format(video_save_path))
    return vid


def download_youtube_bb(dl_dir='./videos', num_threads=12):
    os.makedirs(dl_dir, exist_ok=True)

    for d_set in d_sets:
        d_set_dir = os.path.join(dl_dir, d_set)
        os.makedirs(os.path.join(d_set_dir), exist_ok=True)

        print('Downloading {} annotations'.format(d_set))
        check_call(['wget', web_host+d_set+'.csv.gz'])

        print('Unzipping {} annotations'.format(d_set))
        check_call(['gzip', '-d', '-f', d_set+'.csv.gz'])

        print('Parsing {} annotations into clip data'.format(d_set))
        df = pd.DataFrame.from_csv(d_set+'.csv', header=None, index_col=False)
        df.columns = col_names

        vids = df['youtube_id'].unique()

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            [executor.submit(download_and_cut, vid, df[df['youtube_id'] == vid], d_set_dir) for vid in tqdm(vids)]

        print('All {} videos downloaded'.format(d_set))


if __name__ == '__main__':
    download_youtube_bb()

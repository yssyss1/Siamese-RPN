from __future__ import unicode_literals
from subprocess import check_call
from concurrent import futures
import youtube_dl
import os
import sys
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# The data sets to be downloaded
d_sets = ['yt_bb_detection_train', 'yt_bb_detection_validation']

# Column names for detection CSV files
col_names = ['youtube_id', 'timestamp_ms','class_id','class_name',
             'object_id','object_presence','xmin','xmax','ymin','ymax']

# Host location of segment lists
web_host = 'https://research.google.com/youtube-bb/'


# Help function to get the index of the element in an array the nearest to a value
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


# Download and cut a clip to size
def dl_and_cut(vid, data, d_set_dir, pbar):
    # Use youtube_dl to download the video
    ydl_opts = {'quiet': True, 'ignoreerrors': True, 'no_warnings': True,
                'format': 'best[ext=mp4]',
                'outtmpl': './videos/'+vid+'_temp.mp4'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['youtu.be/'+vid])

    pbar.update(1)

    # Verify that the video has been downloaded. Skip otherwise
    video_path = 'videos/'+vid+'_temp.mp4'
    if os.path.exists(video_path):
        # Use opencv to open the video
        capture = cv2.VideoCapture(video_path)
        fps, total_f = capture.get(5), capture.get(7)

        # Get time stamps (in seconds) for every frame in the video
        # This is necessary because some video from YouTube come at 29.99 fps,
        # other at 30fps, other at 24fps
        timestamps = [i/float(fps) for i in range(int(total_f))]
        labeled_timestamps = data['timestamp_ms'].values / 1000

        # Get nearest frame for every labeled timestamp in CSV file
        indexes = []
        for label in labeled_timestamps:
            frame_index = find_nearest(timestamps, label)
            indexes.append(frame_index)

        i = 0
        for index, row in data.iterrows():
            # Get the actual image corresponding to the frame
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
            class_dir = d_set_dir+str(row.values[2])
            check_call(['mkdir', '-p', class_dir])

            # Save the extracted image
            frame_path = class_dir+'/'+row.values[0]+'_'+str(row.values[1])+\
                '_'+str(row.values[2])+'_'+str(row.values[4])+'.jpg'
            cv2.imwrite(frame_path, image)

        capture.release()

    # Remove the temporary video
    os.remove(d_set_dir+'/'+vid+'_temp.mp4')
    print(video_path)
    return vid


# Parse the annotation csv file and schedule downloads and cuts
def parse_and_sched(dl_dir='videos', num_threads=12):
    """Download the entire youtube-bb data set into `dl_dir`.
    """

    os.makedirs('./{}'.format(dl_dir), exist_ok=True)

    # For each of the two datasets
    for d_set in d_sets:
        d_set_dir = './{}/{}'.format(dl_dir, d_set)
        os.makedirs(os.path.join(d_set_dir), exist_ok=True)

        print('Downloading annotations: {}'.format(d_set))
        # check_call(['wget', web_host+d_set+'.csv.gz'])

        print(d_set+': Unzipping annotations...')
        # check_call(['gzip', '-d', '-f', d_set+'.csv.gz'])

        # Parse csv data using pandas
        print('Parsing annotations into clip data: {}'.format(d_set))
        df = pd.DataFrame.from_csv(d_set+'.csv', header=None, index_col=False)
        df.columns = col_names

        # Get list of unique video files
        vids = df['youtube_id'].unique()
        pbar = tqdm(total=len(vids))
        # Download and cut in parallel threads giving
        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(dl_and_cut, vid, df[df['youtube_id'] == vid], d_set_dir, pbar) for vid in vids]

        print(d_set+': All videos downloaded' )


if __name__ == '__main__':
    # Use the directory `videos` in the current working directory by
    # default, or a directory specified on the command line.
    parse_and_sched()
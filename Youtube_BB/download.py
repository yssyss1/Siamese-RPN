from pytube import YouTube
import os
from tqdm import tqdm


def download_youtube_mp4(video_id, save_foler='./videos'):
    file_name = YouTube('https://youtu.be/{}'.format(video_id))\
        .streams.filter(file_extension='mp4').first().download(save_foler)
    os.rename(file_name, os.path.join(save_foler, '{}.mp4').format(video_id))


def download_video_from_csv(csv_path, save_foler='./videos'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError('csv path is not exist')

    os.makedirs(save_foler, exist_ok=True)

    video_set = set([])

    with open(csv_path, 'r') as reader:
        for line in reader:
            video_set.add(line.split(',')[0])

    for video_id in tqdm(video_set, desc='Download video from Youtube'):
        print(video_id)
        download_youtube_mp4(video_id, save_foler)


if __name__ == '__main__':
    download_video_from_csv('./csv/youtube_boundingboxes_detection_train.csv', save_foler='./videos/validation')
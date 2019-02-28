from tqdm import tqdm
import os


def parsing_folder(root_dir):
    folder_list = os.listdir(root_dir)

    if 'parse' in folder_list:
        folder_list.remove('parse')

    save_foler = os.path.join(root_dir, 'parse')
    os.makedirs(save_foler, exist_ok=True)

    for folder in tqdm(folder_list, desc='folder parsing'):
        images = os.listdir(os.path.join(root_dir, folder))
        id_set = set()
        for image in tqdm(images, desc='{} processing'.format(folder)):
            id_set.add(image.strip('.jpg').split('_')[-1])

        [os.makedirs(os.path.join(save_foler, '{}_{}'.format(folder, id)), exist_ok=True) for id in id_set]

        for image in images:
            id = image.strip('.jpg').split('_')[-1]
            old_path = os.path.join(root_dir, folder)
            new_path = os.path.join(save_foler, '{}_{}'.format(folder, id))
            os.rename(os.path.join(old_path, image), os.path.join(new_path, image))


if __name__ == '__main__':
    parsing_folder('./videos/yt_bb_detection_validation')
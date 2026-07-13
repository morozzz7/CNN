
import os
from sklearn.model_selection import train_test_split
import shutil
import random
from scipy.io import loadmat

def get_class_names(roots_dir):
    meta_path = os.path.join(roots_dir, 'car_devkit', 'devkit', 'cars_meta.mat')
    meta = loadmat(meta_path, squeeze_me=True)
    class_names = [str(name) for name in meta['class_names']]
    return class_names


def load_annotations(root_dir, split='train'):
    if split == 'train':
        annos_path = os.path.join(root_dir, 'car_devkit', 'devkit', 'cars_train_annos.mat')
        img_dir = os.path.join(root_dir, 'cars_train', 'cars_train')
    else:
        annos_path = os.path.join(root_dir, 'car_devkit', 'devkit', 'cars_test_annos_withlabels.mat')
        img_dir = os.path.join(root_dir, 'cars_test', 'cars_test')

    annos = loadmat(annos_path, squeeze_me=True)['annotations']
    samples = []
    for ann in annos:
        fname = ann['fname']

        if isinstance(fname, bytes):
            fname = fname.decode('utf-8')
        elif hasattr(fname, '__iter__') and not isinstance(fname, str):
            fname = str(fname[0])

        img_path = os.path.join(img_dir, fname)
        label = int(ann['class']) - 1 
        samples.append((img_path, label))

    return samples


def create_folder_structure(root_dir, target_dir, selected_indices):
    train_samples = load_annotations(root_dir, 'train')
    test_samples = load_annotations(root_dir, 'test')
    class_names = get_class_names(root_dir)

    selected_set = set(selected_indices)
    filtered_train = [(img, label) for img, label in train_samples if label in selected_set]
    filtered_test = [(img, label) for img, label in test_samples if label in selected_set]

    print(f'после фильтрации train: {len(filtered_train)}, test: {len(filtered_test)}')

    for split in ['train', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)

    for label in selected_indices:
        class_name = class_names[label].replace(' ', '_').replace('/', '_')
        class_dir = f"{label:03d}_{class_name}"

        for split, samples in [('train', filtered_train), ('test', filtered_test)]:
            split_dir = os.path.join(target_dir, split, class_dir)
            os.makedirs(split_dir, exist_ok=True)

            class_samples = [img for img, lbl in samples if lbl == label]

            for img_path in class_samples:
                dst_path = os.path.join(split_dir, os.path.basename(img_path))
                shutil.copy2(img_path, dst_path)

    with open(os.path.join(target_dir, 'classes.txt'), 'w') as f:
        for idx in selected_indices:
            f.write(f'{idx:03d}: {class_names[idx]}\n')









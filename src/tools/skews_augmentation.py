import random
import cv2
import math
from os import listdir
from os.path import isfile, isdir
from tqdm import tqdm
import numpy as np


def trapezoid(src_img, l_angle, r_angle, t_angle, b_angle):
    # Coordinates that you want to Perspective Transform
    pts1 = np.float32([[0, 0], [src_img.shape[1], 0], [0, src_img.shape[0]], [src_img.shape[1], src_img.shape[0]]])

    augmented_l_x = math.tan(math.radians(l_angle)) * src_img.shape[1]
    augmented_r_x = math.tan(math.radians(r_angle)) * src_img.shape[1]

    augmented_t_y = math.tan(math.radians(t_angle)) * src_img.shape[0]
    augmented_b_y = math.tan(math.radians(b_angle)) * src_img.shape[0]
    '''
    pts2 = np.float32([
        [augmented_l_x, augmented_t_y],
        [src_img.shape[1] - augmented_r_x, 0],
        [0, src_img.shape[0]],
        [src_img.shape[1], src_img.shape[0] - augmented_b_y]
    ])
    '''
    # Вот так работает для top и bottom градусах
    '''
    pts2 = np.float32([
        [0, abs(augmented_t_y) if t_angle < 0 else 0],
        [src_img.shape[1], augmented_t_y if t_angle > 0 else 0],
        [0, src_img.shape[0] + (augmented_b_y if b_angle < 0 else 0)],
        [src_img.shape[1], src_img.shape[0] - (augmented_b_y if b_angle > 0 else 0)]
    ])
    '''
    pts2 = np.float32([
        [augmented_l_x if l_angle > 0 else 0, abs(augmented_t_y) if t_angle < 0 else 0],
        [src_img.shape[1] - (augmented_r_x if r_angle > 0 else 0), augmented_t_y if t_angle > 0 else 0],
        [abs(augmented_l_x) if l_angle < 0 else 0, src_img.shape[0] + (augmented_b_y if b_angle < 0 else 0)],
        [src_img.shape[1] - (abs(augmented_r_x) if r_angle < 0 else 0),
         src_img.shape[0] - (augmented_b_y if b_angle > 0 else 0)]
    ])

    aug = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(src_img, aug, (src_img.shape[1], src_img.shape[0]))

    for h in range(dst.shape[0]):
        for w in range(math.ceil(abs(augmented_l_x))):
            if dst[h][w].all() == 0:
                dst[h][w] = src_img[h][1]
            else:
                dst[h][w] = src_img[h][1]
                break

    for h in range(dst.shape[0]):
        for w in range(dst.shape[1] - 1, dst.shape[1] - math.ceil(abs(augmented_r_x)), -1):
            if dst[h][w].all() == 0:
                dst[h][w] = src_img[h][-2]
            else:
                dst[h][w] = src_img[h][-2]
                break

    for w in range(dst.shape[1]):
        for h in range(math.ceil(abs(augmented_t_y))):
            if dst[h][w].all() == 0:
                dst[h][w] = src_img[1][w]
            else:
                dst[h][w] = src_img[1][w]
                break

    for w in range(dst.shape[1]):
        for h in range(dst.shape[0] - 1, dst.shape[0] - math.ceil(abs(augmented_b_y)), -1):
            if dst[h][w].all() == 0:
                dst[h][w] = src_img[-2][w]
            else:
                dst[h][w] = src_img[-2][w]
                break

    return dst


def augment_image(src_fn):
    '''
    Аугментация данных:
    для каждого изображения сделаем
    3 - треугольник вверх
    1 - треугольник вниз
    1 - левый угол положительный
    1 - левый угол отрицательнй
    1 - правый угол положительный
    1 - правый угол отрицательный
    3 - случаная кобинация из всех углов
    углы во всех трансформациях будут разные, из дипазана 1-5 градусов
    '''
    augmented_imgs = []
    '''
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [-1, -1, 0, 0],
    [1, 0, 0, 0],
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, -1, 0, 0],
    [random.choice([1, 0, -1]), random.choice([1, 0, -1]), random.choice([1, 0, -1]), random.choice([1, 0, -1])],
    [random.choice([1, 0, -1]), random.choice([1, 0, -1]), random.choice([1, 0, -1]), random.choice([1, 0, -1])],
    [random.choice([1, 0, -1]), random.choice([1, 0, -1]), random.choice([1, 0, -1]), random.choice([1, 0, -1])],
    '''
    while True:
        transformation_mask = [
            [1, 1, 0, 0],
            [random.choice([1, 0, -1]), random.choice([1, 0, -1]), random.choice([1, 0, -1]), random.choice([1, 0, -1])],
        ]
        if transformation_mask[1][0] != 0 or transformation_mask[1][1] != 0 or transformation_mask[1][2] != 0 or transformation_mask[1][3] != 0:
            break

    src_img = cv2.imread(src_fn)
    for m in transformation_mask:
        angles = [
            random.uniform(1.25, 2.5) * m[0],
            random.uniform(1.25, 2.5) * m[1],
            random.uniform(1.25, 2.5) * m[2],
            random.uniform(1.25, 2.5) * m[3]
        ]

        # print(angles[0], angles[1], angles[2], angles[3])
        augmented_imgs.append(
            trapezoid(src_img, angles[0], angles[1], angles[2], angles[3])
        )

    return augmented_imgs


def join(r, d):
    return r + "/" + d


src_dir = 'E:\AITests\EbaySet\parsingTest24\primers\\ras'
dst_dir = "E:\AITests\EbaySet\parsingTest24\primers\\tmp"
src_files = [join(src_dir, f) for f in listdir(src_dir) if isfile(join(src_dir, f))]

for src_fn in tqdm(src_files):
    augmented_imgs = augment_image(src_fn)
    for id, augmented_img in enumerate(augmented_imgs):
        dst_fn = f"{src_fn[src_fn.index('/') + 1:src_fn.index('.') - 2]}_{id}.jpg"
        cv2.imwrite(join(dst_dir, dst_fn), augmented_img)
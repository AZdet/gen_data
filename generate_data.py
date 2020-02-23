from pathlib import Path
import numpy as np
from skimage import transform as ski_tf
import matplotlib.pyplot as plt
from numpy.random import randint, rand
import os

def getDataLabelPath(folder1, folder2):
    gen1 = Path(folder1).glob('*.png')
    while True:
        file1_path = next(gen1)
        file1_name = file1_path.name
        path2 = Path(folder2)
        file2_path = path2 / file1_name
        if file2_path.exists():
            yield file1_path, file2_path

def getDataPath(folder):
    gen = Path(folder).glob('*.png')
    while True:
        yield next(gen)

def getPathFromTxt(txt):
    with open(txt, 'r') as f:
        path_lst = list(f)
        length = len(path_lst)
        while True:
            idx = randint(length)
            yield path_lst[idx][:-1]



def getBBoxFromMask(mask):
    Y, X = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    mask_X = X[mask > 0.5]
    mask_Y = Y[mask > 0.5]
    return np.array([[mask_X.min(), mask_Y.min()], [mask_X.max(), mask_Y.max()]])

def unionBBox(bbox1, bbox2, low=0.1, high=0.9):
    in_lt = np.maximum(bbox1[0, :], bbox2[0, :])
    in_rb = np.minimum(bbox1[1, :], bbox2[1, :])
    un_lt = np.minimum(bbox1[0, :], bbox2[0, :])
    un_rb = np.maximum(bbox1[1, :], bbox2[1, :])
    wh = (in_rb - in_lt).clip(0)
    inter_area = np.multiply.reduce(wh)
    union_area = np.multiply.reduce(un_rb - un_lt) - inter_area
    iou = inter_area / union_area
    if low <= iou <= high:
        return np.c_[un_lt, un_rb].T
    else:
        # bbox1 and bbox2 have no overlap
        return None

def transform(mask, mode="p2c"):
    '''
    transform the mask, mode can be "p2c", "c2c", "p2p", "c2p", "r2c", "r2p"
    where p stands for person, c stands for car and r stands for random object
    '''
    if mode == "p2c":
        tx, ty = randint(-50, 50), 0
        rot = randint(-10, 10) * np.pi / 180  # from degree to radian
        scale = rand() * 0.4 + 0.6  # (0.6, 1)
    elif mode == "c2c":
        tx, ty = randint(-50, 50), 0
        rot = 0
        scale = 1
    elif mode == "p2p":
        tx, ty = randint(-50, 50), 0
        rot = 0
        scale = rand() * 0.4 + 0.6  # (0.6, 1)
    elif mode == "c2p":
        tx, ty = 0, randint(-10, 50)
        rot = 0
        scale = 1
    elif mode == "r2c" or mode == "r2p":
        tx, ty = randint(-50, 50), randint(-50, 50)
        rot = randint(-10, 10) * np.pi / 180
        scale = rand() * 0.4 + 0.6 
    else:
        raise NotImplementedError

    trans = lambda x: ski_tf.SimilarityTransform(scale=scale, rotation=rot, translation=(tx, ty)).inverse(x)
    mask2 = ski_tf.warp(mask, trans, order=3)
    return mask2

def genOccluded(mask1, img1, mask2, mode="p2c", out_size=128):
    '''
    mask1 and img1 are background image that is occluded.
    mask2 is the object in the front and creates occlusion.
    '''
    out = np.zeros((out_size, out_size, 3))
    while True:
        mask2_transformed = transform(mask2, mode)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(mask2)
        ax[1].imshow(mask2_transformed)
        plt.show()
        #import pdb; pdb.set_trace()
        bbox1 = getBBoxFromMask(mask1)
        bbox2 = getBBoxFromMask(mask2_transformed)
        union_area = unionBBox(bbox1, bbox2)
        if union_area is not None:
            res = img1
            res[mask2_transformed > 0.5] = 0  # occlusion effect
            res = res[union_area[0,1] : union_area[1,1], union_area[0,0]:union_area[1,0]]
            plt.imshow(res)
            plt.show()
            w = union_area[1, 0] - union_area[0, 0]
            h = union_area[1, 1] - union_area[0, 1]
            if w > h:
                scale = out_size / w
                res_scale = ski_tf.rescale(res, scale)
                out[out_size//2 - res_scale.shape[0]//2: out_size//2 + (res_scale.shape[0]-res_scale.shape[0]//2), :] = res_scale
            else:
                scale = out_size / h
                res_scale = ski_tf.rescale(res, scale)
                out[:, out_size//2 - res_scale.shape[1]//2: out_size//2 +(res_scale.shape[1]-res_scale.shape[1]//2)] = res_scale
            break
    return out


def generateData():
    car_gen = getPathFromTxt('real_car.txt')
    person_gen = getPathFromTxt('real_person.txt')
    img_folder = 'full'
    mask_folder = 'full_mask'
    for car_path, person_path in zip(car_gen, person_gen):
        car_img = plt.imread(os.path.join(img_folder, car_path))
        car_mask = plt.imread(os.path.join(mask_folder, car_path))[:,:,0]
        person_mask = plt.imread(os.path.join(img_folder, person_path))[:,:,0]
        out = genOccluded(car_mask, car_img, person_mask)
        plt.imshow(out)
        plt.show()
        break


if __name__ == "__main__":
    generateData()

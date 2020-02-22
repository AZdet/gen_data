from pathlib import Path
import numpy as np
from skimage import transform as ski_tf
import matplotlib.pyplot as plt
def getDataLabelPath(folder1, folder2):
    gen1 = Path(folder1).glob('*.png')
    while True:
        file1_path = next(gen1)
        file1_name = file1_path.name
        path2 = Path(folder2)
        file2_path = path2 / file1_name
        if file2_path.exists():
            yield file1_path, file2_path

def getBBoxFromMask(mask):
    Y, X = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    mask_X = X[mask > 0.5]
    mask_Y = Y[mask > 0.5]
    return np.array([[mask_X.min(), mask_Y.min()], [mask_X.max(), mask_Y.max()]])

def unionBBox(bbox1, bbox2):
    in_lt = np.maximum(bbox1[0, :], bbox2[0, :])
    in_rb = np.minimum(bbox1[1, :], bbox2[1, :])
    wh = (in_rb - in_lt).clip(0)
    if wh.all():
        return np.c_[np.minimum(bbox1[0, :], bbox2[0, :]),np.maximum(bbox1[1, :], bbox2[1, :])].T
    else:
        # bbox1 and bbox2 have no overlap
        return None

def transform(mask):
    tx, ty = 20, 20
    rot = 4 * np.pi / 180 # in radian
    scale = 0.3
    trans = lambda x: ski_tf.SimilarityTransform(scale=scale, rotation=rot, translation=(tx, ty)).inverse(x)
    mask2 = ski_tf.warp(mask, trans, order=2)
    return mask2


def genOccluded(mask1, img1, mask2, out_size=128):
    out = np.zeros((out_size, out_size, 3))
    while True:
        mask2_transformed = transform(mask2)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(mask2)
        # ax[1].imshow(mask2_transformed)
        # plt.show()
        bbox1 = getBBoxFromMask(mask1)
        bbox2 = getBBoxFromMask(mask2_transformed)
        union_area = unionBBox(bbox1, bbox2)
        if union_area is not None:
            res = img1
            res[mask2_transformed > 0.5] = 0 # occlusion effect
            res = res[union_area[0,1] : union_area[1,1], union_area[0,0]:union_area[1,0]]
            plt.imshow(res)
            plt.show()
            w = union_area[1, 0] - union_area[0, 0]
            h = union_area[1, 1] - union_area[0, 1]
            if w > h:
                scale = out_size / w
                res_scale = ski_tf.rescale(res, scale)
                out[out_size//2 - h//2: out_size//2 + h//2 + 1, :] = res_scale
            else:
                scale = out_size / h
                res_scale = ski_tf.rescale(res, scale)
                out[:, out_size//2 - w//2: out_size//2 + w//2] = res_scale
            break
    return out

if __name__ == "__main__":
    folder1 = 'full'
    folder2 = 'full_mask'
    gen = getDataLabelPath(folder1, folder2)
    img1_path, mask1_path = next(gen)
    img1 = plt.imread(str(img1_path))
    mask1 = plt.imread(str(mask1_path))
    img2_path, mask2_path = next(gen)
    img2 = plt.imread(str(img2_path))
    mask2 = plt.imread(str(mask2_path))
    out = genOccluded(mask1[:,:,0], img1, mask2[:,:,0])
    plt.imshow(out)
    plt.show()

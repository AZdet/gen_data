from pathlib import Path
import numpy as np
from skimage import transform as ski_tf
import matplotlib.pyplot as plt
from numpy.random import randint, rand
import os
from matplotlib.patches import Rectangle
eps = 3e-2 # threshold for mask to be taken as 1


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


def getBBoxFromMask(mask, eps=eps):
    Y, X = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    mask_X = X[mask > eps]
    mask_Y = Y[mask > eps]
    if mask_X.size == 0 or mask_Y.size == 0:
        return None
    return np.array([[mask_X.min(), mask_Y.min()], [mask_X.max(), mask_Y.max()]])

def unionBBox(bbox1, bbox2, low=0.2, high=0.85):
    # bbox[0] is x_min, y_min; bbox[1] is x_max, y_max
    in_lt = np.maximum(bbox1[0], bbox2[0])
    in_rb = np.minimum(bbox1[1], bbox2[1])
    un_lt = np.minimum(bbox1[0], bbox2[0])
    un_rb = np.maximum(bbox1[1], bbox2[1])
    wh = (in_rb - in_lt).clip(0)
    inter_area = np.multiply.reduce(wh)
    union_area = np.multiply.reduce(bbox1[1]-bbox1[0]) + np.multiply.reduce(bbox2[1] - bbox2[0]) - inter_area
    iou = inter_area / union_area
    if low <= iou <= high:
        # if in_lt[1] == bbox1[0, 1] and in_rb[1] == bbox1.all():
        #     import pdb; pdb.set_trace()
        #     return np.c_[bbox1[0], np.r_[in_rb[0], in_lt[1]]] 
        # import pdb; pdb.set_trace()
        # Y, X = np.mgrid[bbox1[0, 1]: bbox1[1, 1]+1, bbox1[0, 0]:bbox1[1, 0]+1]
        # inside_inter = np.logical_and(
        #     np.logical_and(X >= in_lt[0], X <= in_rb[0]), 
        #     np.logical_and(Y >= in_lt[1], Y <= in_rb[1])
        #     )
        # #import pdb; pdb.set_trace()
        # X_good = X[np.logical_not(inside_inter)]
        # Y_good = Y[np.logical_not(inside_inter)]
        # if X_good.size == 0 or Y_good.size == 0:
        #     return None
        # # hack fix, add in the bottom part 
        # # if bbox1[1,1] - Y_good.max() > 20:
        # #     return np.array([[X_good.min(), Y_good.min()], [X_good.max(), Y_good.max()+10]])
        # res = np.array([[X_good.min(), Y_good.min()], [X_good.max(), Y_good.max()]])
        # ax = plt.gca()
        # rect1 = Rectangle((bbox1[0, 0], bbox1[0,1]), bbox1[1,0]-bbox1[0,0], bbox1[1,1]-bbox1[0,1], edgecolor='r',fill=False)
        # rect2 = Rectangle((bbox2[0, 0], bbox2[0,1]), bbox2[1,0]-bbox2[0,0], bbox2[1,1]-bbox2[0,1], edgecolor='r', fill=False)        
        # rect3 = Rectangle((res[0, 0], res[0,1]), res[1,0]-res[0,0], res[1,1]-res[0,1], edgecolor='r',fill=False)
        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)
        res = bbox1
        return res
        
    else:
        # bbox1 and bbox2 have no overlap
        return None

def transform(mask, mode="p2c"):
    '''
    transform the mask, mode can be "p2c", "c2c", "p2p", "c2p", "r2c", "r2p"
    where p stands for person, c stands for car and r stands for random object
    '''
    if mode == "p2c":
        tx, ty = randint(-10, 20), 0
        rot = randint(-10, 10) * np.pi / 180  # from degree to radian
        scale = rand() * 0.4 + 0.6  # (0.6, 1)
    elif mode == "c2c":
        tx, ty = randint(-20, 30), randint(20, 40)
        rot = 0
        scale = 1.2
    elif mode == "p2p":
        tx, ty = randint(-20, 30), randint(10, 20)
        rot = 0
        scale = rand() * 0.4 + 0.6  # (0.6, 1)
    elif mode == "c2p":
        tx, ty = randint(-30, 30), randint(20, 40)
        rot = 0
        scale = 1.2
    elif mode == "random":
        tx, ty = randint(-10, 50), randint(-10, 50)
        rot = randint(-10, 10) * np.pi / 180
        scale = rand() * 0.4 + 0.8 
    elif mode == "2p2c1":
        tx, ty = randint(-10, 20), randint(30, 60)
        rot = 0
        scale = rand(0.5, 0.8)
    elif mode == "2p2c2":
        tx, ty = randint(40, 70), randint(30, 60)
        rot = 0
        scale = rand(0.5, 0.8)
    else:
        raise NotImplementedError

    trans = lambda x: ski_tf.SimilarityTransform(scale=scale, rotation=rot, translation=(tx, ty)).inverse(x)
    mask2 = ski_tf.warp(mask, trans, order=3)
    return mask2

def genOccluded(mask1, img1, mask2, mode="p2c", low_high=(0.2, 0.85), out_size=128):
    '''
    mask1 and img1 are background image that is occluded.
    mask2 is the object in the front and creates occlusion.
    '''
    out = np.zeros((out_size, out_size, 3))
    
    mask2_transformed = transform(mask2, mode)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(mask2)
    # ax[1].imshow(mask2_transformed)
    # plt.show()
    bbox1 = getBBoxFromMask(mask1)
    bbox2 = getBBoxFromMask(mask2_transformed)
    if bbox1 is None or bbox2 is None:
        return None
    union_area = unionBBox(bbox1, bbox2, low=low_high[0], high=low_high[1])
    if union_area is not None:
        res = img1
        res[mask2_transformed > eps] = 0  # occlusion 
        # plt.imshow(res)
        # plt.show()
        res = res[union_area[0,1] : union_area[1,1], union_area[0,0]:union_area[1,0]]
        # plt.imshow(union_area)
        # plt.show()
        Y, X = np.mgrid[0:res.shape[0], 0:res.shape[1]]
        # import pdb; pdb.set_trace()
        res_nonzero = (res > eps).any(axis=2)
        X_good = X[res_nonzero]
        Y_good = Y[res_nonzero]
        if (res_nonzero).sum() == 0 or X_good.size == 0 or Y_good.size == 0:
            return None
        res = res[Y_good.min():Y_good.max(), X_good.min():X_good.max()]
        # plt.imshow(res)
        # plt.show()
        w = X_good.max() - X_good.min() #union_area[1, 0] - union_area[0, 0]
        h = Y_good.max() - Y_good.min() #union_area[1, 1] - union_area[0, 1]
        if w == 0 or h == 0:
            return None
        print(max(w/h, h/w))
        #import pdb; pdb.set_trace()

        # if max(w/h, h/w) > 2.1:
        #     return None
        if w > h:
            scale = out_size / w
            res_scale = ski_tf.rescale(res, scale, order=3, multichannel=True)
            out[out_size//2 - res_scale.shape[0]//2: out_size//2 + (res_scale.shape[0]-res_scale.shape[0]//2), :] = res_scale
        else:
            scale = out_size / h
            res_scale = ski_tf.rescale(res, scale, order=3, multichannel=True)
            out[:, out_size//2 - res_scale.shape[1]//2: out_size//2 +(res_scale.shape[1]-res_scale.shape[1]//2)] = res_scale
        #import pdb; pdb.set_trace()
        return out
    else:
        return None   
    


def generateData(mode, n_iter, save_dir):
    txt2load1 = {'c2c': 'real_car.txt', 
                'c2p': 'real_person.txt',
                'p2c':'real_car.txt',
                'p2p': 'real_person.txt'}
    txt2load2 = {'c2c': 'real_car.txt', 
                'c2p': 'real_car.txt',
                'p2c':'real_person.txt',
                'p2p': 'real_person.txt'}
    gen1 = getPathFromTxt(txt2load1[mode])
    gen2 = getPathFromTxt(txt2load2[mode])
    img_folder = 'full'
    mask_folder = 'full_mask'
    out_dir = mode
    out_mask_dir = "mask_" + mode
    N_ITER = n_iter

    i = 0
    for path1, path2 in zip(gen1, gen2):
        if i == N_ITER:
            break
        print(path1)
        print(path2)
        while True:
            car_img = plt.imread(os.path.join(img_folder, path1))
            car_mask = plt.imread(os.path.join(mask_folder, path1))[:,:,0]
            person_mask = plt.imread(os.path.join(img_folder, path2))[:,:,0]
            out = genOccluded(car_mask, car_img, person_mask, mode)
            if out is None:
                path1 = next(gen1)
                path2 = next(gen2)
            else:
                break
        # plt.imshow(out)
        # plt.show()
        out_mask = out > eps
        plt.imsave(os.path.join(save_dir, out_dir, str(i)+path1), out)
        plt.imsave(os.path.join(save_dir, out_mask_dir, str(i)+path1), out_mask.astype(np.float32))
        print('========')
        i += 1

def generateData2p2c(n_iter, save_dir):  
    mode = '2p2c'
    gen1 = getPathFromTxt('real_person.txt')
    gen2 = getPathFromTxt('real_person.txt')
    gen_car = getPathFromTxt('real_car.txt')
    img_folder = 'full'
    mask_folder = 'full_mask'
    out_dir = mode
    out_mask_dir = "mask_" + mode
    N_ITER = n_iter

    i = 0
    for path1 in gen_car:
        if i == N_ITER:
            break
        print(path1)
        while True:
            target_img = plt.imread(os.path.join(img_folder, path1))
            target_mask = plt.imread(os.path.join(mask_folder, path1))[:,:,0]
            p1_path = next(gen1)
            p2_path = next(gen2)
            mask_p1 = plt.imread(os.path.join(mask_folder, p1_path))[:,:,0]
            mask_p2 = plt.imread(os.path.join(mask_folder, p2_path))[:,:,0]
            out = genOccluded(target_mask, target_img, mask_p1, mode+str(1), low_high=(0.1, 0.9))
            if out is None:
                path1 = next(gen_car)
                continue
            out_mask = (out > eps).any(axis=2)
            out = genOccluded(out_mask, out, mask_p2, mode+str(2), low_high=(0, 1))
            if out is None:
                path1 = next(gen_car)
            else:
                break
        # plt.imshow(out)
        # plt.show()
        out_mask = out > eps
        plt.imsave(os.path.join(save_dir, out_dir, str(i)+path1), out)
        plt.imsave(os.path.join(save_dir, out_mask_dir, str(i)+path1), out_mask.astype(np.float32))
        print('========')
        i += 1

def generateDataRandomMask(n_iter, save_dir):
    mode = 'random'
    txt2load = ['real_car.txt', 'real_person.txt']
    if rand() < 0.5:
        txt = txt2load[0]
    else:
        txt = txt2load[1]
    gen1 = getPathFromTxt(txt)
    img_folder = 'full'
    mask_folder = 'full_mask'
    out_dir = mode
    out_mask_dir = "mask_" + mode
    N_ITER = n_iter

    i = 0
    for path1 in gen1:
        if i == N_ITER:
            break
        print(path1)
        while True:
            target_img = plt.imread(os.path.join(img_folder, path1))
            target_mask = plt.imread(os.path.join(mask_folder, path1))[:,:,0]
            mask = get_random_mask(target_mask)
            out = genOccluded(target_mask, target_img, mask, mode,low_high=(0.1, 0.9))
            if out is None:
                path1 = next(gen1)
            else:
                break
        # plt.imshow(out)
        # plt.show()
        out_mask = out > eps
        plt.imsave(os.path.join(save_dir, out_dir, str(i)+path1), out)
        plt.imsave(os.path.join(save_dir, out_mask_dir, str(i)+path1), out_mask.astype(np.float32))
        print('========')
        i += 1

def vis_img_arr(imgs, name):
    length = len(imgs)
    fig, axes = plt.subplots(1, length)
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        ax.axis('off')
    fig.savefig(name, dpi=fig.dpi)
    plt.show()

def get_random_mask(mask1):
    '''
    generate random masks to occlude target objects
    '''
    if rand() < 0.7:
        mask = np.zeros_like(mask1, dtype=bool)
        width = randint(10, 40)
        height = 127
        x = randint(10,110)
        mask[-height:-1, x:x+width] = 1
    else:
        masks = ['random.png', 'random2.png', 'random3.png','random4.png']
        length = len(masks)
        idx = randint(length)
        mask = plt.imread(masks[idx])[:,:,1]
    return mask




if __name__ == "__main__":
    for mode in ['p2c']:
        generateData(mode, 100000-15345, 'train')
        generateData(mode, 10000, 'test')
    for mode in ['c2p']:
        generateData(mode, 50000, 'train')
        generateData(mode, 5000, 'test')

    for mode in ['c2c', 'p2p']:
        generateData(mode, 80000, 'train')
        generateData(mode, 10000, 'test')
        # gen = getDataPath('out_'+mode)
        # length = 5
        # imgs = []
        # for _ in range(length):
        #     img_path = next(gen)
        #     img = plt.imread(str(img_path))
        #     imgs.append(img)
        # vis_img_arr(imgs, name=mode+'.png')
    

import os
import cv2
import glob
import torch
import random
import numpy as np
import albumentations

from torch.utils import data as data
from scipy.linalg import orth
from torchvision.transforms import Compose, ToTensor, Normalize


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def uint2single(img):
    return np.float32(img / 255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.).round())


def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:  # add color Gaussian noise
        img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:  # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:  # add  noise
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_JPEG_noise(img):
    quality_factor = random.randint(30, 95)
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
    return img


class HazeOnlineDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.
    """

    def __init__(self, gt_folder, depth_folder, crop_size):
        super(HazeOnlineDataset, self).__init__()
        self.gt_folder = gt_folder
        self.depth_folder = depth_folder
        self.beta_range = [0.3, 1.5]
        self.A_range = [0.25, 1.0]
        self.color_p = 1.0
        self.color_range = [-0.025, 0.025]
        self.crop_size = crop_size
        self.image_names = [os.path.basename(name) for name in glob.glob(self.gt_folder + '*jpg')]
        self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def __getitem__(self, index):
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_folder + self.image_names[index]
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0

        depth_path = os.path.join(self.depth_folder, gt_path.split('/')[-1].split('.')[0] + '.npy')
        img_depth = np.load(depth_path)
        img_depth = (img_depth - img_depth.min()) / (img_depth.max() - img_depth.min())

        beta = np.random.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]
        t = np.exp(-(1 - img_depth) * 2.0 * beta)
        t = t[:, :, np.newaxis]

        A = np.random.rand(1) * (self.A_range[1] - self.A_range[0]) + self.A_range[0]
        if np.random.rand(1) < self.color_p:
            A_random = np.random.rand(3) * (self.color_range[1] - self.color_range[0]) + self.color_range[0]
            A = A + A_random

        img_lq = img_gt.copy()
        # adjust luminance
        if np.random.rand(1) < 0.5:
            img_lq = np.power(img_lq, np.random.rand(1) * 1.5 + 1.5)
        # add gaussian noise
        if np.random.rand(1) < 0.5:
            img_lq = add_Gaussian_noise(img_lq)

        # add haze
        img_lq = img_lq * t + A * (1 - t)

        # add JPEG noise
        if np.random.rand(1) < 0.5:
            img_lq = add_JPEG_noise(img_lq)

        if img_gt.shape[-1] > 3:
            img_gt = img_gt[:, :, :3]
            img_lq = img_lq[:, :, :3]
        # augmentation for training
        input_gt_size = np.min(img_gt.shape[:2])
        input_lq_size = np.min(img_lq.shape[:2])
        scale = input_gt_size // input_lq_size
        gt_size = self.crop_size

        # random resize
        if input_gt_size > gt_size:
            input_gt_random_size = random.randint(gt_size, input_gt_size)
            input_gt_random_size = input_gt_random_size - input_gt_random_size % scale  # make sure divisible by scale
            resize_factor = input_gt_random_size / input_gt_size
        else:
            resize_factor = (gt_size + 1) / input_gt_size
        img_gt = random_resize(img_gt, resize_factor)
        img_lq = random_resize(img_lq, resize_factor)
        t = random_resize(t, resize_factor)

        # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, input_gt_size // input_lq_size,
                                            gt_path)

        # flip, rotation
        img_gt, img_lq = augment([img_gt, img_lq], True, False)

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return self.norm(img_lq), self.norm(img_gt), self.image_names[index]

    def __len__(self):
        return len(self.image_names)


class TrainData(data.Dataset):
    def __init__(self, folder, crop_size):
        super().__init__()
        self.folder = folder
        self.crop_size = crop_size
        self.image_names = [f'{i}.png' for i in range(500)]

        self.rescaler = albumentations.SmallestMaxSize(max_size=crop_size)
        self.cropper = albumentations.RandomCrop(height=crop_size, width=crop_size)
        self.preprocessor = Compose(
            [ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    def get_images(self, index):
        sky_img = f'{self.folder}Sky/{self.image_names[index]}'
        desky_img = f'{self.folder}Non-sky/{self.image_names[index]}'
        img = f'{self.folder}Full/{self.image_names[index]}'
        sky_img = cv2.imread(sky_img)
        desky_img = cv2.imread(desky_img)
        img = cv2.imread(img)

        stack = np.concatenate((sky_img, desky_img, img), axis=2)
        try:
            imgs = self.cropper(image=stack)["image"]
        except Exception:
            imgs = self.rescaler(image=stack)["image"]
            imgs = self.cropper(image=imgs)["image"]

        sky = imgs[:, :, :3]
        desky = imgs[:, :, 3:6]
        full = imgs[:, :, 6:]

        sky = cv2.cvtColor(sky, cv2.COLOR_BGR2RGB)
        sky = self.preprocessor(sky / 255.).float()
        desky = cv2.cvtColor(desky, cv2.COLOR_BGR2RGB)
        desky = self.preprocessor(desky / 255.).float()
        full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
        full = self.preprocessor(full / 255.).float()

        return sky, desky, full

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.image_names)

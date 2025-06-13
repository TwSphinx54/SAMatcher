import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from threading import Thread

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

matplotlib.use('Agg')


class AverageTimer:
    """Class to help manage printing simple timing of code execution."""

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.0
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1.0 / total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """Class to help process image streams.

    Four types of possible inputs:
    1.) USB Webcam.
    2.) An IP camera
    3.) A directory of images (files in directory matching 'image_glob').
    4.) A video file, such as an .mp4 or .avi file.
    """

    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        # Initialize IP camera variables
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        
        # Initialize general variables
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        
        # Determine input type and initialize accordingly
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            # Collect all image files matching the glob patterns
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError("No images found (maybe bad 'image_glob' ?)")
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError(
                'VideoStreamer input "{}" not recognized.'.format(basedir))
        
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """Read image as grayscale and resize to img_size."""
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(grayim, (w_new, h_new), interpolation=self.interp)
        return grayim

    def next_frame(self):
        """Return the next frame, and increment internal counter."""
        if self.i == self.max_length:
            return (None, False)
        
        if self.camera:
            if self._ip_camera:
                # Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(0.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new), interpolation=self.interp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file)
        
        self.i = self.i + 1
        return (image, True)

    def start_ip_camera_thread(self):
        """Start IP camera thread for continuous frame capture."""
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        """Update IP camera frames in separate thread."""
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1

    def cleanup(self):
        """Clean up resources."""
        self._ip_running = False


# --- PREPROCESSING FUNCTIONS ---

def process_resize(w, h, resize):
    """Process resize parameters and return new dimensions."""
    assert len(resize) > 0 and len(resize) <= 2
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    return w_new, h_new


def frame2tensor(frame, device):
    """Convert frame to tensor format."""
    return torch.from_numpy(frame / 255.0).float()[None, None].to(device)


def read_overlap_image(
    path,
    device,
    resize,
    rotation,
    resize_float,
    grayscale=False,
    align='disk',
    overlap=False,
):
    """Read image with overlap processing capabilities."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if not align:
        image = image[:, :, ::-1]  # BGR to RGB
    if image is None:
        return None, None, None
    
    image = image.astype(np.float32)
    w, h = image.shape[:2][::-1]
    
    # Determine new dimensions based on alignment requirements
    if align == 'disk':
        w_new = math.ceil(w / 32) * 32
        h_new = math.ceil(h / 32) * 32
    elif align == 'loftr':
        w_new = math.ceil(w / 8) * 8
        h_new = math.ceil(h / 8) * 8
    else:
        w_new, h_new = process_resize(w, h, [-1])

    if overlap:
        if len(resize) == 1 and resize[0] == -1:
            w_new_overlap, h_new_overlap = w, h
        else:
            w_new_overlap, h_new_overlap = resize[0], resize[0]
    else:
        w_new, h_new = process_resize(w, h, resize)

    scales = (float(w) / float(w_new), float(h) / float(h_new))
    if overlap:
        overlap_scales = (
            float(w_new) / float(w_new_overlap),
            float(h_new) / float(h_new_overlap),
        )

    # Resize image based on float precision requirement
    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
        if overlap:
            overlap_image = cv2.resize(image.astype('float32'),
                                       (w_new_overlap, h_new_overlap))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')
        if overlap:
            overlap_image = cv2.resize(image.astype('float32'),
                                       (w_new_overlap, h_new_overlap))

    # Apply rotation if specified
    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    
    # Process overlap image if needed
    if overlap:
        overlap_inp = overlap_image[None]
        overlap_inp = torch.from_numpy(overlap_inp / 255.0).float().to(device)

    # Convert to tensor format
    if grayscale:
        image_g = image.copy()
        image_g = cv2.cvtColor(image_g, cv2.COLOR_BGR2GRAY)
        inp = image_g[None, None]
    else:
        inp = image.transpose((2, 0, 1))[None]
    inp = torch.from_numpy(inp / 255.0).float().to(device)

    if overlap:
        return image, overlap_inp, inp, scales, overlap_scales
    else:
        return image, inp, scales


def resize_pad_images(
        path,
        device,
        scale,
        rotation,
        resize_float,
        grayscale=False,
        size_divisor=32,
        overlap=False,
):
    """Resize and pad images to specified dimensions."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if not overlap:
        image = image[:, :, ::-1]  # BGR to RGB
    if image is None:
        return None, None, None

    h, w, c = image.shape
    scale_factor = min(scale[0] / w, scale[1] / h)
    new_size = int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)
    
    # Resize with appropriate precision
    if resize_float:
        img_resize = cv2.resize(image.astype('float32'), new_size)
    else:
        img_resize = cv2.resize(image, new_size).astype('float32')

    # Pad image to align with size divisor
    pad_scale = [
        math.ceil(scale[0] / size_divisor) * size_divisor,
        math.ceil(scale[1] / size_divisor) * size_divisor,
    ]

    overlap_inp = np.zeros((pad_scale[1], pad_scale[0], c), dtype=image.dtype)
    overlap_inp[:img_resize.shape[0], :img_resize.shape[1], :] = img_resize
    overlap_inp = overlap_inp[None]
    
    # Create mask for valid regions
    mask = np.zeros(
        (int(pad_scale[1] / size_divisor), int(pad_scale[0] / size_divisor)),
        dtype=bool)
    mask[:int(img_resize.shape[0] / size_divisor), :int(img_resize.shape[1] /
                                                        size_divisor), ] = True
    
    # Convert to tensor format
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inp = image[None, None]
    else:
        inp = image.transpose((2, 0, 1))[None]  # HxWxC to CxHxW
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    inp = torch.from_numpy(inp / 255.0).float().to(device)
    overlap_inp = torch.from_numpy(overlap_inp / 255.0).float().to(device)
    mask = torch.from_numpy(mask)[None].float().to(device)
    
    return (
        image,
        overlap_inp,
        inp,
        (1 / scale_factor, 1 / scale_factor),
        mask,
    )


def read_image(
        path,
        device,
        resize,
        rotation,
        resize_float,
        grayscale=False,
        align='disk',
        overlap=False,
):
    """Read and process image with various options."""
    if grayscale:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if not align:
            image = image[:, :, ::-1]  # BGR to RGB
    
    if image is None:
        return None, None, None
    
    image = image.astype(np.float32)
    w, h = image.shape[:2][::-1]
    w_new, h_new = process_resize(w, h, resize)
    
    # Align dimensions based on model requirements
    if align == 'disk':
        w_new = math.ceil(w_new / 16) * 16
        h_new = math.ceil(h_new / 16) * 16
    elif align == 'loftr':
        w_new = math.ceil(w_new / 8) * 8
        h_new = math.ceil(h_new / 8) * 8

    scales = (float(w) / float(w_new), float(h) / float(h_new))

    # Resize with appropriate precision
    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    # Apply rotation if specified
    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    
    # Convert to tensor format
    if overlap:
        inp = image[None]  # HxWxC to CxHxW
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        if grayscale:
            image_g = image.copy()
            image_g = cv2.cvtColor(image_g, cv2.COLOR_BGR2GRAY)
            inp = image_g[None, None]
        else:
            inp = image.transpose((2, 0, 1))[None]  # HxWxC to CxHxW
    inp = torch.from_numpy(inp / 255.0).float().to(device)

    return image, inp, scales


def overlap_crop(image1, bbox1, image2, bbox2):
    """Crop overlapping regions and normalize heights."""
    left = image1[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
    right = image2[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
    w1, h1 = left.shape[:2][::-1]
    w2, h2 = right.shape[:2][::-1]
    ratio1, ratio2 = (1, 1), (1, 1)
    
    # Normalize heights and adjust widths proportionally
    if h1 > h2:
        w_new = int(float(h1) / float(h2) * w2)
        right = cv2.resize(right.astype('float32'), (w_new, h1))
        ratio2 = float(h1) / float(h2)
    else:
        w_new = int(float(h2) / float(h1) * w1)
        left = cv2.resize(left.astype('float32'), (w_new, h2))
        ratio1 = float(h2) / float(h1)
    
    return left, right, ratio1, ratio2


def patch_resize(origin_w, origin_h, w, h, extractor_name):
    """Calculate resize ratio and new dimensions for patch processing."""
    if extractor_name != 'disk':
        if float(origin_w) / float(w) > float(origin_h) / float(h):
            ratio = float(origin_h) / float(h)
            new_w, new_h = ratio * float(w), origin_h
            ratio = [[ratio, ratio]]
        else:
            ratio = float(origin_w) / float(w)
            new_w, new_h = origin_w, ratio * float(h)
            ratio = [[ratio, ratio]]
    else:
        ratio = [[float(origin_w) / float(w), float(origin_h) / float(h)]]
        new_w, new_h = origin_w, origin_h
    return ratio, int(new_w), int(new_h)


def overlap_filter(mkpts1, bbox1, mkpts2, bbox2):
    """Filter keypoints that fall within specified bounding boxes."""
    valid1 = np.logical_and(
        np.logical_and(mkpts1[:, 0] > bbox1[0], mkpts1[:, 0] < bbox1[2]),
        np.logical_and(mkpts1[:, 1] > bbox1[1], mkpts1[:, 1] < bbox1[3]),
    )
    valid2 = np.logical_and(
        np.logical_and(mkpts2[:, 0] > bbox2[0], mkpts2[:, 0] < bbox2[2]),
        np.logical_and(mkpts2[:, 1] > bbox2[1], mkpts2[:, 1] < bbox2[3]),
    )
    valid = np.logical_and(valid1, valid2)
    return valid


def tensor_overlap_crop(image1,
                        bbox1,
                        image2,
                        bbox2,
                        extractor_name,
                        size_divisor=1):
    """Crop overlapping regions from tensor images and resize intelligently."""
    bbox1 = bbox1[0].int()
    bbox2 = bbox2[0].int()
    
    # Extract crop regions
    left = image1[0, :, bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
    right = image2[0, :, bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]

    w1, h1 = left.shape[1:][::-1]
    w2, h2 = right.shape[1:][::-1]
    
    # Choose larger crop as reference for resizing
    crop_area1 = w1 * h1
    crop_area2 = w2 * h2
    
    if crop_area1 >= crop_area2:
        reference_w, reference_h = w1, h1
    else:
        reference_w, reference_h = w2, h2
    
    # Calculate resize ratios and new dimensions
    ratio1, new_w1, new_h1 = patch_resize(reference_w, reference_h, w1, h1, extractor_name)
    ratio2, new_w2, new_h2 = patch_resize(reference_w, reference_h, w2, h2, extractor_name)

    # Convert to OpenCV format and resize
    cv_left = left.permute((1, 2, 0)).cpu().numpy() * 255
    cv_right = right.permute((1, 2, 0)).cpu().numpy() * 255

    cv_left = cv2.resize(cv_left.astype('float32'), (new_w1, new_h1),
                         interpolation=cv2.INTER_CUBIC)
    cv_right = cv2.resize(cv_right.astype('float32'), (new_w2, new_h2),
                          interpolation=cv2.INTER_CUBIC)

    # Apply size divisor alignment if needed
    if size_divisor > 1:
        new_w1 = math.ceil(new_w1 / size_divisor) * size_divisor
        new_h1 = math.ceil(new_h1 / size_divisor) * size_divisor
        cv_left = cv2.resize(cv_left.astype('float32'), (new_w1, new_h1),
                             interpolation=cv2.INTER_CUBIC)
        
        new_w2 = math.ceil(new_w2 / size_divisor) * size_divisor
        new_h2 = math.ceil(new_h2 / size_divisor) * size_divisor
        cv_right = cv2.resize(cv_right.astype('float32'), (new_w2, new_h2),
                              interpolation=cv2.INTER_CUBIC)

    # Convert back to tensor format
    if len(cv_left.shape) == 3:
        left = (torch.from_numpy(cv_left / 255).float().to(
            image1.device).permute((2, 0, 1)))
        right = (torch.from_numpy(cv_right / 255).float().to(
            image1.device).permute((2, 0, 1)))
    else:
        left = torch.from_numpy(cv_left / 255).float().to(image1.device)[None]
        right = torch.from_numpy(cv_right / 255).float().to(image1.device)[None]

    return left[None], right[None], ratio1, ratio2


def visualize_overlap_crop(image1, bbox1, image2, bbox2, output=None):
    """Visualize cropped overlap regions."""
    left = image1[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
    right = image2[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
    w1, h1 = left.shape[:2][::-1]
    w2, h2 = right.shape[:2][::-1]
    
    # Normalize heights
    if h1 > h2:
        w_new = int(float(h1) / float(h2) * w2)
        right = cv2.resize(right.astype('float32'), (w_new, h1))
    else:
        w_new = int(float(h2) / float(h1) * w1)
        left = cv2.resize(left.astype('float32'), (w_new, h2))
    
    if output:
        plot_image_pair([left, right])
        plt.savefig(str(output), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return left, right


def visualize_overlap(image1, bbox1, image2, bbox2, output=None):
    """Visualize overlap regions with bounding boxes."""
    left = cv2.rectangle(image1, tuple(bbox1[0:2]), tuple(bbox1[2:]),
                         (0, 0, 255), 7)
    right = cv2.rectangle(image2, tuple(bbox2[0:2]), tuple(bbox2[2:]),
                          (0, 0, 255), 7)
    if output:
        plot_image_pair([left, right])
        plt.savefig(str(output), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return left, right


def visualize_overlap_gt(image1, bbox1, gt1, image2, bbox2, gt2, output=None):
    """Visualize overlap regions with both predicted and ground truth boxes."""
    left = cv2.rectangle(image1, tuple(bbox1[0:2]), tuple(bbox1[2:]),
                         (255, 0, 0), 5)
    right = cv2.rectangle(image2, tuple(bbox2[0:2]), tuple(bbox2[2:]),
                          (255, 0, 0), 5)
    left = cv2.rectangle(left, tuple(gt1[0:2]), tuple(gt1[2:]), (0, 255, 0), 5)
    right = cv2.rectangle(right, tuple(gt2[0:2]), tuple(gt2[2:]), (0, 255, 0),
                          5)

    if output:
        plot_image_pair([left, right])
        plt.savefig(str(output), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return left, right


# --- GEOMETRY FUNCTIONS ---

def rotate_intrinsics(K, image_shape, rot):
    """Rotate camera intrinsics matrix based on image rotation."""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array(
            [[fy, 0.0, cy], [0.0, fx, w - 1 - cx], [0.0, 0.0, 1.0]],
            dtype=K.dtype)
    elif rot == 2:
        return np.array(
            [[fx, 0.0, w - 1 - cx], [0.0, fy, h - 1 - cy], [0.0, 0.0, 1.0]],
            dtype=K.dtype,
        )
    else:  # if rot == 3:
        return np.array(
            [[fy, 0.0, h - 1 - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]],
            dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    """Rotate pose matrix in-plane by specified rotation."""
    rotation_matrices = [
        np.array(
            [
                [np.cos(r), -np.sin(r), 0.0, 0.0],
                [np.sin(r), np.cos(r), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ) for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    """Scale camera intrinsics matrix."""
    scales = np.diag([1.0 / scales[0], 1.0 / scales[1], 1.0])
    return np.dot(scales, K)


def pose_auc(errors, thresholds):
    """Calculate Area Under Curve for pose errors."""
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


# --- VISUALIZATION FUNCTIONS ---

def plot_image_pair(imgs, dpi=100, size=20, pad=0.5):
    """Plot a pair of images side by side."""
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size * n, size * 3 / 4) if size is not None else None
    _, ax = plt.subplots(n, 1, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i].astype('uint8'),
                     cmap=plt.get_cmap('gray'),
                     vmin=0,
                     vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    """Plot keypoints on both images."""
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    """Plot matching lines between keypoints."""
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [
        matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]),
            (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1,
            transform=fig.transFigure,
            c=color[i],
            linewidth=lw,
        ) for i in range(len(kpts0))
    ]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(
        image0,
        image1,
        kpts0,
        kpts1,
        mkpts0,
        mkpts1,
        color,
        text,
        path,
        bbox0,
        bbox1,
        mask0, 
        mask1,
        show_keypoints=False,
        fast_viz=False,
        opencv_display=False,
        opencv_title='matches',
        small_text=None,
):
    """Create a comprehensive matching visualization plot."""
    if fast_viz:
        make_matching_plot_fast(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            path,
            show_keypoints,
            400,
            opencv_display,
            opencv_title,
            small_text,
        )
        return
    
    # Apply mask overlay if provided
    if mask0 is not None:
        image0 = apply_mask_overlay(image0, mask0, alpha=0.3, mask_color=(0, 255, 0))
        image1 = apply_mask_overlay(image1, mask1, alpha=0.3, mask_color=(0, 255, 0))

    # Draw bounding boxes if provided
    if bbox0 is not None:
        bbox0 = tuple(bbox0.astype(np.int8))
        bbox1 = tuple(bbox1.astype(np.int8))
        image0 = draw_bbox(image0, bbox0, tag=False)
        image1 = draw_bbox(image1, bbox1, tag=False)

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_matching_plot_fast(
        image0,
        image1,
        kpts0,
        kpts1,
        mkpts0,
        mkpts1,
        color,
        text,
        path=None,
        show_keypoints=False,
        margin=100,
        opencv_display=False,
        opencv_title='',
        small_text=None,
):
    """Create a fast matching visualization using OpenCV."""
    # Convert to grayscale if needed
    if len(image0.shape) == 3:
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]
    H, W = max(H0, H1), W0 + W1 + margin
    
    # Create output canvas
    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin:] = image1
    out = np.stack([out] * 3, -1)

    # Draw keypoints if requested
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y),
                       2,
                       black,
                       -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y),
                       1,
                       white,
                       -1,
                       lineType=cv2.LINE_AA)

    # Draw matches
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(
            out,
            (x0, y0),
            (x1 + margin + W0, y1),
            color=c,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # Draw line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales
    sc = min(H / 640.0, 2.0)

    # Add main text
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(
            out,
            t,
            (int(8 * sc), Ht * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * sc,
            txt_color_bg,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (int(8 * sc), Ht * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * sc,
            txt_color_fg,
            1,
            cv2.LINE_AA,
        )

    # Add small text at bottom
    if small_text is None:
        small_text = []
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(
            out,
            t,
            (int(8 * sc), int(H - Ht * (i + 0.6))),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5 * sc,
            txt_color_bg,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (int(8 * sc), int(H - Ht * (i + 0.6))),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5 * sc,
            txt_color_fg,
            1,
            cv2.LINE_AA,
        )
    
    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def stack_image_pair(image0, image1):
    """Stack two images with different color channels for comparison."""
    image0 = np.copy(image0)
    image1 = np.copy(image1)
    
    # Convert to 3-channel if needed
    if len(image0.shape) == 2:
        image0 = np.tile(np.expand_dims(image0, 2), [1, 1, 3])
    if len(image1.shape) == 2:
        image1 = np.tile(np.expand_dims(image1, 2), [1, 1, 3])
    
    # Change color channels for better comparison
    image0.setflags(write=1)
    image1.setflags(write=1)
    image0[:, :, 1:] = 0  # Keep only red channel
    image1[:, :, :2] = 0  # Keep only blue channel
    stack_img = cv2.addWeighted(image0, 0.5, image1, 0.5, 0)
    return stack_img


def vis_aligned_image(image0, image1, H, path=None):
    """Visualize image alignment before and after transformation."""
    warpped_image1 = cv2.warpPerspective(image1, H,
                                         (image0.shape[1], image0.shape[0]))

    before = stack_image_pair(image0, image1)
    after = stack_image_pair(image0, warpped_image1)
    align_img = np.concatenate(
        [
            cv2.resize(before, dsize=None, fx=0.5, fy=0.5),
            cv2.resize(after, dsize=None, fx=0.5, fy=0.5),
        ],
        axis=1,
    )
    if path is not None:
        cv2.imwrite(str(path), align_img)


def error_colormap(x):
    """Generate error colormap for visualization."""
    return np.clip(
        np.stack([2 - x * 2, x * 2,
                  np.zeros_like(x),
                  np.ones_like(x)], -1), 0, 1)


def get_foreground_mask(img_data, **kwargs):
    """Extract foreground mask using adaptive methods."""
    sys.path.append(
        os.path.join(os.path.dirname(__file__),
                     '../../models/ImagePreprocess'))
    from adaptive_foreground_extractor import AdaptiveForegroundExtractor

    fg_extractor = AdaptiveForegroundExtractor()
    if kwargs['method_type'] == 'method1':
        mask_data = fg_extractor.method1(
            img_data,
            min_area_close=kwargs['min_area_close'],
            close_ratio=kwargs['close_ratio'],
            remain_connect_regions_num=kwargs['remain_connect_regions_num'],
            min_area_deleting=kwargs['min_area_deleting'],
            connectivity=kwargs['connectivity'],
        )

    if kwargs['method_type'] == 'method2':
        mask_data = fg_extractor.method2(
            img_data,
            close_ratio=kwargs['close_ratio'],
            min_area_close=kwargs['min_area_close'],
            remain_connect_regions_num=kwargs['remain_connect_regions_num'],
            min_area_deleting=kwargs['min_area_deleting'],
            connectivity=kwargs['connectivity'],
            flood_fill_seed_point=kwargs['flood_fill_seed_point'],
            flood_fill_low_diff=kwargs['flood_fill_low_diff'],
            flood_fill_up_diff=kwargs['flood_fill_up_diff'],
        )

    return mask_data


def pad_mask(mask, bbox=None, outer_factor=50, inner_factor=60, threshold=-0.5):
    """Pad mask using morphological operations."""
    if bbox is None:
        n, c, h, w = mask.shape
        bbox = torch.tensor([[0, 0, h, w]]).int()
    
    outer_kernel = np.ones(
        ((bbox[0, 2] - bbox[0, 0]).int() // outer_factor, (bbox[0, 3] - bbox[0, 1]).int() // outer_factor), np.uint8
    )
    
    mask_arr = mask.cpu().squeeze().numpy().astype(np.uint8)
    mask_outer = cv2.dilate(mask_arr, outer_kernel, iterations=2)
    mask_outer = torch.tensor(mask_outer, device=mask.device).unsqueeze(0).unsqueeze(1)

    return mask_outer


def draw_bbox(img, box, score=None, gt_box=None, tag=True, background=True):
    """Draw bounding box on image with optional score and ground truth."""
    img_b = img.copy()
    
    # Create background mask
    img_b[:box[1], :, :] = 255
    img_b[box[3]:, :, :] = 255
    img_b[:, :box[0], :] = 255
    img_b[:, box[2]:, :] = 255

    # Add text backgrounds if tagging is enabled
    if tag:
        if gt_box is not None:
            cv2.rectangle(img_b, (gt_box[2] - 215, gt_box[3] - 36), gt_box[2:], (71, 99, 255), -1)
        cv2.rectangle(img_b, box[:2], (box[0] + 290, box[1] + 37), (255, 144, 30), -1)

    # Blend background if enabled
    if background:
        img = cv2.addWeighted(img, 0.4, img_b, 0.6, 0)

    # Draw ground truth box
    if gt_box is not None:
        cv2.rectangle(img, gt_box[:2], gt_box[2:], (71, 99, 255), 2)
        if tag:
            cv2.putText(
                img=img,
                text='Ground Truth',
                org=(gt_box[2] - 210, gt_box[3] - 8),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA)

    # Draw predicted box
    cv2.rectangle(img, box[:2], box[2:], (30, 144, 255), 2)
    if tag:
        text = 'Our DetMatcher' if score is None else 'Confidence:{:.3f}'.format(float(score))
        cv2.putText(
            img=img,
            text=text,
            org=(box[0] + 5, box[1] + 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA)

    return img


def apply_mask_overlay(image, mask, alpha=0.3, mask_color=(0, 0, 255)):
    """Apply mask overlay on image with specified transparency and color."""
    if mask is None:
        return image
    
    # Ensure image is 3-channel
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Process mask format
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Ensure mask is 2D
    if len(mask_np.shape) > 2:
        mask_np = mask_np.squeeze()
    
    # Scale mask to 0-1 range
    if mask_np.max() > 1:
        mask_np = mask_np.astype(np.float32) / 255.0
    
    # Resize mask to match image
    if mask_np.shape != image.shape[:2]:
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
    
    # Create colored mask
    mask_colored = np.zeros_like(image, dtype=np.float32)
    mask_colored[:, :, 0] = mask_color[0] / 255.0  # R
    mask_colored[:, :, 1] = mask_color[1] / 255.0  # G  
    mask_colored[:, :, 2] = mask_color[2] / 255.0  # B
    
    # Apply mask to each color channel
    for i in range(3):
        mask_colored[:, :, i] = mask_colored[:, :, i] * mask_np
    
    # Normalize image
    image_normalized = image.astype(np.float32)
    if image_normalized.max() > 1:
        image_normalized = image_normalized / 255.0
    
    # Apply alpha blending
    result = (1 - alpha) * image_normalized + alpha * mask_colored
    
    # Convert back to uint8
    result = (result * 255).astype(np.uint8)
    
    return result
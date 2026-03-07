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
from PIL import Image, ImageDraw, ImageFont

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

        # Safe checks for basedir type
        if isinstance(basedir, int) or (isinstance(basedir, str) and basedir.isdigit()):
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif isinstance(basedir, str) and basedir.startswith(('http', 'rtsp')):
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
    pad_square=True,
):
    """Read image, apply stride-aligned resize, optional square padding, and build tensors."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    is_rgb = False
    if not align:
        image = image[:, :, ::-1]  # BGR -> RGB
        is_rgb = True
    if image is None:
        return None, None, None

    image = image.astype(np.float32)
    w, h = image.shape[:2][::-1]

    # Infer stride from align mode.
    if align == 'disk':
        step = 16
    elif align == 'loftr':
        step = 8
    else:
        step = 1

    # 1) Resize by long side or explicit size.
    if len(resize) == 1 and resize[0] > -1:
        target = int(resize[0])
        scale = target / max(h, w)
        w_r, h_r = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    elif len(resize) == 1 and resize[0] == -1:
        w_r, h_r = w, h
    else:
        w_r, h_r = int(resize[0]), int(resize[1])

    # 2) Align resized content to stride.
    w_new = int(math.ceil(w_r / step) * step)
    h_new = int(math.ceil(h_r / step) * step)

    # 3) Map aligned content coordinates back to original image coordinates.
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    # 4) Resize to aligned content size.
    if resize_float:
        img_resized = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        img_resized = cv2.resize(image, (w_new, h_new)).astype('float32')

    # 5) Optional square padding at bottom/right.
    if pad_square:
        pad_size = max(w_new, h_new)
        if pad_size == w_new and pad_size == h_new:
            img_for_inp = img_resized
        else:
            padded = np.zeros((pad_size, pad_size, image.shape[2]), dtype=img_resized.dtype)
            padded[:h_new, :w_new, :] = img_resized
            img_for_inp = padded
    else:
        img_for_inp = img_resized

    if overlap:
        overlap_image = img_for_inp.copy()

    # 6) Apply rotation.
    if rotation != 0:
        img_for_inp = np.rot90(img_for_inp, k=rotation)
        if overlap:
            overlap_image = np.rot90(overlap_image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    # 7) Build tensors.
    if overlap:
        overlap_inp = torch.from_numpy(overlap_image[None] / 255.0).float().to(device)

    # Always compute overlap scales from aligned content (padding does not change origin mapping).
    overlap_scales = (float(w) / float(w_new), float(h) / float(h_new))
    if rotation % 2:
        overlap_scales = overlap_scales[::-1]

    # Keep visualization output in RGB.
    out_vis = img_for_inp if is_rgb else cv2.cvtColor(img_for_inp, cv2.COLOR_BGR2RGB)

    if grayscale:
        image_g = cv2.cvtColor(img_for_inp.copy(), cv2.COLOR_BGR2GRAY)
        inp = torch.from_numpy(image_g[None, None] / 255.0).float().to(device)
    else:
        inp = torch.from_numpy(img_for_inp.transpose((2, 0, 1))[None] / 255.0).float().to(device)

    if overlap:
        return out_vis, overlap_inp, inp, scales, overlap_scales
    return out_vis, inp, scales


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
        pad_square=False,  # Optional square padding; default preserves legacy behavior.
):
    """Read image with stride-aligned resize, optional square padding, and tensor conversion."""
    if grayscale:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        is_rgb = True
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        # track if we've converted to RGB already
        is_rgb = False
        if not align:
            image = image[:, :, ::-1]  # BGR to RGB
            is_rgb = True

    if image is None:
        return None, None, None

    image = image.astype(np.float32)
    w, h = image.shape[:2][::-1]

    # Infer stride from align mode.
    if align == 'disk':
        step = 16
    elif align == 'loftr':
        step = 8
    else:
        step = 1

    # Resize by long side when resize=[target], otherwise use legacy resize behavior.
    if len(resize) == 1 and resize[0] > -1:
        target = int(resize[0])
        scale = target / max(h, w)
        w_r, h_r = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    else:
        w_r, h_r = process_resize(w, h, resize)

    # Align content size to stride.
    w_new = int(math.ceil(w_r / step) * step)
    h_new = int(math.ceil(h_r / step) * step)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    # 4) 缩放到对齐后的内容尺寸
    if resize_float:
        img_resized = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        img_resized = cv2.resize(image, (w_new, h_new)).astype('float32')

    # 5) 可选：右下方形 padding
    if pad_square and (len(resize) == 1 and resize[0] > -1):
        pad_size = max(w_new, h_new)
        if pad_size != w_new or pad_size != h_new:
            padded = np.zeros((pad_size, pad_size, image.shape[2]), dtype=img_resized.dtype)
            padded[:h_new, :w_new, :] = img_resized
            img_final = padded
        else:
            img_final = img_resized
    else:
        img_final = img_resized

    # 6) 旋转
    if rotation != 0:
        img_final = np.rot90(img_final, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    # 7) 组装张量（保证 image_vis 始终为 RGB）
    if overlap:
        inp = img_final[None]
        image_vis = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB) if not is_rgb else img_final
    else:
        if grayscale:
            image_g = img_final.copy()
            image_g = cv2.cvtColor(image_g, cv2.COLOR_BGR2GRAY)
            inp = image_g[None, None]
            image_vis = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB) if not is_rgb else img_final
        else:
            inp = img_final.transpose((2, 0, 1))[None]
            image_vis = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB) if not is_rgb else img_final

    inp = torch.from_numpy(inp / 255.0).float().to(device)
    return image_vis, inp, scales


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
    """Crop tensor overlap regions and resize them with optional stride alignment."""
    bbox1 = bbox1[0].int()
    bbox2 = bbox2[0].int()

    origin_w1, origin_h1 = image1.shape[2:][::-1]
    origin_w2, origin_h2 = image2.shape[2:][::-1]

    # Extract crop regions
    left = image1[0, :, bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
    right = image2[0, :, bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]

    w1, h1 = left.shape[1:][::-1]
    w2, h2 = right.shape[1:][::-1]

    # Choose larger crop as reference for resizing
    area1 = origin_w1 * origin_h1
    area2 = origin_w2 * origin_h2

    if area1 >= area2:
        reference_w, reference_h = origin_w1, origin_h1
    else:
        reference_w, reference_h = origin_w2, origin_h2

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

    # Final resize ratio = final input size / original crop size (after alignment if used).
    ratio1 = [[float(new_w1) / max(1.0, float(w1)), float(new_h1) / max(1.0, float(h1))]]
    ratio2 = [[float(new_w2) / max(1.0, float(w2)), float(new_h2) / max(1.0, float(h2))]]
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
        ratio1 = float(h2) / float(h1)

    if output:
        plot_image_pair([left, right])
        plt.savefig(str(output), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return left, right


def visualize_overlap(image1, bbox1, image2, bbox2, output=None):
    """Visualize overlap regions with bounding boxes."""
    # work on copies, convert RGB->BGR for cv2 drawing, then back to RGB
    img1 = _to_hwc_uint8(image1)
    img2 = _to_hwc_uint8(image2)
    # bgr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    # bgr2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    bgr1 = img1
    bgr2 = img2
    cv2.rectangle(bgr1, tuple(bbox1[0:2]), tuple(bbox1[2:]), (255, 0, 0), 7)
    cv2.rectangle(bgr2, tuple(bbox2[0:2]), tuple(bbox2[2:]), (255, 0, 0), 7)
    left = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    if output:
        plot_image_pair([left, right])
        plt.savefig(str(output), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return left, right


def visualize_overlap_gt(image1, bbox1, gt1, image2, bbox2, gt2, output=None):
    """Visualize overlap regions with both predicted and ground truth boxes."""
    img1 = _to_hwc_uint8(image1)
    img2 = _to_hwc_uint8(image2)
    bgr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    bgr2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr1, tuple(bbox1[0:2]), tuple(bbox1[2:]), (0, 0, 255), 5)
    cv2.rectangle(bgr2, tuple(bbox2[0:2]), tuple(bbox2[2:]), (0, 0, 255), 5)
    cv2.rectangle(bgr1, tuple(gt1[0:2]), tuple(gt1[2:]), (0, 255, 0), 5)
    cv2.rectangle(bgr2, tuple(gt2[0:2]), tuple(gt2[2:]), (0, 255, 0), 5)
    left = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
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


# --- VISUALIZATION HELPERS (CONSOLIDATED) ---

def _to_hwc_uint8(img):
    """Convert many image formats to HWC uint8 (safe)."""
    if img is None:
        return None
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = np.array(img)
    # handle batch dims
    if img.ndim == 4:
        img = img[0]
    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    # gray -> 3 channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    # float -> uint8
    if img.dtype.kind == 'f':
        if img.max() <= 1.0:
            img = (img * 255.0).round()
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def apply_mask_overlay(image, mask, alpha=0.35, mask_color=(0, 0, 0)):
    """Overlay a mask on an RGB image by darkening masked-out pixels (mask==0)."""
    if image is None:
        return image
    img_u8 = _to_hwc_uint8(image)
    img = img_u8.astype(np.float32) / 255.0
    if mask is None:
        return img_u8

    if torch.is_tensor(mask):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.array(mask)
    mask_np = np.squeeze(mask_np).astype(np.float32)
    if mask_np.max() > 1.0:
        mask_np = mask_np / 255.0
    mask_np = np.clip(mask_np, 0.0, 1.0)

    binary_like = np.logical_or(np.isclose(mask_np, 0.0), np.isclose(mask_np, 1.0)).all()
    if binary_like:
        alpha = max(alpha, 0.70)

    interp = cv2.INTER_NEAREST if binary_like else cv2.INTER_LINEAR

    # Resize helper: preserve aspect ratio and pad bottom/right (content anchored top-left).
    def _resize_pad_mask_top_left(m, target_h, target_w, interpolation):
        mh, mw = m.shape[:2]
        if mh <= 0 or mw <= 0:
            return np.zeros((target_h, target_w), dtype=np.float32)
        # aspect-preserving scale (like your image preprocessing)
        s = min(float(target_w) / float(mw), float(target_h) / float(mh))
        new_w = max(1, int(round(mw * s)))
        new_h = max(1, int(round(mh * s)))
        m_rs = cv2.resize(m, (new_w, new_h), interpolation=interpolation)
        out = np.zeros((target_h, target_w), dtype=np.float32)
        out[:new_h, :new_w] = m_rs
        return out

    if mask_np.shape != img.shape[:2]:
        Ht, Wt = int(img.shape[0]), int(img.shape[1])
        mh, mw = int(mask_np.shape[0]), int(mask_np.shape[1])
        # For square-padded targets, avoid stretching: resize + top-left placement.
        if Ht == Wt and (mh != Ht or mw != Wt):
            mask_r = _resize_pad_mask_top_left(mask_np, Ht, Wt, interpolation=interp)
        else:
            mask_r = cv2.resize(mask_np, (Wt, Ht), interpolation=interp)
    else:
        mask_r = mask_np

    mask_r = np.clip(mask_r, 0.0, 1.0)[..., None]  # 1=keep
    inv = 1.0 - mask_r
    color_layer = np.zeros_like(img)
    color_layer[..., 0] = mask_color[0] / 255.0
    color_layer[..., 1] = mask_color[1] / 255.0
    color_layer[..., 2] = mask_color[2] / 255.0
    out = (1.0 - alpha * inv) * img + (alpha * inv) * color_layer
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def draw_small_label(img, label_text, font_size=12, padding=4):
    """Draw a small black box with white text at top-left. Input/output: RGB uint8."""
    if img is None:
        return img
    im = _to_hwc_uint8(img)
    pil = Image.fromarray(im)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()

    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), label_text, font=font)
        tw, th = (r - l), (b - t)
    else:
        tw, th = draw.textsize(label_text, font=font)

    draw.rectangle([0, 0, tw + 2 * padding, th + 2 * padding], fill=(0, 0, 0))
    draw.text((padding, padding), label_text, fill=(255, 255, 255), font=font)
    return np.array(pil)


def draw_bbox(img, box, score=None, gt_box=None, tag=True, background=False):
    """Draw predicted/gt boxes on RGB image and return a new RGB image."""
    if img is None:
        return img
    img_out = img.copy()
    h_img, w_img = img_out.shape[:2]

    x0 = int(round(float(box[0])))
    y0 = int(round(float(box[1])))
    x1 = int(round(float(box[2])))
    y1 = int(round(float(box[3])))

    x0 = max(0, min(w_img - 1, x0))
    x1 = max(0, min(w_img - 1, x1))
    y0 = max(0, min(h_img - 1, y0))
    y1 = max(0, min(h_img - 1, y1))

    if background:
        mask = np.ones_like(img_out) * 255
        mask[y0:y1, x0:x1, :] = img_out[y0:y1, x0:x1, :]
        img_out = cv2.addWeighted(img_out, 0.6, mask, 0.4, 0)

    cv2.rectangle(img_out, (x0, y0), (x1, y1), (255, 0, 0), 2)

    if gt_box is not None and len(gt_box) >= 4:
        gx0 = int(round(float(gt_box[0])))
        gy0 = int(round(float(gt_box[1])))
        gx1 = int(round(float(gt_box[2])))
        gy1 = int(round(float(gt_box[3])))

        gx0 = max(0, min(w_img - 1, gx0))
        gx1 = max(0, min(w_img - 1, gx1))
        gy0 = max(0, min(h_img - 1, gy0))
        gy1 = max(0, min(h_img - 1, gy1))

        cv2.rectangle(img_out, (gx0, gy0), (gx1, gy1), (200, 100, 0), 2)

    # Optional text tag intentionally omitted for cleaner default rendering.
    return img_out


def get_foreground_mask(img_data, **kwargs):
    """Thin wrapper to project's AdaptiveForegroundExtractor (keeps same signature)."""
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../models/ImagePreprocess'))
    try:
        from adaptive_foreground_extractor import AdaptiveForegroundExtractor
    except Exception as e:
        raise ImportError("AdaptiveForegroundExtractor not found: %s" % e)
    fg_extractor = AdaptiveForegroundExtractor()
    method = kwargs.get('method_type', 'method1')
    if method == 'method1':
        return fg_extractor.method1(
            img_data,
            min_area_close=kwargs.get('min_area_close'),
            close_ratio=kwargs.get('close_ratio'),
            remain_connect_regions_num=kwargs.get('remain_connect_regions_num'),
            min_area_deleting=kwargs.get('min_area_deleting'),
            connectivity=kwargs.get('connectivity'),
        )
    elif method == 'method2':
        return fg_extractor.method2(
            img_data,
            close_ratio=kwargs.get('close_ratio'),
            min_area_close=kwargs.get('min_area_close'),
            remain_connect_regions_num=kwargs.get('remain_connect_regions_num'),
            min_area_deleting=kwargs.get('min_area_deleting'),
            connectivity=kwargs.get('connectivity'),
            flood_fill_seed_point=kwargs.get('flood_fill_seed_point'),
            flood_fill_low_diff=kwargs.get('flood_fill_low_diff'),
            flood_fill_up_diff=kwargs.get('flood_fill_up_diff'),
        )
    else:
        raise ValueError("get_foreground_mask: unsupported method_type '%s'" % method)


def make_matching_plot(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text, path,
        bbox0=None, bbox1=None, mask0=None, mask1=None,
        show_keypoints=False, fast_viz=False, opencv_display=False,
        opencv_title='matches', small_text=None, label0=None, label1=None, margin=100,
        crop_mask0=None, crop_mask1=None, match_keep=None, match_weights=None,
        auto_crop_zero_pad=True,
        covisualize=False,
        covis_path_suffix='_covis',
        covis_mask0=None,
        covis_mask1=None,
):
    """
    Render two images side-by-side with optional masks, boxes, keypoints and matches.
    Optionally save an additional covisibility visualization.
    """
    img0 = _to_hwc_uint8(image0)
    img1 = _to_hwc_uint8(image1)
    # Assume inputs are already RGB; do NOT flip channels here.

    # --- New: trim right/bottom zero-padding ---
    def _trim_rb_zero(img):
        if img.ndim == 3:
            nz_rows = np.where(img.sum(axis=(1, 2)) > 0)[0]
            nz_cols = np.where(img.sum(axis=(0, 2)) > 0)[0]
        else:
            nz_rows = np.where(img.sum(axis=1) > 0)[0]
            nz_cols = np.where(img.sum(axis=0) > 0)[0]
        if nz_rows.size and nz_cols.size:
            rmax = int(nz_rows[-1]) + 1
            cmax = int(nz_cols[-1]) + 1
            return img[:rmax, :cmax]
        return img

    if auto_crop_zero_pad:
        img0 = _trim_rb_zero(img0)
        img1 = _trim_rb_zero(img1)

    # apply global masks (mask interpreted as 1=valid)
    if mask0 is not None:
        img0 = apply_mask_overlay(img0, mask0, alpha=0.5, mask_color=(0, 0, 0))
    if mask1 is not None:
        img1 = apply_mask_overlay(img1, mask1, alpha=0.5, mask_color=(0, 0, 0))

    # overlay crop masks inside bbox if provided
    def _apply_crop_tint_in_bbox(img, crop_mask, bbox, color=(30, 180, 30), alpha=0.25):
        if crop_mask is None or bbox is None:
            return img
        if torch.is_tensor(crop_mask):
            cm_np = crop_mask.detach().cpu().numpy()
        else:
            cm_np = np.array(crop_mask)
        cm_np = np.squeeze(cm_np).astype(np.float32)
        if cm_np.max() > 1.0:
            cm_np = cm_np / 255.0
        cm_np = np.clip(cm_np, 0.0, 1.0)

        x0,y0,x1,y1 = (bbox.detach().cpu().numpy().reshape(-1)[:4].astype(int)
                        if torch.is_tensor(bbox) else np.array(bbox).reshape(-1)[:4].astype(int))
        x0,y0 = max(0,x0), max(0,y0)
        x1,y1 = min(img.shape[1],x1), min(img.shape[0],y1)
        if x1 <= x0 or y1 <= y0:
            return img
        w_box, h_box = x1 - x0, y1 - y0

        # --- NEW: crop_mask is derived from binary masks used for filtering -> keep it crisp ---
        cm_r = cv2.resize(cm_np, (w_box, h_box), interpolation=cv2.INTER_NEAREST)
        cm_r = np.clip(cm_r, 0.0, 1.0)[..., None]

        tint = np.zeros((h_box, w_box, 3), dtype=np.float32)
        tint[..., 0] = color[0] / 255.0
        tint[..., 1] = color[1] / 255.0
        tint[..., 2] = color[2] / 255.0
        sub = img[y0:y1, x0:x1].astype(np.float32) / 255.0
        out_sub = (1.0 - alpha * (1.0 - cm_r)) * sub + (alpha * (1.0 - cm_r)) * tint
        out = img.copy()
        out[y0:y1, x0:x1] = np.clip(out_sub * 255.0, 0, 255).astype(np.uint8)
        return out

    if crop_mask0 is not None and bbox0 is not None:
        img0 = _apply_crop_tint_in_bbox(img0, crop_mask0, bbox0)
    if crop_mask1 is not None and bbox1 is not None:
        img1 = _apply_crop_tint_in_bbox(img1, crop_mask1, bbox1)

    # draw labels
    # if label0:
    #     img0 = draw_small_label(img0, label0, font_size=12)
    # if label1:
    #     img1 = draw_small_label(img1, label1, font_size=12)

    # draw bboxes (operate on RGB arrays; color tuple values won't affect coords)
    if bbox0 is not None:
        b0 = bbox0.detach().cpu().numpy() if torch.is_tensor(bbox0) else np.array(bbox0)
        b0 = b0.reshape(-1)[:4].astype(np.int32)
        img0 = draw_bbox(img0, tuple(b0.tolist()))
    if bbox1 is not None:
        b1 = bbox1.detach().cpu().numpy() if torch.is_tensor(bbox1) else np.array(bbox1)
        b1 = b1.reshape(-1)[:4].astype(np.int32)
        img1 = draw_bbox(img1, tuple(b1.tolist()))

    # prepare canvas (RGB)
    H0, W0 = img0.shape[:2]
    H1, W1 = img1.shape[:2]
    H = max(H0, H1)
    W = W0 + W1 + margin
    canvas = 255 * np.ones((H, W, 3), dtype=np.uint8)
    canvas[:H0, :W0] = img0
    canvas[:H1, W0 + margin: W0 + margin + W1] = img1

    draw_canvas = canvas

    # draw keypoints if requested (draw on draw_canvas)
    if show_keypoints and (kpts0 is not None and kpts1 is not None):
        k0 = np.round(np.asarray(kpts0)).astype(int)
        k1 = np.round(np.asarray(kpts1)).astype(int)
        for x, y in k0:
            cv2.circle(draw_canvas, (int(x), int(y)), 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(draw_canvas, (int(x), int(y)), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        for x, y in k1:
            cv2.circle(draw_canvas, (int(x), int(y)), 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(draw_canvas, (int(x), int(y)), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # draw matches (mkpts0 and mkpts1 are Nx2) on draw_canvas
    if mkpts0 is not None and mkpts1 is not None and len(mkpts0) > 0 and len(mkpts1) > 0:
        mk0 = np.round(np.asarray(mkpts0)).astype(int)
        mk1 = np.round(np.asarray(mkpts1)).astype(int)

        # If postfilter info provided -> color by kept/filtered and modulate by weight
        color_arr = None
        if match_keep is not None:
            mk_bool = np.asarray(match_keep).astype(bool)
            if mk_bool.shape[0] != len(mk0):
                mk_bool = mk_bool[:len(mk0)]
            if match_weights is None:
                weights = np.ones(len(mk0), dtype=np.float32)
            else:
                weights = np.asarray(match_weights, dtype=np.float32)
                if weights.shape[0] != len(mk0):
                    weights = np.ones(len(mk0), dtype=np.float32)
            # create RGB color per match: kept -> green, filtered -> red; scale by weight
            color_arr = np.zeros((len(mk0), 3), dtype=int)
            green = np.array([0, 255, 0], dtype=float)   # RGB green
            red = np.array([255, 0, 0], dtype=float)     # RGB red
            for i in range(len(mk0)):
                base = green if mk_bool[i] else red
                w = float(np.clip(weights[i], 0.0, 1.0))
                scale = 0.6 + 0.4 * w if mk_bool[i] else 0.25 + 0.75 * w
                col = (base * scale).astype(int)
                color_arr[i, :] = np.clip(col, 0, 255)
        else:
            # fallback to existing color semantics (assume input colors are RGB)
            if color is None or len(color) == 0:
                color_arr = np.tile(np.array([255, 255, 255]), (len(mk0), 1))  # white fallback (RGB)
            else:
                c = np.asarray(color)
                if c.ndim == 1 and c.shape[0] == len(mk0):
                    mask_bool = c.astype(bool)
                    color_arr = np.zeros((len(mk0), 3), dtype=int)
                    color_arr[mask_bool] = np.array([0, 255, 0], dtype=int)   # GREEN (RGB)
                    color_arr[~mask_bool] = np.array([255, 0, 0], dtype=int)  # RED (RGB)
                else:
                    if c.dtype.kind == 'f' and c.max() <= 1.0:
                        c = (c[:, :3] * 255).astype(int)
                    else:
                        c = c[:, :3].astype(int)
                    # input assumed RGB -> keep as RGB
                    color_arr = c[:, :3]

        for (x0, y0), (x1, y1), col in zip(mk0, mk1, color_arr):
            col_t = tuple(int(v) for v in col.tolist())
            pt0 = (int(x0), int(y0))
            pt1 = (int(x1 + margin + W0), int(y1))
            cv2.line(draw_canvas, pt0, pt1, col_t, 1, lineType=cv2.LINE_AA)
            cv2.circle(draw_canvas, pt0, 2, col_t, -1, lineType=cv2.LINE_AA)
            cv2.circle(draw_canvas, pt1, 2, col_t, -1, lineType=cv2.LINE_AA)

    # canvas already RGB (we drew in-place)
    # add small_text legend if provided (compose as a short label)
    # if small_text is not None:
    #     legend = '\n'.join(small_text)
    #     for i, line in enumerate(small_text):
    #         y_offset = i * 14
    #         canvas = draw_small_label(canvas, line, font_size=10, padding=4)

    # save or display
    if path is not None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        cv2.imwrite(str(path), canvas[..., ::-1])

    # Optional covisibility output on fixed 1024x1024 padded inputs.
    if covisualize and path is not None:
        raw0 = _to_hwc_uint8(image0)
        raw1 = _to_hwc_uint8(image1)

        # Resize each image by long side to 1024, then pad to 1024x1024 (bottom/right).
        def _resize_pad_1024(img):
            h, w = img.shape[:2]
            if h == 0 or w == 0:
                return np.zeros((1024, 1024, 3), dtype=np.uint8)
            scale = 1024.0 / max(h, w)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas_1024 = np.zeros((1024, 1024, 3), dtype=np.uint8)
            canvas_1024[:new_h, :new_w] = img_resized
            return canvas_1024

        im0_1024 = _resize_pad_1024(raw0)
        im1_1024 = _resize_pad_1024(raw1)

        # Resize covisibility masks with nearest-neighbor to preserve hard boundaries.
        def _resize_mask_nn(m, target_h=1024, target_w=1024):
            if m is None:
                return np.zeros((target_h, target_w), dtype=np.float32)
            if torch.is_tensor(m):
                m_np = m.detach().cpu().float().squeeze().numpy()
            else:
                m_np = np.array(m)
            m_np = np.squeeze(m_np)
            if m_np.ndim == 3:
                m_np = m_np.squeeze(0)
            if m_np.ndim != 2:
                return np.zeros((target_h, target_w), dtype=np.float32)
            if m_np.max() > 1.0:
                m_np = m_np / 255.0
            m_np = np.clip(m_np, 0.0, 1.0).astype(np.float32)

            # --- NEW: if target is square-padded, preserve aspect and pad bottom/right (top-left content) ---
            mh, mw = m_np.shape
            if target_h == target_w and (mh != target_h or mw != target_w):
                s = min(float(target_w) / float(mw), float(target_h) / float(mh))
                new_w = max(1, int(round(mw * s)))
                new_h = max(1, int(round(mh * s)))
                m_rs = cv2.resize(m_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                out = np.zeros((target_h, target_w), dtype=np.float32)
                out[:new_h, :new_w] = m_rs
                return out

            return cv2.resize(m_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        m0_1024 = _resize_mask_nn(covis_mask0)
        m1_1024 = _resize_mask_nn(covis_mask1)

        # Apply semi-transparent magenta fill with opaque magenta contours.
        def _apply_covis_with_contour(im, m, color=(255, 0, 255), alpha=0.35):
            """Overlay covisibility region and contour on an RGB uint8 image."""
            im_f = im.astype(np.float32)
            out = im_f.copy()  # <-- FIX: define `out` before using it
            m = np.clip(m, 0.0, 1.0)
            # 二值 mask 用于区域 + 轮廓
            bin_m = (m > 0.4).astype(np.uint8)
            if bin_m.sum() == 0:
                return im.copy()

            # 半透明区域叠加
            mask3 = np.repeat(bin_m[..., None], 3, axis=-1).astype(bool)
            tint = np.zeros_like(im_f)
            tint[..., 0] = color[0]
            tint[..., 1] = color[1]
            tint[..., 2] = color[2]
            out[mask3] = ((1.0 - alpha) * out[mask3] + alpha * tint[mask3])

            # 勾勒边缘（不透明洋红线条）
            # 注意：cv2.findContours 要求单通道 uint8
            contours, _hier = cv2.findContours(
                bin_m,
                cv2.RETR_CCOMP,  # Keep both outer and inner contours.
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, color, 2, lineType=cv2.LINE_AA)

            return np.clip(out, 0, 255).astype(np.uint8)

        im0_covis = _apply_covis_with_contour(im0_1024, m0_1024)
        im1_covis = _apply_covis_with_contour(im1_1024, m1_1024)

        # 5. 水平拼接成 1024x2048
        cov_canvas = np.zeros((1024, 2048, 3), dtype=np.uint8)
        cov_canvas[:1024, :1024] = im0_covis
        cov_canvas[:1024, 1024:2048] = im1_covis

        # 6. 保存共视图（BGR）
        root, ext = os.path.splitext(path)
        cov_path = root + covis_path_suffix + ext
        cv2.imwrite(str(cov_path), cov_canvas[..., ::-1])

    if opencv_display:
        cv2.imshow(opencv_title, canvas[..., ::-1])
        cv2.waitKey(1)
    return canvas


# def plot_image_pair(images, titles=None, figsize=(12, 6), cmap=None):
#     """Plot list of HWC images side-by-side on current matplotlib figure.
#     Kept minimal: does not save/close the figure (callers use plt.savefig / plt.close)."""
#     n = len(images)
#     plt.figure(figsize=figsize)
#     for i, img in enumerate(images):
#         ax = plt.subplot(1, n, i + 1)
#         im = _to_hwc_uint8(img)
#         # handle grayscale display
#         if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
#             if isinstance(cmap, (list, tuple)):
#                 cmap_i = cmap[i] if i < len(cmap) else None
#             else:
#                 cmap_i = cmap
#             ax.imshow(im[..., 0] if im.ndim == 3 else im, cmap=cmap_i)
#         else:
#             ax.imshow(im)
#         ax.axis('off')
#         if titles and i < len(titles):
#             ax.set_title(titles[i])
#     plt.tight_layout()
    
def plot_image_pair(imgs, dpi=100, size=20, pad=0.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size * n, size * 3 / 4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(
            imgs[i].astype('uint8'),
            cmap=plt.get_cmap('gray'),
            vmin=0,
            vmax=255,
        )
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def vis_aligned_image(
    ref_image,
    src_image,
    H=None,
    bbox_ref=None,
    bbox_src=None,
    output=None,
    alpha=0.5,
    border_value=(0, 0, 0),
):
    """
    Visualize src_image aligned to ref_image by homography H (src -> ref).
    If H is None or invalid, return a side-by-side canvas.
    """
    im_ref = _to_hwc_uint8(ref_image)
    im_src = _to_hwc_uint8(src_image)
    if im_ref is None or im_src is None:
        return None

    h_ref, w_ref = im_ref.shape[:2]
    h_src, w_src = im_src.shape[:2]

    def _draw_xyxy(img, box, color, offset_x=0):
        if box is None:
            return
        b = np.asarray(box).reshape(-1)
        if b.size < 4:
            return
        x0, y0, x1, y1 = [int(round(float(v))) for v in b[:4]]
        cv2.rectangle(img, (x0 + offset_x, y0), (x1 + offset_x, y1), color, 2)

    if H is None:
        margin = 20
        Hc = max(h_ref, h_src)
        Wc = w_ref + w_src + margin
        canvas = 255 * np.ones((Hc, Wc, 3), dtype=np.uint8)
        canvas[:h_ref, :w_ref] = im_ref
        canvas[:h_src, w_ref + margin:w_ref + margin + w_src] = im_src

        _draw_xyxy(canvas, bbox_ref, (255, 0, 0), offset_x=0)
        _draw_xyxy(canvas, bbox_src, (0, 255, 0), offset_x=w_ref + margin)

        if output is not None:
            os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
            cv2.imwrite(str(output), canvas[..., ::-1])
        return canvas

    H_np = np.asarray(H, dtype=np.float64)
    if H_np.shape != (3, 3) or not np.isfinite(H_np).all():
        return vis_aligned_image(ref_image, src_image, H=None, bbox_ref=bbox_ref, bbox_src=bbox_src, output=output)

    warped = cv2.warpPerspective(
        im_src,
        H_np,
        (w_ref, h_ref),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    gray_warp = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    mask = (gray_warp > 0).astype(np.float32)[..., None]

    ref_f = im_ref.astype(np.float32) / 255.0
    warped_f = warped.astype(np.float32) / 255.0
    alpha_arr = alpha * mask
    out_f = ref_f * (1.0 - alpha_arr) + warped_f * alpha_arr
    out = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)

    _draw_xyxy(out, bbox_ref, (255, 0, 0), offset_x=0)

    if bbox_src is not None:
        b = np.asarray(bbox_src).reshape(-1)
        if b.size >= 4:
            xs = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], dtype=np.float64)
            ones = np.ones((4, 1), dtype=np.float64)
            pts = np.hstack([xs, ones])
            pts_t = (H_np @ pts.T).T
            valid = np.abs(pts_t[:, 2]) > 1e-12
            if valid.all():
                pts_xy = pts_t[:, :2] / pts_t[:, 2:3]
                pts_int = np.round(pts_xy).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)

    if output is not None:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        cv2.imwrite(str(output), out[..., ::-1])
    return out


def ensure_prediction_fields(pred, data=None, device=None):
    """
    Ensure all required prediction fields exist with default values.
    
    Args:
        pred (dict): Prediction dictionary
        data (dict, optional): Input data dictionary to get image dimensions
        device (torch.device, optional): Device to place tensors on
    
    Returns:
        dict: Updated prediction dictionary with all required fields
    """
    if device is None:
        if data is not None:
            if 'image0' in data:
                device = data['image0'].device
            elif 'overlap_image0' in data:
                device = data['overlap_image0'].device
            else:
                device = torch.device('cpu')
        else:
            # Try to infer device from existing tensors in pred
            for v in pred.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
            else:
                device = torch.device('cpu')
    
    # Get image dimensions for bbox defaults
    if data is not None and 'image0' in data:
        h0, w0 = data['image0'].shape[2], data['image0'].shape[3]
        h1, w1 = data['image1'].shape[2], data['image1'].shape[3]
    else:
        # Use sensible defaults if no data available
        h0, w0, h1, w1 = 480, 640, 480, 640
    
    # Ensure bbox0 exists
    if 'bbox0' not in pred:
        pred['bbox0'] = torch.tensor([[0.0, 0.0, w0, h0]], device=device)
    
    # Ensure bbox1 exists
    if 'bbox1' not in pred:
        pred['bbox1'] = torch.tensor([[0.0, 0.0, w1, h1]], device=device)
    
    # Ensure ratio0 exists
    if 'ratio0' not in pred:
        pred['ratio0'] = torch.tensor([[1.0, 1.0]], device=device)
    
    # Ensure ratio1 exists
    if 'ratio1' not in pred:
        pred['ratio1'] = torch.tensor([[1.0, 1.0]], device=device)
    
    # Ensure overlap_time exists
    if 'overlap_time' not in pred:
        pred['overlap_time'] = torch.tensor([0.0], device=device)
    
    return pred

def visualize_box_mask_constraint_pair(
        image0,
        image1,
        bbox0,
        bbox1,
        mask0,
        mask1,
        output_path,
        thresh=0.5,
        margin=24,
        alpha_green=0.40,
):
    """
    Visualize mask+bbox with a style consistent with `viz` mode:
      1) keep magenta mask visualization,
      2) keep original brightness outside mask,
      3) highlight constrained mask inside bbox in green with contour.
    """
    def _to_mask_hw(mask, h, w):
        if mask is None:
            return None
        if torch.is_tensor(mask):
            m = mask.detach().cpu().float().squeeze().numpy()
        else:
            m = np.asarray(mask).squeeze().astype(np.float32)
        if m.ndim != 2:
            return None
        if m.max() > 1.0:
            m = m / 255.0
        m = np.clip(m, 0.0, 1.0)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return m

    def _to_box_xyxy(box, h, w):
        if box is None:
            return None
        if torch.is_tensor(box):
            b = box.detach().cpu().numpy().reshape(-1)[:4]
        else:
            b = np.asarray(box).reshape(-1)[:4]
        x0, y0, x1, y1 = [int(round(float(v))) for v in b]
        x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w, x1))
        y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _render_one(img_in, mask_in, box_in):
        img = _to_hwc_uint8(img_in)
        h, w = img.shape[:2]
        m = _to_mask_hw(mask_in, h, w)
        box = _to_box_xyxy(box_in, h, w)

        out_u8 = img.copy()

        # 1) Global mask: magenta overlay (do NOT darken non-mask area)
        if m is not None:
            m_soft = np.clip(m, 0.0, 1.0).astype(np.float32)
            alpha_magenta = 0.35
            out_f = out_u8.astype(np.float32) / 255.0
            magenta = np.array([1.0, 0.0, 1.0], dtype=np.float32)
            a = (alpha_magenta * m_soft)[..., None]
            out_f = (1.0 - a) * out_f + a * magenta
            out_u8 = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)

            # magenta contour
            m_bin = (m_soft > float(thresh)).astype(np.uint8)
            if m_bin.sum() > 0:
                contours, _ = cv2.findContours(m_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out_u8, contours, -1, (255, 0, 255), 2, lineType=cv2.LINE_AA)

        # 2) BBox style: keep existing bbox style
        if box is not None:
            out_u8 = draw_bbox(out_u8, box)

        # 3) Green constrained-mask inside bbox + green contour
        if m is not None and box is not None:
            x0, y0, x1, y1 = box
            roi_bin = (m[y0:y1, x0:x1] > float(thresh)).astype(np.uint8)
            if roi_bin.size > 0 and roi_bin.sum() > 0:
                out_f = out_u8.astype(np.float32) / 255.0
                sub = out_f[y0:y1, x0:x1]
                green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                sub = (1.0 - alpha_green * roi_bin[..., None]) * sub + (alpha_green * roi_bin[..., None]) * green
                out_f[y0:y1, x0:x1] = sub
                out_u8 = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)

                contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    c_shift = c + np.array([[[x0, y0]]], dtype=c.dtype)
                    cv2.drawContours(out_u8, [c_shift], -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        return out_u8

    vis0 = _render_one(image0, mask0, bbox0)
    vis1 = _render_one(image1, mask1, bbox1)

    h = max(vis0.shape[0], vis1.shape[0])
    w = vis0.shape[1] + vis1.shape[1] + margin
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:vis0.shape[0], :vis0.shape[1]] = vis0
    canvas[:vis1.shape[0], vis0.shape[1] + margin:vis0.shape[1] + margin + vis1.shape[1]] = vis1

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(str(output_path), canvas[..., ::-1])
    return canvas
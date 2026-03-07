import cv2
import math
import torch
import inspect
import functools
import numpy as np
from typing import Dict, List, NamedTuple, Optional, Tuple, Union


# --- GEOMETRY ---
def estimate_pose(kpts0, kpts1, K0, K1, ransac, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    if ransac:
        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=norm_thresh,
            prob=conf,
            method=cv2.RANSAC)
    else:
        E, mask = cv2.findFundamentalMat(
            kpts0, kpts1, method=cv2.FM_8POINT
        )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(
                _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
                      + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


"""
Convenience classes for an SE3 pose and a pinhole Camera with lens distortion.
Based on PyTorch tensors: differentiable, batched, with GPU support.
"""


def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def so3exp_map(w, eps: float = 1e-7):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    small = theta < eps
    div = torch.where(small, torch.ones_like(theta), theta)
    W = skew_symmetric(w / div)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res)  # first-order Taylor approx
    return torch.eye(3).to(W) + res


def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3)."""
    z = torch.zeros_like(v[..., 0])
    M = torch.stack(
        [
            z,
            -v[..., 2],
            v[..., 1],
            v[..., 2],
            z,
            -v[..., 0],
            -v[..., 1],
            v[..., 0],
            z,
        ],
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))
    return M


@torch.jit.script
def distort_points(pts, dist):
    """Distort normalized 2D coordinates
    and check for validity of the distortion model.
    """
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]
    undist = pts
    valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts ** 2, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2 ** 2
        undist = undist + pts * radial

        # The distortion model is supposedly only valid within the image
        # boundaries. Because of the negative radial distortion, points that
        # are far outside of the boundaries might actually be mapped back
        # within the image. To account for this, we discard points that are
        # beyond the inflection point of the distortion model,
        # e.g. such that d(r + k_1 r^3 + k2 r^5)/dr = 0
        limited = ((k2 > 0) & ((9 * k1 ** 2 - 20 * k2) > 0)) | ((k2 <= 0) & (k1 > 0))
        limit = torch.abs(
            torch.where(
                k2 > 0,
                (torch.sqrt(9 * k1 ** 2 - 20 * k2) - 3 * k1) / (10 * k2),
                1 / (3 * k1),
            )
        )
        valid = valid & torch.squeeze(~limited | (r2 < limit), -1)

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            uv = torch.prod(pts, -1, keepdim=True)
            undist = undist + 2 * p12 * uv + p21 * (r2 + 2 * pts ** 2)
            # TODO: handle tangential boundaries

    return undist, valid


@torch.jit.script
def J_distort_points(pts, dist):
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]

    J_diag = torch.ones_like(pts)
    J_cross = torch.zeros_like(pts)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts ** 2, -1, keepdim=True)
        uv = torch.prod(pts, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2 ** 2
        d_radial = 2 * k1 + 4 * k2 * r2
        J_diag += radial + (pts ** 2) * d_radial
        J_cross += uv * d_radial

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            J_diag += 2 * p12 * pts.flip(-1) + 6 * p21 * pts
            J_cross += 2 * p12 * pts + 2 * p21 * pts.flip(-1)

    J = torch.diag_embed(J_diag) + torch.diag_embed(J_cross).flip(-1)
    return J


def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
    if they are numpy arrays. Use the device and dtype of the wrapper.
    """

    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device("cpu")
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


class TensorWrapper:
    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape[:-1]

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self._data.cpu())

    def cuda(self):
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        return self.__class__(self._data.pin_memory())

    def float(self):
        return self.__class__(self._data.float())

    def double(self):
        return self.__class__(self._data.double())

    def detach(self):
        return self.__class__(self._data.detach())

    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return self.stack(*args, **kwargs)
        else:
            return NotImplemented


class Pose(TensorWrapper):
    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 12
        super().__init__(data)

    @classmethod
    @autocast
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        """Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        """
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    @classmethod
    @autocast
    def from_aa(cls, aa: torch.Tensor, t: torch.Tensor):
        """Pose from an axis-angle rotation vector and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            aa: axis-angle rotation vector with shape (..., 3).
            t: translation vector with shape (..., 3).
        """
        assert aa.shape[-1] == 3
        assert t.shape[-1] == 3
        assert aa.shape[:-1] == t.shape[:-1]
        return cls.from_Rt(so3exp_map(aa), t)

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor):
        """Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        """
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)

    @classmethod
    def from_colmap(cls, image: NamedTuple):
        """Pose from a COLMAP Image."""
        return cls.from_Rt(image.qvec2rotmat(), image.tvec)

    @property
    def R(self) -> torch.Tensor:
        """Underlying rotation matrix with shape (..., 3, 3)."""
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def t(self) -> torch.Tensor:
        """Underlying translation vector with shape (..., 3)."""
        return self._data[..., -3:]

    def inv(self) -> "Pose":
        """Invert an SE(3) pose."""
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: "Pose") -> "Pose":
        """Chain two SE(3) poses: T_B2C.compose(T_A2B) -> T_A2C."""
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    @autocast
    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points.
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        """
        assert p3d.shape[-1] == 3
        # assert p3d.shape[:-2] == self.shape  # allow broadcasting
        return p3d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)

    def __mul__(self, p3D: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points: T_A2B * p3D_A -> p3D_B."""
        return self.transform(p3D)

    def __matmul__(
            self, other: Union["Pose", torch.Tensor]
    ) -> Union["Pose", torch.Tensor]:
        """Transform a set of 3D points: T_A2B * p3D_A -> p3D_B.
        or chain two SE(3) poses: T_B2C @ T_A2B -> T_A2C."""
        if isinstance(other, self.__class__):
            return self.compose(other)
        else:
            return self.transform(other)

    @autocast
    def J_transform(self, p3d_out: torch.Tensor):
        # [[1,0,0,0,-pz,py],
        #  [0,1,0,pz,0,-px],
        #  [0,0,1,-py,px,0]]
        J_t = torch.diag_embed(torch.ones_like(p3d_out))
        J_rot = -skew_symmetric(p3d_out)
        J = torch.cat([J_t, J_rot], dim=-1)
        return J  # N x 3 x 6

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.t.numpy()

    def magnitude(self) -> Tuple[torch.Tensor]:
        """Magnitude of the SE(3) transformation.
        Returns:
            dr: rotation anngle in degrees.
            dt: translation distance in meters.
        """
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        cos = torch.clamp((trace - 1) / 2, -1, 1)
        dr = torch.acos(cos).abs() / math.pi * 180
        dt = torch.norm(self.t, dim=-1)
        return dr, dt

    def __repr__(self):
        return f"Pose: {self.shape} {self.dtype} {self.device}"


class Camera(TensorWrapper):
    eps = 1e-4

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] in {6, 8, 10}
        super().__init__(data)

    @classmethod
    def from_colmap(cls, camera: Union[Dict, NamedTuple]):
        """Camera from a COLMAP Camera tuple or dictionary.
        We use the corner-convetion from COLMAP (center of top left pixel is (0.5, 0.5))
        """
        if isinstance(camera, tuple):
            camera = camera._asdict()

        model = camera["model"]
        params = camera["params"]

        if model in ["OPENCV", "PINHOLE", "RADIAL"]:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if model == "SIMPLE_RADIAL":
                params = np.r_[params, 0.0]
        else:
            raise NotImplementedError(model)

        data = np.r_[camera["width"], camera["height"], fx, fy, cx, cy, params]
        return cls(data)

    @classmethod
    @autocast
    def from_calibration_matrix(cls, K: torch.Tensor):
        cx, cy = K[..., 0, 2], K[..., 1, 2]
        fx, fy = K[..., 0, 0], K[..., 1, 1]
        data = torch.stack([2 * cx, 2 * cy, fx, fy, cx, cy], -1)
        return cls(data)

    @autocast
    def calibration_matrix(self):
        K = torch.zeros(
            *self._data.shape[:-1],
            3,
            3,
            device=self._data.device,
            dtype=self._data.dtype,
        )
        K[..., 0, 2] = self._data[..., 4]
        K[..., 1, 2] = self._data[..., 5]
        K[..., 0, 0] = self._data[..., 2]
        K[..., 1, 1] = self._data[..., 3]
        K[..., 2, 2] = 1.0
        return K

    @property
    def size(self) -> torch.Tensor:
        """Size (width height) of the images, with shape (..., 2)."""
        return self._data[..., :2]

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self._data[..., 2:4]

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self._data[..., 4:6]

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., {0, 2, 4})."""
        return self._data[..., 6:]

    @autocast
    def scale(self, scales: torch.Tensor):
        """Update the camera parameters after resizing an image."""
        s = scales
        data = torch.cat([self.size * s, self.f * s, self.c * s, self.dist], -1)
        return self.__class__(data)

    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        """Update the camera parameters after cropping an image."""
        left_top = self._data.new_tensor(left_top)
        size = self._data.new_tensor(size)
        data = torch.cat([size, self.f, self.c - left_top, self.dist], -1)
        return self.__class__(data)

    @autocast
    def in_image(self, p2d: torch.Tensor):
        """Check if 2D points are within the image boundaries."""
        assert p2d.shape[-1] == 2
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
        return valid

    @autocast
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Project 3D points into the camera plane and check for visibility."""
        z = p3d[..., -1]
        valid = z > self.eps
        z = z.clamp(min=self.eps)
        p2d = p3d[..., :-1] / z.unsqueeze(-1)
        return p2d, valid

    def J_project(self, p3d: torch.Tensor):
        x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
        zero = torch.zeros_like(z)
        z = z.clamp(min=self.eps)
        J = torch.stack([1 / z, zero, -x / z ** 2, zero, 1 / z, -y / z ** 2], dim=-1)
        J = J.reshape(p3d.shape[:-1] + (2, 3))
        return J  # N x 2 x 3

    @autocast
    def distort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates
        and check for validity of the distortion model.
        """
        assert pts.shape[-1] == 2
        # assert pts.shape[:-2] == self.shape  # allow broadcasting
        return distort_points(pts, self.dist)

    def J_distort(self, pts: torch.Tensor):
        return J_distort_points(pts, self.dist)  # N x 2 x 2

    @autocast
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)

    @autocast
    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return (p2d - self.c.unsqueeze(-2)) / self.f.unsqueeze(-2)

    def J_denormalize(self):
        return torch.diag_embed(self.f).unsqueeze(-3)  # 1 x 2 x 2

    @autocast
    def cam2image(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Transform 3D points into 2D pixel coordinates."""
        p2d, visible = self.project(p3d)
        p2d, mask = self.distort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid

    def J_world2image(self, p3d: torch.Tensor):
        p2d_dist, valid = self.project(p3d)
        J = self.J_denormalize() @ self.J_distort(p2d_dist) @ self.J_project(p3d)
        return J, valid

    @autocast
    def image2cam(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert 2D pixel corrdinates to 3D points with z=1"""
        assert self._data.shape
        p2d = self.normalize(p2d)
        # iterative undistortion
        return to_homogeneous(p2d)

    def to_cameradict(self, camera_model: Optional[str] = None) -> List[Dict]:
        data = self._data.clone()
        if data.dim() == 1:
            data = data.unsqueeze(0)
        assert data.dim() == 2
        b, d = data.shape
        if camera_model is None:
            camera_model = {6: "PINHOLE", 8: "RADIAL", 10: "OPENCV"}[d]
        cameras = []
        for i in range(b):
            if camera_model.startswith("SIMPLE_"):
                params = [x.item() for x in data[i, 3: min(d, 7)]]
            else:
                params = [x.item() for x in data[i, 2:]]
            cameras.append(
                {
                    "model": camera_model,
                    "width": int(data[i, 0].item()),
                    "height": int(data[i, 1].item()),
                    "params": params,
                }
            )
        return cameras if self._data.dim() == 2 else cameras[0]

    def __repr__(self):
        return f"Camera {self.shape} {self.dtype} {self.device}"

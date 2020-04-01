import cv2
import math
import numpy as np

from utils.image_utils import polygons_to_mask


def rotate_point(M, point):
    """Rotating point using perspective matrix M"""
    x = M[0, 0] * point[0] + M[0, 1] * point[1] + M[0, 2]
    y = M[1, 0] * point[0] + M[1, 1] * point[1] + M[1, 2]
    return [int(x), int(y)]


def affine_transform(image, pts, padding=(2, 2, 2, 2)):
    left = max(0, int(min(pts[:, 0])) - padding[0])
    right = min(int(max(pts[:, 0])) + padding[1], image.shape[1] - 1)
    top = max(0, int(min(pts[:, 1])) - padding[2])
    below = min(int(max(pts[:, 1])) + padding[3], image.shape[0] - 1)

    h, w, c = image.shape
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    tl[0] = max(0, tl[0] - padding[0])
    tl[1] = max(0, tl[1] - padding[2])

    tr[0] = min(w - 1, tr[0] + padding[1])
    tr[1] = max(0, tr[1] - padding[2])

    br[0] = min(w - 1, br[0] + padding[1])
    br[1] = min(h - 1, br[1] + padding[3])

    bl[0] = max(0, bl[0] - padding[0])
    bl[1] = min(h - 1, bl[1] + padding[3])

    new_box = np.array([tl, tr, br, bl])

    # crop rectangle image
    cropped_image = image[top: below, left: right, :]

    cropped_h, cropped_w, cropped_c = cropped_image.shape

    angle = np.arctan2(tr[1] - tl[1], tr[0] - tl[0])
    angle = angle * 180 / np.pi

    if angle == 0:
        return cropped_image, new_box

    center_x, center_y = cropped_w / 2., cropped_h / 2.

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(heightA), int(heightB))

    radians = math.radians(angle)
    sin, cos = math.sin(radians), math.cos(radians)
    abs_sin, abs_cos = abs(sin), abs(cos)

    # Calculate new width and height
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Calculate perspective matrix
    M = np.array([
        [cos, sin, new_width / 2 - center_x + (1 - cos) * center_x - sin * center_y],
        [-sin, cos, new_height / 2 - center_y + sin * center_x + (1 - cos) * center_y]
    ])

    # Get new rotated image
    rotated_img = cv2.warpAffine(
        src=cropped_image,
        M=M,
        dsize=(new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    rotated_points = [rotate_point(M, point) for point in new_box - np.array([[left, top]])]
    mask = polygons_to_mask([np.array(rotated_points, np.float32)], new_height, new_width)
    x, y, w, h = cv2.boundingRect(mask)

    mask = np.expand_dims(np.float32(mask), axis=-1)
    rotated_image = rotated_img * mask
    crop_image = rotated_image[y:y + h, x:x + w, :]

    return crop_image, new_box


def perspective_trasform(image, pts, padding=(2, 2, 2, 2)):
    h, w, c = image.shape
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    tl[0] = max(0, tl[0] - padding[0])
    tl[1] = max(0, tl[1] - padding[2])

    tr[0] = min(w - 1, tr[0] + padding[1])
    tr[1] = max(0, tr[1] - padding[2])

    br[0] = min(w - 1, br[0] + padding[1])
    br[1] = min(h - 1, br[1] + padding[3])

    bl[0] = max(0, bl[0] - padding[0])
    bl[1] = min(h - 1, bl[1] + padding[3])

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    dst = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return dst, np.array([tl, tr, br, bl])


def np_thin_palte_spline_transform(image, pts):
    def _repeat(x, n_repeats):
        rep = np.transpose(
            np.expand_dims(np.ones([n_repeats]), 1),
            [1, 0]
        )
        rep = rep.astype(np.int32)
        x = np.matmul(x.reshape([-1, 1]), rep)

        return x.reshape((-1))

    def _interpolate(im, x, y, out_size):
        num_batch, height, width, channels = im.shape

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        height_f = float(height)
        width_f = float(width)

        out_height = out_size[0]
        out_width = out_size[1]

        zero = np.zeros([], dtype=np.int32)
        max_y = im.shape[1] - 1
        max_x = im.shape[2] - 1

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * width_f / 2.0
        y = (y + 1.0) * height_f / 2.0

        x0 = np.floor(x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int32)
        y1 = y0 + 1

        x0 = np.clip(x0, zero, max_x)
        x1 = np.clip(x1, zero, max_x)
        y0 = np.clip(y0, zero, max_y)
        y1 = np.clip(y1, zero, max_y)

        dim2 = width
        dim1 = width * height

        base = _repeat(np.arange(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore channels dim
        im_flat = im.reshape((-1, channels))
        im_flat = im_flat.astype(np.float32)
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # and finally calculate interpolated values
        x0_f = x0.astype(np.float32)
        x1_f = x1.astype(np.float32)
        y0_f = y0.astype(np.float32)
        y1_f = y1.astype(np.float32)
        wa = np.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = np.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = np.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = np.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _meshgrid(height, width, source):
        x_t = np.tile(
            np.reshape(np.linspace(-1.0, 1.0, width), [1, width]),
            [height, 1]
        )
        y_t = np.tile(
            np.reshape(np.linspace(-1.0, 1.0, height), [height, 1]),
            [1, width]
        )

        x_t_flat = np.reshape(x_t, (1, 1, -1))
        y_t_flat = np.reshape(y_t, (1, 1, -1))

        num_batch = source.shape[0]
        px = np.expand_dims(source[:, :, 0], 2)  # [bn, pn, 1]
        py = np.expand_dims(source[:, :, 1], 2)  # [bn, pn, 1]
        d2 = np.square(x_t_flat - px) + np.square(y_t_flat - py)
        r = d2 * np.log(d2 + 1e-6)  # [bn, pn, h*w]
        x_t_flat_g = np.tile(x_t_flat, [num_batch, 1, 1])  # [bn, 1, h*w]
        y_t_flat_g = np.tile(y_t_flat, [num_batch, 1, 1])  # [bn, 1, h*w]
        ones = np.ones_like(x_t_flat_g)  # [bn, 1, h*w]

        grid = np.concatenate([ones, x_t_flat_g, y_t_flat_g, r], 1)  # [bn, 3+pn, h*w]

        return grid

    def _transform(T, source, input_dim, out_size):
        num_batch, height, width, num_channels = input_dim.shape

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width, source)  # [2, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = np.matmul(T, grid)  #
        x_s = T_g[:, 0:1, :]
        y_s = T_g[:, 1:2, :]
        # x_s = np.slice(T_g, [0, 0, 0], [-1, 1, -1])
        # y_s = np.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = np.reshape(x_s, [-1])
        y_s_flat = np.reshape(y_s, [-1])

        input_transformed = _interpolate(
            input_dim, x_s_flat, y_s_flat, out_size
        )

        output = np.reshape(
            input_transformed,
            [num_batch, out_height, out_width, num_channels]
        )

        return output

    def _solve_system(source, target):
        num_batch = source.shape[0]
        num_point = source.shape[1]

        ones = np.ones([num_batch, num_point, 1], dtype="float32")
        p = np.concatenate([ones, source], 2)  # [bn, pn, 3]

        p_1 = np.reshape(p, [num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = np.reshape(p, [num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = np.sum(np.square(p_1 - p_2), 3)  # [bn, pn, pn]
        r = d2 * np.log(d2 + 1e-6)  # [bn, pn, pn]

        zeros = np.zeros([num_batch, 3, 3], dtype="float32")
        W_0 = np.concatenate([p, r], 2)  # [bn, pn, 3+pn]
        W_1 = np.concatenate([zeros, np.transpose(p, [0, 2, 1])], 2)  # [bn, 3, pn+3]
        W = np.concatenate([W_0, W_1], 1)  # [bn, pn+3, pn+3]
        W_inv = np.linalg.inv(W)

        tp = np.pad(
            target,
            [[0, 0], [0, 3], [0, 0]],
            "constant"
        )  # [bn, pn+3, 2]
        T = np.matmul(W_inv, tp)  # [bn, pn+3, 2]
        T = np.transpose(T, [0, 2, 1])  # [bn, 2, pn+3]

        return T

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    source = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
    ])
    source[:, 0] = ((source[:, 0] * 2) / maxWidth) - 1.
    source[:, 1] = ((source[:, 1] * 2) / maxHeight) - 1.

    target = rect
    target[:, 0] = ((target[:, 0] * 2) / maxWidth) - 1.
    target[:, 1] = ((target[:, 1] * 2) / maxHeight) - 1.

    source = source.reshape([1, -1, 2])
    target = target.reshape([1, -1, 2])

    T = _solve_system(source, target)
    output = _transform(T, source, image, [maxHeight, maxWidth, 3])

    return output[0]

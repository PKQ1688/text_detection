import cv2
import numpy as np
import pycocotools.mask as cocomask

from PIL import Image
from skimage import measure
from shapely.geometry import Polygon


def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)

    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def draw_boxes(im, bboxes, color=(0, 0, 0)):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w, = im.shape[:2]
    thick = int((h + w) / 300)
    i = 0
    for box in bboxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
        cx = np.mean([x1, x2, x3, x4])
        cy = np.mean([y1, y2, y3, y4])
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)), c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)), c, 1, lineType=cv2.LINE_AA)
        mess = str(i)
        cv2.putText(tmp, mess, (int(cx), int(cy)), 0, 1e-3 * h, c, thick // 2)
        i += 1
    return tmp


def draw_lines(im, bboxes, color=(0, 0, 0), lineW=3):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w = im.shape[:2]
    i = 0
    for box in bboxes:
        x1, y1, x2, y2 = box
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, lineW, lineType=cv2.LINE_AA)
        i += 1
    return tmp


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)
    fx = new_w / image_w
    fy = new_h / image_h
    dx = (w - new_w) // 2
    dy = (h - new_h) // 2

    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, (dx, dy))
    return boxed_image, fx, fy, dx, dy


def exp(x):
    x = np.clip(x, -6, 6)
    y = 1 / (1 + np.exp(-x))
    return y


def minAreaLine(coords):
    """

    """
    rect = cv2.minAreaRect(coords[:, ::-1])
    box = cv2.boxPoints(rect)
    box = box.reshape((8,)).tolist()

    box = sort_box(box)
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    degree, w, h, cx, cy = solve(box)
    if w < h:
        xmin = (x1 + x2) / 2
        xmax = (x3 + x4) / 2
        ymin = (y1 + y2) / 2
        ymax = (y3 + y4) / 2

    else:
        xmin = (x1 + x4) / 2
        xmax = (x2 + x3) / 2
        ymin = (y1 + y4) / 2
        ymax = (y2 + y3) / 2

    return [xmin, ymin, xmax, ymax]


def minAreaRectBox(coords):
    """
    多边形外接矩形
    """
    rect = cv2.minAreaRect(coords[:, ::-1])
    box = cv2.boxPoints(rect)
    box = box.reshape((8,)).tolist()
    box = sort_box(box)
    return box


def sort_box(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    pts = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    pts = np.array(pts, dtype="float32")
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = _order_points(pts)
    return x1, y1, x2, y2, x3, y3, x4, y4


from scipy.spatial import distance as dist


def _order_points(pts):
    # 根据x坐标对点进行排序
    """
    --------------------- 
    作者：Tong_T 
    来源：CSDN 
    原文：https://blog.csdn.net/Tong_T/article/details/81907132 
    版权声明：本文为博主原创文章，转载请附上博文链接！
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # 从排序中获取最左侧和最右侧的点
    # x坐标点
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # 现在，根据它们的y坐标对最左边的坐标进行排序，这样我们就可以分别抓住左上角和左下角
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # 现在我们有了左上角坐标，用它作为锚来计算左上角和右上角之间的欧氏距离;
    # 根据毕达哥拉斯定理，距离最大的点将是我们的右下角
    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    # 返回左上角，右上角，右下角和左下角的坐标
    return np.array([tl, tr, br, bl], dtype="float32")


def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


#####################直线处理#####################

def fit_line(p1, p2):
    """A = Y2 - Y1
       B = X1 - X2
       C = X2*Y1 - X1*Y2
       AX+BY+C=0
    直线一般方程
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def line_point_line(point1, point2):
    """
    A1x+B1y+C1=0 
    A2x+B2y+C2=0
    x = (B1*C2-B2*C1)/(A1*B2-A2*B1)
    y = (A2*C1-A1*C2)/(A1*B2-A2*B1)
    求解两条直线的交点
    """
    A1, B1, C1 = fit_line(point1[0], point1[1])
    A2, B2, C2 = fit_line(point2[0], point2[1])
    x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
    y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
    return x, y


def sqrt(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_points(p, points, alpha=10):
    ##点到点之间的距离 
    sqList = [sqrt(p, point) for point in points]
    if max(sqList) < alpha:
        return True
    else:
        return False


def point_line_cor(p, A, B, C):
    ##判断点与之间的位置关系
    # 一般式直线方程(Ax+By+c)=0
    x, y = p
    r = A * x + B * y + C
    return r


def line_to_line(points1, points2, alpha=10):
    """
    线段之间的距离
    """
    x1, y1, x2, y2 = points1
    ox1, oy1, ox2, oy2 = points2
    A1, B1, C1 = fit_line((x1, y1), (x2, y2))
    A2, B2, C2 = fit_line((ox1, oy1), (ox2, oy2))
    flag1 = point_line_cor([x1, y1], A2, B2, C2)
    flag2 = point_line_cor([x2, y2], A2, B2, C2)

    if (flag1 > 0 and flag2 > 0) or (flag1 < 0 and flag2 < 0):

        x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
        y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
        p = (x, y)
        r0 = sqrt(p, (x1, y1))
        r1 = sqrt(p, (x2, y2))

        if min(r0, r1) < alpha:

            if r0 < r1:
                points1 = [p[0], p[1], x2, y2]
            else:
                points1 = [x1, y1, p[0], p[1]]

    return points1


def get_table_rowcols(rows, cols, prob, fx, fy, size=(512, 512), row=100, col=100):
    labels = measure.label(rows > prob, connectivity=2)
    regions = measure.regionprops(labels)
    RowsLines = [minAreaLine(line.coords) for line in regions if line.bbox[3] - line.bbox[1] > row]

    labels = measure.label(cols > prob, connectivity=2)
    regions = measure.regionprops(labels)
    ColsLines = [minAreaLine(line.coords) for line in regions if line.bbox[2] - line.bbox[0] > col]

    tmp = np.zeros(size[::-1], dtype='uint8')
    tmp = draw_lines(tmp, ColsLines + RowsLines, color=255, lineW=1)
    labels = measure.label(tmp > 0, connectivity=2)
    regions = measure.regionprops(labels)

    for region in regions:
        ymin, xmin, ymax, xmax = region.bbox
        label = region.label
        if ymax - ymin < 20 or xmax - xmin < 20:
            labels[labels == label] = 0
    labels = measure.label(labels > 0, connectivity=2)

    indY, indX = np.where(labels > 0)
    xmin, xmax = indX.min(), indX.max()
    ymin, ymax = indY.min(), indY.max()
    RowsLines = [p for p in RowsLines if
                 xmin <= p[0] <= xmax and xmin <= p[2] <= xmax and ymin <= p[1] <= ymax and ymin <= p[3] <= ymax]
    ColsLines = [p for p in ColsLines if
                 xmin <= p[0] <= xmax and xmin <= p[2] <= xmax and ymin <= p[1] <= ymax and ymin <= p[3] <= ymax]
    RowsLines = [[box[0] / fx, box[1] / fy, box[2] / fx, box[3] / fy] for box in RowsLines]
    ColsLines = [[box[0] / fx, box[1] / fy, box[2] / fx, box[3] / fy] for box in ColsLines]
    return RowsLines, ColsLines


def adjust_lines(RowsLines, ColsLines, alph=50):
    ##调整line

    nrow = len(RowsLines)
    ncol = len(ColsLines)
    newRowsLines = []
    newColsLines = []
    for i in range(nrow):

        x1, y1, x2, y2 = RowsLines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(nrow):
            if i != j:
                x3, y3, x4, y4 = RowsLines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    if r < alph:
                        newRowsLines.append([x1, y1, x3, y3])
                    r = sqrt((x1, y1), (x4, y4))
                    if r < alph:
                        newRowsLines.append([x1, y1, x4, y4])

                    r = sqrt((x2, y2), (x3, y3))
                    if r < alph:
                        newRowsLines.append([x2, y2, x3, y3])
                    r = sqrt((x2, y2), (x4, y4))
                    if r < alph:
                        newRowsLines.append([x2, y2, x4, y4])

    for i in range(ncol):
        x1, y1, x2, y2 = ColsLines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(ncol):
            if i != j:
                x3, y3, x4, y4 = ColsLines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    if r < alph:
                        newColsLines.append([x1, y1, x3, y3])
                    r = sqrt((x1, y1), (x4, y4))
                    if r < alph:
                        newColsLines.append([x1, y1, x4, y4])

                    r = sqrt((x2, y2), (x3, y3))
                    if r < alph:
                        newColsLines.append([x2, y2, x3, y3])
                    r = sqrt((x2, y2), (x4, y4))
                    if r < alph:
                        newColsLines.append([x2, y2, x4, y4])

    return newRowsLines, newColsLines


def get_table_ceilboxes(ow, oh, rows, cols, prob, fx, fy, size=(512, 512), row=100, col=100, alph=50):
    """
    获取单元格
    """
    w, h = size
    RowsLines, ColsLines = get_table_rowcols(rows=rows, cols=cols, prob=prob, fx=fx, fy=fy, size=size, row=row, col=col)
    newRowsLines, newColsLines = adjust_lines(RowsLines, ColsLines, alph=alph)
    RowsLines = newRowsLines + RowsLines
    ColsLines = ColsLines + newColsLines

    nrow = len(RowsLines)
    ncol = len(ColsLines)

    for i in range(nrow):
        for j in range(ncol):
            RowsLines[i] = line_to_line(RowsLines[i], ColsLines[j], 32)
            ColsLines[j] = line_to_line(ColsLines[j], RowsLines[i], 32)

    tmp = np.zeros((oh, ow), dtype='uint8')
    tmp = draw_lines(tmp, ColsLines + RowsLines, color=255, lineW=1)

    tabelLabels = measure.label(tmp == 0, connectivity=2)
    regions = measure.regionprops(tabelLabels)
    rboxes = []
    for region in regions:
        if region.bbox_area < h * w - 10:
            rbox = minAreaRectBox(region.coords)
            rboxes.append(rbox)

    return rboxes, ColsLines, RowsLines


def polygons_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.
    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.
    Returns:
        a binary mask matrix of shape (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def quadrangle_distance(quadrangle_a, quadrangle_b):
    """
    四边形iou
    :param quadrangle_a: 2维numpy数组[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    :param quadrangle_b: 2维numpy数组[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    :return:
    """
    # print(quadrangle_a, quadrangle_b)
    a = Polygon(quadrangle_a)
    b = Polygon(quadrangle_b)
    if a.intersection(b).area == min(a.area, b.area):
        return 0
    if np.mean(quadrangle_a[:, 0]) < np.mean(quadrangle_b[:, 0]):
        x_a = np.max(quadrangle_a[:, 0])
        x_b = np.min(quadrangle_b[:, 0])
        dis = x_b - x_a
    else:
        x_a = np.min(quadrangle_a[:, 0])
        x_b = np.max(quadrangle_b[:, 0])
        dis = x_a - x_b

    h_a = np.max(quadrangle_a[:, 1]) - np.min(quadrangle_a[:, 1])
    h_b = np.max(quadrangle_b[:, 1]) - np.min(quadrangle_b[:, 1])
    sim = min(h_a, h_b) / max(h_a, h_b)
    max_y1 = max(min(quadrangle_a[0][1], quadrangle_a[1][1]), min(quadrangle_b[0][1], quadrangle_b[1][1]))
    min_y2 = min(max(quadrangle_a[2][1], quadrangle_a[3][1]), max(quadrangle_b[2][1], quadrangle_b[3][1]))
    overlaps = max(0, min_y2 - max_y1) / min(h_a, h_b)
    # print(min_y2, max_y1, h_a, h_b, sim, overlaps)
    if sim > 0.7 and overlaps > 0.8:
        return dis
    else:
        return 100000


def _connect_quadrangles_by_distance(quadrangle, quadrangles):
    """
    使用最小外接四边形将四边形合并
    quadrangle: 2D vector [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    quadrangles: [boxes_count, (x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    """
    res = [quadrangle]
    for quad in quadrangles:
        res.append(quad)
    res = np.array(res)
    res = [[np.min(res[:, :, 0]), np.min(res[:, :, 1])],
           [np.max(res[:, :, 0]), np.min(res[:, :, 1])],
           [np.max(res[:, :, 0]), np.max(res[:, :, 1])],
           [np.min(res[:, :, 0]), np.max(res[:, :, 1])]]
    return np.array(res)


def connect_quadrangles_by_distance(quadrangles, scores, distance_threshold=5):
    """
    若在同一水平线且距离相近，则进行box合并
    :param quadrangles: [n,(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    :param scores: [n]
    :param iou_threshold:
    :return:
    """
    if len(quadrangles) == 0:
        return quadrangles, scores

    ixs = np.argsort(quadrangles[:, 0, 1])[::-1]
    pick = []
    while ixs.size > 0:
        # 选择得分最高的
        i = ixs[0]
        # 逐个计算iou
        overlap = np.array([quadrangle_distance(quadrangles[i], quadrangles[t]) for t in ixs[1:]])
        # 连接iou超过阈值的四边形
        remove_ixs = np.where(overlap < distance_threshold)[0] + 1
        if len(remove_ixs) == 0:
            pick.append(i)
            ixs = np.delete(ixs, 0)
        else:
            quadrangles[i] = _connect_quadrangles_by_distance(quadrangles[i], quadrangles[ixs[remove_ixs]])
            scores[i] = np.max([scores[i]] + list(scores[ixs[remove_ixs]]))
            ixs = np.delete(ixs, remove_ixs)
    return quadrangles[pick], scores[pick]

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
import itertools
import cv2
import sys
import glob


def distance_and_index(idx_3p, xy_dst, xy_src, tree):
    """Compute Affine transformation based on 3 corresponding points.
    Then apply the transformation on all points and compute the distances,
    and indices to the src points."""
    matrix = cv2.getAffineTransform(xy_dst[idx_3p], xy_src)
    npoints = cv2.transform(np.array([xy_dst]), matrix)
    return tree.query(np.squeeze(npoints))


def find_labels(xy_src, labels_src, xy_dst, idx_3p_dst):
    """Find the labels corresponding to the coordinates xy_dst."""
    # build a tree for fast distance evaluation
    tree = cKDTree(xy_src)

    N_src, N_dst = len(xy_src), len(xy_dst)

    # all possible combinations
    permutations = itertools.permutations(range(N_src), 3)

    # now restrict to only counter clockwise patterns
    permutations_ccw = [p for p in permutations if
                        ccw(xy_src[p[0]], xy_src[p[1]], xy_src[p[2]]) > 0]

    dist = []
    for p in permutations_ccw:
        d, idx = distance_and_index(idx_3p_dst, xy_dst, xy_src[p, :], tree)
        dist.append([p, np.sum(d), N_dst-len(set(idx))])

    dist = np.array(dist, dtype=object)

    i = np.lexsort((dist[:, 1], dist[:, 2]))
    # take the top 5 hits
    top5 = i[:5]
    # search in top5 for best distance
    k = np.argmin(dist[top5][:, 1])

    idx = dist[top5][k, 0]
    print(dist[top5][k])

    d, idx = distance_and_index(idx_3p_dst, xy_dst, xy_src[idx, :], tree)

    dp = []
    if len(set(idx)) < len(xy_dst):
        print('double labels!')
        s = np.sort(idx)
        #de = s[:-1] - s[1:]
        dp = s[np.nonzero(s[:-1] - s[1:] == 0)]

    return [(i, j, labels_src[j]) for i, j in enumerate(idx) if j not in dp]


def ccw(a, b, c):
    """Check if 3 points are in counter clockwise direction (return >1)."""
    return np.cross(b-a, c-a)


def find_corner_points(xy):
    """Find 4 corner points."""
    idx = [i for i in range(len(xy))]
    if len(idx) < 5:
        print('not enought points for transformation')
        return None

    # top most
    i0 = np.argmax(xy[idx, 1])
    idx.remove(i0)
    # left most
    i1 = np.argmin(xy[idx, 0])
    i1 = idx[i1]
    idx.remove(i1)
    # lowest
    i2 = np.argmin(xy[idx, 1])
    i2 = idx[i2]
    idx.remove(i2)
    # right most
    i3 = np.argmax(xy[idx, 0])
    i3 = idx[i3]

    return np.array([i0, i1, i2, i3])


def read_label_positions(filename):
    """Read label positions from XML file."""
    # label the boxes
    path, objects = parse_XML(filename)

    labels_src = [k for k, v in objects]
    xy_src = .5*np.array([[v[0]+v[2], v[1]+v[3]]
                          for k, v in objects], dtype=np.float32)

    return labels_src, xy_src


def parse_XML(filename):
    """Parse XML data from file."""
    root = ET.parse(filename).getroot()
    img_path = root.find('path').text

    objects = []
    for i in root.iter('object'):
        name = i.find('name').text
        xyminmax = np.array([int(v.text) for v in
                             i.find('bndbox')], dtype=np.int64)
        objects.append((name, xyminmax))

    return img_path, objects


# get a list of labels and a list of corresponding coordinates,
# here for the right side
labels_src, xy_src = read_label_positions("DSC_2529.xml")
#print(labels_src, xy_src)

# get positions for bounding boxes from some training data,
# the input can be a list of coordinates of mid points of bounding (xy_bbox)
# boxes from the object detection network.
for f in glob.glob("DSC_*.xml"):
    print(f)
    labels_o, xy_bbox = read_label_positions(f)

    # you can randomize the coordinates any way you want,
    # it has no influence on the performance or outcome
    r = np.random.permutation(len(xy_bbox))
    xy_bbox = xy_bbox[r]
    #print(labels_o, xy_bbox)

    # find 4 corner points
    corner_p = find_corner_points(xy_bbox)
    if corner_p is None:
        print("not enough points!")
        sys.exit()

    # pick 3 points, for the right side
    corner_3_idx = corner_p[[0, 2, 3]]
    # for the left side
    #corner_3_idx = corner_p[[0, 1, 3]]

    # now find the labels for the given coordinates (xy_bbox)
    labels_dst = find_labels(xy_src, labels_src, xy_bbox, corner_3_idx)
    print(labels_dst)

    # plot known (labels, coordinates of src) and coordinates of position to find the labels for
    plt.subplot(121)
    plt.title("known coords and labels")
    for label, x in zip(labels_src, xy_src):
        plt.plot(x[0], x[1], 'o')
        plt.text(x[0], x[1], label)

    plt.subplot(122)
    plt.title("known coords found labels")
    for label, x in zip(labels_dst, xy_bbox):
        plt.plot(x[0], x[1], 'o')
        plt.text(x[0], x[1], label[2])
    plt.show()

"""."""
import numpy as np
import cv2

from scipy.ndimage import gaussian_filter
import xml.etree.ElementTree as ET
import os


# start point of tuft
lab_F = {'F1': [3271, 620],
         'F2': [3419, 852],
         'F3': [3700, 1096],
         'F4': [2885, 684],
         'F5': [3002, 1005]}

lab_A = {'A1': [3559, 2777],
         'A2': [3149, 2600],
         'A3': [3000, 2347],
         'A4': [2987, 1973],
         'A5': [3199, 1628],
         'A6': [3545, 1496],
         'A7': [3845, 1576],
         'A8': [4023, 1734],
         'A9': [4116, 1924],
         'A10': [4164, 2077],
         'A11': [4126, 2309],
         'A12': [3998, 2575],
         'A13': [4329, 2050],
         'A14': [4241, 1757],
         'A15': [4059, 1548]}

lab_B = {'B1': [3746, 2584],
         'B9': [3810, 1733]}

lab_G = {'G1': [3574, 3031],
         'G2': [3472, 3201],
         'G3': [3159, 3115],
         'G4': [3166, 2959],
         'Antenna': [2987, 3241]}


all_labels = {}
for label in [lab_A, lab_F, lab_G]:
    for k, v in label.items():
        all_labels[k] = v

#IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE,  VOL.  20,  NO.  2,  FEBRUARY  1998113 An Unbiased Detector of Curvilinear StructuresCarsten Steger

def compute_eigen(img):
    """Compute eigenvalues of Hessian."""
    lw = 3.
    sigma = lw/np.sqrt(3)
    Ixx = gaussian_filter(img, sigma, order=[0, 2], output=np.float32, mode='nearest')
    Iyy = gaussian_filter(img, sigma, order=[2, 0], output=np.float32, mode='nearest')
    Ixy = gaussian_filter(img, sigma, order=[1, 1], output=np.float32, mode='nearest')

    h, w = img.shape
    # Hessian matrices
    H = np.array([[[Ixx[i, j], Ixy[i, j]], [Ixy[i, j], Iyy[i, j]]] for i in range(h) for j in range(w)])

    # compute eigenvalues and eigenvectors
    # print('computing eigen values/vectors')
    ev, evv = np.linalg.eig(H)

    #plt.imshow(img)
    #plt.show()
    ev = ev.reshape(h, w, 2)
    evv = evv.reshape(h, w, 2, 2)

    # maximum absolute eigenvalues
    abs_ev = np.abs(ev)
    # indices of max abs eigenvalue
    #ii = np.where(abs_ev == np.max(abs_ev, axis=2)[:, :, np.newaxis]) #this fails if the max values are equal!
    ii = np.identity(abs_ev.shape[2], bool)[abs_ev.argmax(axis=2)]
    #print("ii=", ii, np.sum(ii))
    #print(abs_ev.shape, evv.shape, w, h, ii.shape, evv[ii].shape)

    max_evv = evv[ii].reshape(h, w, 2)
    max_evv[:, :, 0] *= -1

    Ix = gaussian_filter(img, sigma, order=[0, 1], output=np.float32, mode='nearest')
    Iy = gaussian_filter(img, sigma, order=[1, 0], output=np.float32, mode='nearest')

    den = (max_evv[:, :, 0]**2*Ixx + 2*max_evv[:, :, 0]*max_evv[:, :, 1]*Ixy +
           max_evv[:, :, 1]**2*Iyy)

    t = -(max_evv[:, :, 0]*Ix + max_evv[:, :, 1]*Iy) / den
    p = t[:, :, np.newaxis] * max_evv[:, :]

    return p, max_evv, ev


def angle_to_indices(angle):
    """."""
    if np.abs(angle) <= np.deg2rad(22.5):
        return np.array([[1, 1], [0, 1], [-1, 1]])
    if np.abs(angle) >= np.deg2rad(157.5):
        return np.array([[1, -1], [0, -1], [-1, -1]])
    if angle > np.deg2rad(22.5) and angle <= np.deg2rad(67.5):
        return np.array([[-1, 0], [-1, 1], [0, 1]])
    if angle > np.deg2rad(67.5) and angle <= np.deg2rad(112.5):
        return np.array([[-1, -1], [-1, 0], [-1, 1]])
    if angle > np.deg2rad(112.5) and angle < np.deg2rad(157.5):
        return np.array([[0, -1], [-1, -1], [-1, 0]])
    if angle > np.deg2rad(-157.5) and angle <= np.deg2rad(-112.5):
        return np.array([[0, -1], [1, -1], [1, 0]])
    if angle > np.deg2rad(-112.5) and angle <= np.deg2rad(-67.5):
        return np.array([[1, -1], [1, 0], [1, 1]])
    if angle > np.deg2rad(-67.5) and angle < np.deg2rad(-22.5):
        return np.array([[0, 1], [1, 1], [1, 0]])


def link(clist, angles, max_ev, ij, h, w):
    """."""
    ev_limit = 0.2*max_ev[ij]
    angle_limit = np.deg2rad(30.)

    for i in range(150):
        current_angle = angles[ij]
        #print('current angle:', ij, np.rad2deg(current_angle))
        ij += angle_to_indices(current_angle)

        if (np.any(ij[:, 0] >= h) or np.any(ij[:, 0] < 0) or
                np.any(ij[:, 1] >= w) or np.any(ij[:, 1] < 0)):
            # print('end of curve, boundary encountered')
            break

        if np.all(max_ev[ij[:, 0], ij[:, 1]] < ev_limit):
            # print('end of curve, no ev large enough')
            break

        delta = angle_diff(current_angle, angles[ij[:, 0], ij[:, 1]])

        if np.min(delta) > angle_limit:
            # print('end of curve, angle change to large', np.rad2deg(delta))
            break

        j = np.argmin(delta)
        ij = tuple(ij[j])
        #print('adding', ij)
        #ev_limit = 0.8*max_ev[ij]
        clist.append(ij)

    return clist


def angle_diff(a1, a2):
    """Angle difference computed by vector inproduct."""
    n1 = np.c_[np.cos(a1), np.sin(a1)]
    n2 = np.c_[np.cos(a2[:]), np.sin(a2[:])]
    angle = np.arccos((n1*n2).sum(1))
    return angle


def linking(p, evv, ev, img, start=None):
    """."""
    h, w = img.shape

    # first derivatives == 0
    abs_p = np.abs(p)
    ind_p = np.logical_and(abs_p[:, :, 0] < 1., abs_p[:, :, 1] < 1.)

    # largest eigen values, this mean dark on bright background curves only
    max_ev = np.max(ev, axis=2)
    ind_ev = max_ev > 0.15*np.amax(max_ev)

    # only compute angles for curve pixels defined by the 2 criteria above
    indices = np.logical_and(ind_p, ind_ev)

    # angles along the curve in [-pi, pi]
    angles = 10000*np.ones((h, w), dtype=np.float64)
    angles[indices] = -np.arctan2(evv[indices, 0], -evv[indices, 1])

    # start point based on strongest curve pixel
    if start is None:
        sub_window = max_ev[10:-10, 10:-10]
        i, j = np.where(sub_window == np.amax(sub_window))
        clist = [(i[0]+10, j[0]+10)]
    else:
        i, j = start
        delta = 10
        sub = max_ev[i:i+delta, j-delta:j+delta]
        k, l = np.where(sub == np.amax(sub))
        clist = [(i+k[0], j-delta+l[0])]
    #print('start at ', i, j)

    clist = link(clist, angles, max_ev, clist[0], h, w)

    clist.reverse()

    # now search in opposite direction
    angles[indices] = -np.arctan2(-evv[indices, 0], evv[indices, 1])
    clist = link(clist, angles, max_ev, clist[-1], h, w)

    return np.array(clist)


def find_curves(img, labels):
    """."""
    curves = {}
    for k, l in labels.items():
        #if not k == 'A15':
    #        continue
        print(k)

        img_w = img[l[1]:l[3], l[0]:l[2]]

        ImageGray = cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY)

        #plt.imshow(ImageGray, cmap='gray')
        #plt.show()
        
        p, evv, ev = compute_eigen(ImageGray)
        try:
            curves[k] = linking(p, evv, ev, ImageGray) + np.array([l[1], l[0]])
        except:
            pass

    return curves

if __name__=="__main__":
    import colorsys
    N = 27
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    color_dict={}
    gray_dict={}
    
    i=0
    for label in [lab_A, lab_F, lab_G]:
        for k, v in label.items():
            color_dict.setdefault(k,(RGB_tuples[i]))
            gray_dict.setdefault(k,1/27*(i+1))
            i+=1

    print(color_dict)
    print(gray_dict)
    bad_list=["DSC_2427","DSC_2430","DSC_2431","DSC_2432","DSC_2433","DSC_2661","DSC_2663","DSC_2673","DSC_2674"]

    for line in open("dataset/ImageSets/Main/all.txt","r").readlines()[:100]:
        f_id=line.strip()
        # f_id="DSC_2410"
        # if f_id in bad_list:
        #     continue
        if os.path.exists("dataset/GT/{}.png".format(f_id)):
            continue
        print("------"+f_id+"----------")
        img = cv2.imread('/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/JPEGImages/{}.JPG'.format(f_id))
        xml_path="/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/Annotations/{}.xml".format(f_id)
        box_dict={}
        tree = ET.parse(xml_path)
        for obj in tree.findall("object"):
            cls = "tuft"
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue

            bbox = obj.find("bndbox")
            bbox = [int(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]

            box_dict.setdefault(obj.find("name").text, bbox)
            
        curves=find_curves(img, box_dict)
        mask_gray=np.zeros(img.shape[:2])
        mask_color=np.zeros(img.shape)
        for k, c in curves.items():
            for j in range(c.shape[0]):
                mask_gray[c[j,0]-5:c[j,0]+5,c[j,1]-5:c[j,1]+5]=gray_dict[k]
                mask_color[c[j,0]-5:c[j,0]+5,c[j,1]-5:c[j,1]+5]=color_dict[k]
            # for j in range(c.shape[0]-1):
            #     mask=cv2.circle(mask, (c[j,0],c[j,1]),(c[j+1,0],c[j+1,1]),color_dict[k],10)

        cv2.imwrite("dataset/GT/{}.png".format(f_id),(mask_gray*255).astype(np.uint8))
        cv2.imwrite("dataset/GT_color/{}.png".format(f_id),(mask_color*255).astype(np.uint8))

        # break
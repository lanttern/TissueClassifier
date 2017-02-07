import cv2
import sys
import numpy as np

# img = cv2.imread(<file_path>)
def compute_white_area_1(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,210,255,0)
    return float(sum(sum(gray==255)))/ float((sum(sum(gray==255)) + sum(sum(gray==0))))

def compute_white_mask(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,210,255,0)
    return (gray==0)

def count_healthy_blobs_1(img):
    params = cv2.SimpleBlobDetector_Params()
    # Set threshold
    params.minThreshold = 0; params.maxThreshold = 150;
    # Filter by Area.
    params.filterByArea = True; params.minArea = 50; params.maxArea = 400
    # Filter by Circularity
    params.filterByCircularity = False; params.minCircularity = 0.2
    # Filter by Convexity
    params.filterByConvexity = True; params.minConvexity = 0.5
    # Filter by Inertia
    params.filterByInertia = True; params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    return len(keypoints)

# same as 1, but parameter tuned finer
def count_healthy_blobs_2(img):
    params = cv2.SimpleBlobDetector_Params()
    # Set threshold
    params.minThreshold = 0; params.maxThreshold = 115;
    # Filter by Area.
    params.filterByArea = True; params.minArea = 50; params.maxArea = 400
    # Filter by Circularity
    params.filterByCircularity = False; params.minCircularity = 0.2
    # Filter by Convexity
    params.filterByConvexity = True; params.minConvexity = 0.5
    # Filter by Inertia
    params.filterByInertia = True; params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    return len(keypoints)
    

def calc_feature_1(img):
    white_area = float(compute_white_area_1(img))
    if white_area > 0.95: return -1.
    return (  float(count_healthy_blobs_1(img)) / ((1. - white_area)*get_pix(img))   )

def calc_feature_2(img):
    white_area = float(compute_white_area_1(img))
    if white_area > 0.95: return -1.
    return (float(count_healthy_blobs_2(img)) / ((1. - white_area)*get_pix(img)))

def compute_avg_red(img):
    mask = compute_white_mask(img)
    imgr = img[:,:,0]
    return np.sum(imgr * mask) / np.sum(mask)

def compute_avg_green(img):
    mask = compute_white_mask(img)
    imgg = img[:,:,1]
    return np.sum(imgg * mask) / np.sum(mask)

def compute_avg_blue(img):
    mask = compute_white_mask(img)
    imgb = img[:,:,2]
    return np.sum(imgb * mask) / np.sum(mask)

def sift0_per_area(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,210,255,0)
    active_area = float(sum(sum(gray==0)))/ float((sum(sum(gray==255)) + sum(sum(gray==0))))
    if active_area < 0.05 : return -1
    sift = cv2.xfeatures2d.SIFT_create(0)
    kp,des = sift.detectAndCompute(gray,None)
    pix = np.shape(img)[0]*np.shape(img)[1]
    return float(len(kp)) / (active_area*get_pix(img))

def color_compactness(img):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    return (ret / float(len(Z))) * compute_white_area_1(img)

def get_pix(img):
    return float(np.shape(img)[0])*float(np.shape(img)[1])

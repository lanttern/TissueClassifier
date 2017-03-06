import cv2
import sys
import numpy as np

# img = cv2.imread(<file_path>)


def compute_white_area_1(img):
    """
    Compute the percentage of white area in the image
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert from RGB to greyscale
    ret, gray = cv2.threshold(gray,210,255,0) # apply a threshold to the image (210), maxVal=(255)
    return float(sum(sum(gray==255)))/ float((sum(sum(gray==255)) + sum(sum(gray==0))))

def compute_white_mask(img):
    """
    Mask out the white area in the image
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,210,255,0)
    return (gray==0)

def count_healthy_blobs_1(img):
    """
    Blob counting algorithm to identify nuclei
    """
    params = cv2.SimpleBlobDetector_Params()

    # Range for thresholding image
    params.minThreshold = 0; params.maxThreshold = 150;

    # Area defined by number of pixels
    params.filterByArea = True; params.minArea = 50; params.maxArea = 400

    # A circle has circularity of 1
    params.filterByCircularity = False; params.minCircularity = 0.2

    # Blob area divided by tightest convex blob around the image
    params.filterByConvexity = True; params.minConvexity = 0.5

    # How elongated a shape is; circle = 1, line=0
    params.filterByInertia = True; params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    return len(keypoints)

def count_healthy_blobs_2(img):
    """
    Same as other blob detection, but finely tuned
    """
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
    """
    Blobs per unit active area
    """
    white_area = float(compute_white_area_1(img))
    if white_area > 0.95: return -1.
    return (  float(count_healthy_blobs_1(img)) / ((1. - white_area)*get_pix(img))   )

def calc_feature_2(img):
    """
    Finer tuned blobs per unit area
    """
    white_area = float(compute_white_area_1(img))
    if white_area > 0.95: return -1.
    return (float(count_healthy_blobs_2(img)) / ((1. - white_area)*get_pix(img)))

def compute_avg_red(img):
    """
    Average red on active area of image
    """
    mask = compute_white_mask(img)
    imgr = img[:,:,0]
    return np.sum(imgr * mask) / np.sum(mask)

def compute_avg_green(img):
    """
    Average green on active area of image
    """
    mask = compute_white_mask(img)
    imgg = img[:,:,1]
    return np.sum(imgg * mask) / np.sum(mask)

def compute_avg_blue(img):
    """
    Average blue on active area of image
    """
    mask = compute_white_mask(img)
    imgb = img[:,:,2]
    return np.sum(imgb * mask) / np.sum(mask)

def sift0_per_area(img):
    """
    Edge detection algorithm for tissue discontinuity
    """
    # need the gray image, so redo this here rather than using white area function
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray,210,255,0)
    active_area = float(sum(sum(gray==0)))/ float((sum(sum(gray==255)) + sum(sum(gray==0))))
    if active_area < 0.05 : return -1
    # now apply sift
    sift = cv2.xfeatures2d.SIFT_create(0)
    kp,des = sift.detectAndCompute(gray,None)
    #pix = np.shape(img)[0]*np.shape(img)[1]
    return float(len(kp)) / (active_area*get_pix(img))

def color_compactness(img):
    """
    Measure of color variability in an image
    """
    # image preprocessing
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    # stop at 10 iterations or eps=1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3 # number of clusters
    # run the clustering
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # scale to number of pixels and penalize for white area
    return (ret / float(len(Z))) * compute_white_area_1(img)

def get_pix(img):
    """
    Total number of pixels in the image
    """
    return float(np.shape(img)[0])*float(np.shape(img)[1])

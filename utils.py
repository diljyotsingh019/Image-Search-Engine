import numpy as np #Used to handle pixel values from an image
import cv2 #Used for computer vision operations


def feature_extraction(image): #Used to extract histograms from different image fragments
    (h, w) = image.shape[:2] #Extracting the shape of the image
    (cx, cy) = (int(w//2), int(h//2)) # Calculating x center and y center for 4 quarters and an ellipse 
    (ax, ay) =  (int(w*0.75//2), int(h*0.75//2)) #Measuring the axes values for creating an ellipse

    mask = np.zeros(shape = image.shape[:2], dtype = "uint8") #Creating a zero matrix for masking
    segments = [(0, cx, 0, cy), (0, cx, cy, h),
                (cx, w, cy, h), (cx, w, 0, cy)] #Four segments (4 quarters without the ellipse)
    hist = []
    for x1, x2, y1, y2 in segments: #Iterating through different quarters of the image
        mask = cv2.rectangle(mask, pt1 = (x1, y1), pt2 = (x2, y2),
                             color = (255, 255, 255),
                             thickness = -1) #Selecting a quarter to unmask
        mask = cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, color = (0, 0, 0),
                           thickness = -1) #Masking the central ellipse

        for channel in range(3): # Calculating histograms for different channels(B,G,R)
            roi = cv2.calcHist([image], [channel], mask, [256], [0, 256])
            hist.append(roi)

    mask = np.zeros(shape=image.shape[:2], dtype = "uint8")
    mask = cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, color=(255, 255, 255),
                       thickness=-1) #Unmask just the central ellipse

    for channel in range(3): # Calculating histogram from different channels(B,G,R)
        roi = cv2.calcHist([image], [channel], mask, [256], [0, 256])  # Calculating the histogram for central ellipse
        hist.append(roi)

    return np.array(hist).flatten()

def chi2_distance(hist1, hist2, eps = 1e-10): #Used to calculate chi-square distance between two histograms
    d = 0.5*np.sum([((i-j)**2)/(i+j+eps) for i, j in zip(hist1, hist2)])
    return d

def search(image, hist_data): # Searching for similar images
    distance = []
    hist = feature_extraction(image) # Extracting the histogram of the image using the same user defined function
    for i in hist_data: #Iterating through the entire dataset to calculate distances between the image and every record in the dataset
        distance.append(chi2_distance(hist, i))

    return np.array(distance)


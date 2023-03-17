import cv2
import numpy as np

def face_detection(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        return face_image

def canny_edge(image, blur_kernel_size=5, low_threshold=100, high_threshold=200):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges
def canny_edge_auto(image, blur_kernel_size=5):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)
    
    v = np.median(blurred_image)
    sigma = 0.33
    low_threshold = int(max(0, (1.0 - sigma) * v))
    high_threshold = int(min(255, (1.0 + sigma) * v))
    
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges


def auto_canny(image, sigma=0.33):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    v = np.median(blurred_image)
    low_threshold = int(max(0, (1.0 - sigma) * v))
    high_threshold = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges

def laplacian_of_gaussian(image, ksize, sigma):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (ksize, ksize), sigma)
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    _, edges = cv2.threshold(np.absolute(laplacian), 10, 255, cv2.THRESH_BINARY)
    edges = edges.astype(np.uint8)
    return edges

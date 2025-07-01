import cv2
import numpy as np
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing class for face detection and edge detection operations."""
    
    def __init__(self):
        """Initialize the image processor with face cascade."""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if self.face_cascade.empty():
                raise ValueError("Failed to load face cascade classifier")
        except Exception as e:
            logger.error(f"Error initializing face cascade: {e}")
            raise
    
    def detect_face(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and extract the largest face from the image.
        If no face is detected, return the original image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Face region or original image if no face detected
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                logger.warning("No face detected, using full image")
                return image
            
            # Get the largest face
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            
            # Add some padding around the face
            padding = int(0.2 * min(w, h))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            face_image = image[y:y+h, x:x+w]
            logger.info(f"Face detected with dimensions: {w}x{h}")
            return face_image
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return image
    
    def canny_edge_detection(
        self, 
        image: np.ndarray, 
        blur_kernel_size: int = 5, 
        low_threshold: int = 100, 
        high_threshold: int = 200
    ) -> np.ndarray:
        """
        Apply Canny edge detection with Gaussian blur.
        
        Args:
            image: Input image
            blur_kernel_size: Size of Gaussian blur kernel (must be odd)
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            
        Returns:
            Edge-detected image
        """
        try:
            # Ensure kernel size is odd and positive
            blur_kernel_size = max(1, blur_kernel_size)
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(
                gray_image, 
                (blur_kernel_size, blur_kernel_size), 
                0
            )
            edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
            
            # Invert edges for better line drawing effect
            edges = cv2.bitwise_not(edges)
            return edges
            
        except Exception as e:
            logger.error(f"Error in Canny edge detection: {e}")
            raise
    
    def auto_canny_edge_detection(
        self, 
        image: np.ndarray, 
        blur_kernel_size: int = 5, 
        sigma: float = 0.33
    ) -> np.ndarray:
        """
        Apply automatic Canny edge detection based on image statistics.
        
        Args:
            image: Input image
            blur_kernel_size: Size of Gaussian blur kernel
            sigma: Threshold calculation parameter
            
        Returns:
            Edge-detected image
        """
        try:
            # Ensure kernel size is odd and positive
            blur_kernel_size = max(1, blur_kernel_size)
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(
                gray_image, 
                (blur_kernel_size, blur_kernel_size), 
                0
            )
            
            # Automatic threshold calculation
            v = np.median(blurred_image)
            low_threshold = int(max(0, (1.0 - sigma) * v))
            high_threshold = int(min(255, (1.0 + sigma) * v))
            
            logger.info(f"Auto thresholds: low={low_threshold}, high={high_threshold}")
            
            edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
            
            # Invert edges for better line drawing effect
            edges = cv2.bitwise_not(edges)
            return edges
            
        except Exception as e:
            logger.error(f"Error in auto Canny edge detection: {e}")
            raise
    
    def laplacian_of_gaussian(
        self, 
        image: np.ndarray, 
        blur_kernel_size: int = 5, 
        sigma: float = 1.0,
        threshold: float = 10.0
    ) -> np.ndarray:
        """
        Apply Laplacian of Gaussian edge detection.
        
        Args:
            image: Input image
            blur_kernel_size: Size of Gaussian blur kernel
            sigma: Standard deviation for Gaussian blur
            threshold: Threshold for edge detection
            
        Returns:
            Edge-detected image
        """
        try:
            # Ensure kernel size is odd and positive
            blur_kernel_size = max(1, blur_kernel_size)
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(
                gray_image, 
                (blur_kernel_size, blur_kernel_size), 
                sigma
            )
            
            laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
            _, edges = cv2.threshold(
                np.absolute(laplacian), 
                threshold, 
                255, 
                cv2.THRESH_BINARY
            )
            edges = edges.astype(np.uint8)
            
            # Invert edges for better line drawing effect
            edges = cv2.bitwise_not(edges)
            return edges
            
        except Exception as e:
            logger.error(f"Error in Laplacian of Gaussian: {e}")
            raise

# Global instance for backward compatibility
_processor = None

def get_processor() -> ImageProcessor:
    """Get or create the global image processor instance."""
    global _processor
    if _processor is None:
        _processor = ImageProcessor()
    return _processor

# Backward compatibility functions
def face_detection(image: np.ndarray) -> np.ndarray:
    """Backward compatibility wrapper for face detection."""
    return get_processor().detect_face(image)

def canny_edge(image: np.ndarray, blur_kernel_size: int = 5, 
               low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    """Backward compatibility wrapper for Canny edge detection."""
    return get_processor().canny_edge_detection(
        image, blur_kernel_size, low_threshold, high_threshold
    )

def canny_edge_auto(image: np.ndarray, blur_kernel_size: int = 5) -> np.ndarray:
    """Backward compatibility wrapper for auto Canny edge detection."""
    return get_processor().auto_canny_edge_detection(image, blur_kernel_size)

def laplacian_of_gaussian(image: np.ndarray, blur_kernel_size: int = 5, 
                         sigma: float = 1.0) -> np.ndarray:
    """Backward compatibility wrapper for Laplacian of Gaussian."""
    return get_processor().laplacian_of_gaussian(image, blur_kernel_size, sigma)

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import io
import os
import logging
from PIL import Image
from werkzeug.utils import secure_filename
from selfie_line_drawing import ImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    MAX_IMAGE_DIMENSION = 2048  # Max width/height for processing

app.config.from_object(Config)

# Initialize image processor
try:
    image_processor = ImageProcessor()
    logger.info("Image processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize image processor: {e}")
    image_processor = None

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def resize_image_if_needed(image, max_dimension=Config.MAX_IMAGE_DIMENSION):
    """Resize image if it exceeds maximum dimensions."""
    height, width = image.shape[:2]
    
    if height > max_dimension or width > max_dimension:
        # Calculate new dimensions maintaining aspect ratio
        if height > width:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        else:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
    
    return image

def validate_parameters(form_data):
    """Validate and sanitize form parameters."""
    try:
        method = form_data.get('method', 'canny')
        if method not in ['canny', 'auto_canny', 'log']:
            method = 'canny'
        
        blur_kernel_size = int(form_data.get('blur_kernel_size', 5))
        blur_kernel_size = max(1, min(31, blur_kernel_size))  # Clamp between 1-31
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1  # Ensure odd number
        
        low_threshold = int(form_data.get('low_threshold', 100))
        low_threshold = max(0, min(255, low_threshold))
        
        high_threshold = int(form_data.get('high_threshold', 200))
        high_threshold = max(0, min(255, high_threshold))
        
        # Ensure high_threshold > low_threshold
        if high_threshold <= low_threshold:
            high_threshold = low_threshold + 50
            high_threshold = min(255, high_threshold)
        
        sigma = float(form_data.get('sigma', 1.0))
        sigma = max(0.1, min(5.0, sigma))
        
        return {
            'method': method,
            'blur_kernel_size': blur_kernel_size,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'sigma': sigma
        }
    except (ValueError, TypeError) as e:
        logger.warning(f"Parameter validation error: {e}")
        # Return default values on error
        return {
            'method': 'canny',
            'blur_kernel_size': 5,
            'low_threshold': 100,
            'high_threshold': 200,
            'sigma': 1.0
        }

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error occurred.'}), 500

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/line-drawing', methods=['POST'])
def line_drawing():
    """
    Process uploaded image and convert to line drawing.
    
    Returns:
        JSON response with base64 encoded result image or error message
    """
    try:
        # Check if image processor is available
        if image_processor is None:
            return jsonify({'error': 'Image processing service unavailable'}), 503
        
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(Config.ALLOWED_EXTENSIONS)}'}), 400

        # Validate and sanitize parameters
        params = validate_parameters(request.form)
        logger.info(f"Processing with parameters: {params}")

        # Load and validate image
        try:
            input_image = Image.open(file.stream)
            # Convert to RGB if necessary
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return jsonify({'error': 'Invalid image file or corrupted data'}), 400

        # Resize image if too large
        input_image = resize_image_if_needed(input_image)

        # Detect face (with fallback to full image)
        try:
            face_image = image_processor.detect_face(input_image)
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            face_image = input_image  # Fallback to full image

        # Apply edge detection based on method
        try:
            if params['method'] == 'canny':
                edge_image = image_processor.canny_edge_detection(
                    face_image, 
                    params['blur_kernel_size'], 
                    params['low_threshold'], 
                    params['high_threshold']
                )
            elif params['method'] == 'auto_canny':
                edge_image = image_processor.auto_canny_edge_detection(
                    face_image, 
                    params['blur_kernel_size']
                )
            elif params['method'] == 'log':
                edge_image = image_processor.laplacian_of_gaussian(
                    face_image, 
                    params['blur_kernel_size'],
                    params['sigma']
                )
            else:
                return jsonify({'error': 'Invalid processing method'}), 400
                
        except Exception as e:
            logger.error(f"Error in edge detection ({params['method']}): {e}")
            return jsonify({'error': f'Error processing image with {params["method"]} method'}), 500

        # Encode result image
        try:
            _, buffer = cv2.imencode('.jpg', edge_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            output_image_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding result image: {e}")
            return jsonify({'error': 'Error generating result image'}), 500

        logger.info("Image processing completed successfully")
        return jsonify({
            'image': output_image_base64,
            'method': params['method'],
            'parameters': params
        })

    except Exception as e:
        logger.error(f"Unexpected error in line_drawing: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'image_processor': image_processor is not None,
        'version': '2.0.0'
    }
    return jsonify(status)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}, debug={debug}")
    app.run(debug=debug, port=port, host='0.0.0.0')

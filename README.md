# AI Line Drawing Converter

A sophisticated web application that transforms photos into beautiful line drawings using advanced computer vision and edge detection algorithms.

## 🚀 Features

### Core Functionality
- **Smart Face Detection**: Automatically detects and focuses on faces in uploaded images
- **Multiple Edge Detection Methods**:
  - **Manual Canny**: Full control over edge detection parameters
  - **Auto Canny**: Intelligent automatic threshold detection
  - **Laplacian of Gaussian (LoG)**: Advanced edge detection with Gaussian filtering

### User Experience
- **Modern Responsive UI**: Beautiful, mobile-friendly interface with Bootstrap 5
- **Drag & Drop Upload**: Intuitive file upload with drag-and-drop support
- **Real-time Preview**: Instant image preview before processing
- **Live Parameter Adjustment**: Interactive sliders with real-time value display
- **Conditional UI**: Parameters shown based on selected method
- **Progress Indicators**: Loading spinners and status messages
- **Side-by-side Comparison**: View original and processed images together
- **Download Results**: One-click download of generated line drawings

### Technical Improvements
- **Robust Error Handling**: Comprehensive validation and error reporting
- **Security Features**: File type validation, size limits, input sanitization
- **Performance Optimization**: Image resizing, efficient processing
- **Logging System**: Detailed logging for debugging and monitoring
- **Health Check Endpoint**: Application monitoring support

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd line-drawing-converter
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the application**:
   Open your browser and navigate to `http://localhost:5005`

## 📋 Usage

1. **Upload an Image**:
   - Click "Select Image" or drag and drop an image file
   - Supported formats: JPG, PNG, GIF, BMP, WebP
   - Maximum file size: 16MB

2. **Choose Processing Method**:
   - **Manual Canny**: Adjust low/high thresholds for custom edge detection
   - **Auto Canny**: Let the algorithm automatically determine optimal thresholds
   - **LoG Filter**: Use Laplacian of Gaussian for different edge characteristics

3. **Adjust Parameters**:
   - **Blur Kernel Size**: Controls image smoothing (1-31, odd numbers)
   - **Low/High Threshold**: Edge detection sensitivity (Manual Canny only)
   - **Sigma**: Gaussian blur standard deviation (LoG only)

4. **Process and Download**:
   - Click "Convert to Line Drawing"
   - View the comparison between original and processed images
   - Download the result with one click

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 5005)
- `FLASK_DEBUG`: Enable debug mode (default: True)

### Application Limits
- Maximum file size: 16MB
- Maximum image dimension: 2048px (auto-resized)
- Supported formats: PNG, JPG, JPEG, GIF, BMP, WebP

## 🏗️ Architecture

### Backend (`app.py`)
- **Flask Web Framework**: RESTful API design
- **Error Handling**: Comprehensive exception handling and user feedback
- **Input Validation**: File type, size, and parameter validation
- **Security**: Input sanitization and file upload restrictions
- **Logging**: Structured logging for monitoring and debugging

### Image Processing (`selfie_line_drawing.py`)
- **Object-Oriented Design**: Clean, maintainable `ImageProcessor` class
- **Advanced Face Detection**: Improved accuracy with padding and fallback
- **Multiple Edge Detection Algorithms**: Canny, Auto-Canny, and LoG
- **Error Resilience**: Graceful handling of processing failures
- **Parameter Validation**: Automatic correction of invalid parameters

### Frontend (`templates/index.html`)
- **Modern Design**: Gradient backgrounds, card-based layout, animations
- **Responsive Layout**: Mobile-first design with CSS Grid and Flexbox
- **Interactive UI**: Method selection cards, range sliders with live values
- **Drag & Drop**: Native HTML5 file upload with visual feedback
- **Real-time Validation**: Client-side file type and size checking
- **Accessibility**: Semantic HTML, proper ARIA labels, keyboard navigation

## 🔄 API Endpoints

### `POST /api/line-drawing`
Process an uploaded image and return the line drawing result.

**Parameters**:
- `file`: Image file (multipart/form-data)
- `method`: Processing method (`canny`, `auto_canny`, `log`)
- `blur_kernel_size`: Blur kernel size (1-31, odd numbers)
- `low_threshold`: Low threshold (0-255, Canny only)
- `high_threshold`: High threshold (0-255, Canny only)
- `sigma`: Gaussian sigma (0.1-5.0, LoG only)

**Response**:
```json
{
  "image": "base64-encoded-image",
  "method": "processing-method",
  "parameters": {
    "blur_kernel_size": 5,
    "low_threshold": 100,
    "high_threshold": 200
  }
}
```

### `GET /health`
Health check endpoint for monitoring.

**Response**:
```json
{
  "status": "healthy",
  "image_processor": true,
  "version": "2.0.0"
}
```

## 🚀 Deployment

### Production Deployment
```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5005 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5005

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5005", "app:app"]
```

## 🔍 Key Improvements Made

### Code Quality
- ✅ Object-oriented design with proper separation of concerns
- ✅ Comprehensive error handling and logging
- ✅ Type hints and documentation
- ✅ Input validation and sanitization
- ✅ Removed unused code and fixed bugs

### Security
- ✅ File type and size validation
- ✅ Input parameter sanitization
- ✅ Secure file handling
- ✅ Error message sanitization

### Performance
- ✅ Image resizing for large files
- ✅ Optimized face cascade loading
- ✅ Efficient image encoding
- ✅ Proper resource management

### User Experience
- ✅ Modern, responsive design
- ✅ Drag & drop file upload
- ✅ Real-time preview and feedback
- ✅ Loading states and progress indicators
- ✅ Error messages and validation
- ✅ Download functionality

### Maintainability
- ✅ Clean, documented code
- ✅ Modular architecture
- ✅ Configuration management
- ✅ Version control friendly structure

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support, please open an issue in the GitHub repository or contact the development team.
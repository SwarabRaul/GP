# 1. Sobel, Prewitt, and Robert Operator
import numpy as np
import cv2

def convolve2d(image, kernel):
    """Manual implementation of 2D convolution."""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)
    return output

# Sobel Kernels
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Prewitt Kernels
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Roberts Kernels
roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

# Usage
image = cv2.imread("example.jpg", cv2.IMREAD_GRAYSCALE)
sobel_result = np.sqrt(convolve2d(image, sobel_x)**2 + convolve2d(image, sobel_y)**2)


# 2. Region Growing
def region_growing(image, seed, threshold):
    """Manual region growing algorithm."""
    h, w = image.shape
    visited = np.zeros((h, w), dtype=bool)
    region = np.zeros_like(image)
    stack = [seed]

    while stack:
        x, y = stack.pop()
        if not visited[x, y]:
            visited[x, y] = True
            region[x, y] = image[x, y]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    if abs(int(image[nx, ny]) - int(image[x, y])) < threshold:
                        stack.append((nx, ny))
    return region

# Usage
seed_point = (50, 50)  # Example seed point
region = region_growing(image, seed_point, threshold=10)



# 3. Erosion and Dilation
def erosion(image, kernel):
    """Manual erosion operation."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kh, j:j + kw]
            output[i, j] = np.min(region[kernel == 1])
    return output

def dilation(image, kernel):
    """Manual dilation operation."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kh, j:j + kw]
            output[i, j] = np.max(region[kernel == 1])
    return output

# Example kernel
kernel = np.ones((3, 3), dtype=np.uint8)
eroded = erosion(image, kernel)
dilated = dilation(image, kernel)



# 4. Homomorphic Filtering
def homomorphic_filtering(image, cutoff=30, gamma_h=2.0, gamma_l=0.5):
    """Homomorphic filtering in the frequency domain."""
    rows, cols = image.shape
    x = np.linspace(-cols//2, cols//2, cols)
    y = np.linspace(-rows//2, rows//2, rows)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)

    # High-pass filter
    H = 1 - np.exp(- (D**2) / (2 * (cutoff**2)))
    H = np.fft.ifftshift(H)

    # Apply filtering
    image_log = np.log1p(image.astype(np.float64))  # Log transform
    fft_image = np.fft.fft2(image_log)
    filtered_image = np.fft.ifft2(fft_image * H)
    result = np.expm1(np.real(filtered_image))
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

# Usage
homomorphic_result = homomorphic_filtering(image)




# 5. Watershed Algorithm
def manual_watershed(image, markers):
    """Simple manual implementation of watershed segmentation."""
    from scipy.ndimage import label
    gradient = np.abs(np.gradient(image)[0]) + np.abs(np.gradient(image)[1])
    labels, num_features = label(markers)
    segmented = np.zeros_like(image)

    for label_id in range(1, num_features + 1):
        mask = labels == label_id
        segment_mean = np.mean(image[mask])
        segmented[mask] = segment_mean

    return segmented

# Usage
# Create a marker image where objects are marked with unique integer values
markers = np.zeros_like(image)
markers[50:100, 50:100] = 1
markers[150:200, 150:200] = 2
watershed_result = manual_watershed(image, markers)




# 6. Skeletonization
def skeletonize(image):
    """Manual implementation of skeletonization."""
    from scipy.ndimage import binary_erosion
    import numpy as np

    skeleton = np.zeros_like(image, dtype=bool)
    eroded = image.copy()

    while np.any(eroded):
        eroded_next = binary_erosion(eroded)
        temp = np.logical_xor(eroded, eroded_next)  # Fix: Use logical_xor
        skeleton = np.logical_or(skeleton, temp)
        eroded = eroded_next

    return skeleton.astype(np.uint8) * 255

# Usage
binary_image = (image > 128).astype(np.uint8)  # Convert to binary
skeleton = skeletonize(binary_image)





# 7. Background Extraction
def background_extraction(video_frames, alpha=0.5):
    """Background extraction using running average."""
    avg_frame = np.zeros_like(video_frames[0], dtype=np.float32)

    for frame in video_frames:
        avg_frame = alpha * frame + (1 - alpha) * avg_frame

    return avg_frame.astype(np.uint8)

# Usage
# Assume video_frames is a list of grayscale frames
video_frames = [cv2.imread(f"frame{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(1, 6)]
background = background_extraction(video_frames)




# 8. Flood Filling
def flood_fill(image, seed_point, new_value):
    """Manual flood-fill implementation."""
    old_value = image[seed_point]
    if old_value == new_value:
        return image

    h, w = image.shape
    stack = [seed_point]
    while stack:
        x, y = stack.pop()
        if image[x, y] == old_value:
            image[x, y] = new_value
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    stack.append((nx, ny))
    return image

# Usage
seed = (50, 50)
filled_image = flood_fill(image.copy(), seed, 255)





# 9. GLCM (Gray-Level Co-occurrence Matrix)
def compute_glcm(image, distance=1, angle=0):
    """Manual GLCM computation."""
    h, w = image.shape
    max_intensity = image.max() + 1
    glcm = np.zeros((max_intensity, max_intensity), dtype=np.int32)

    dx = int(distance * np.cos(angle))
    dy = int(distance * np.sin(angle))

    for x in range(h):
        for y in range(w):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                i, j = image[x, y], image[nx, ny]
                glcm[i, j] += 1

    return glcm

# Usage
gray_image = (image / 255 * 15).astype(np.uint8)  # Reduce intensity levels for simplicity
glcm = compute_glcm(gray_image, distance=1, angle=0)





# 10. Blob Detection
def detect_blobs(image, threshold=10):
    """Manual blob detection based on connected components using OpenCV."""
    # Threshold the image
    binary_image = (image > threshold).astype(np.uint8)
    
    # Find connected components
    num_features, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Extract blob centers (excluding the background)
    blobs = [centroid for centroid in centroids[1:]]
    
    return blobs

# Usage
blobs = detect_blobs(image, threshold=128)




# 11. Laplacian and Gaussian
def laplacian(image):
    """Manual Laplacian filter."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolve2d(image, kernel)

def gaussian(image, sigma=1.0):
    """Manual Gaussian filter."""
    size = int(2 * (3 * sigma) + 1)
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    gaussian_kernel = np.outer(kernel, kernel)
    return convolve2d(image, gaussian_kernel)

# Usage
laplacian_result = laplacian(image)
gaussian_result = gaussian(image, sigma=1.5)




# 12. Active Contours
def active_contours(image, snake, alpha=0.1, beta=0.1, gamma=0.1, iterations=100):
    """Simple active contour model."""
    import numpy as np  # Ensure numpy is imported
    from scipy.ndimage import gaussian_filter

    # Ensure snake is a float array to allow float calculations
    snake = snake.astype(np.float64)

    for _ in range(iterations):
        # Compute gradients of the image
        fx, fy = np.gradient(image)
        force = np.array([fx, fy])
        snake_force = np.zeros_like(snake)

        for i, point in enumerate(snake):
            x, y = int(point[0]), int(point[1])
            snake_force[i] = force[:, y, x]  # Corrected indexing for force application

        # Update snake positions
        snake += alpha * (np.roll(snake, -1, axis=0) - 2 * snake + np.roll(snake, 1, axis=0))
        snake += beta * (np.roll(snake, -1, axis=0) - snake) - gamma * snake_force
        
        # Apply smoothing
        snake = gaussian_filter(snake, sigma=1)

    return snake


# Usage
snake = np.array([[50, 50], [50, 100], [100, 100], [100, 50]])
contour = active_contours(image, snake)



# Segmentation Function
import cv2
import numpy as np

def segment_image(image, method="binary", threshold=128, max_value=255, block_size=11, C=2):
    """
    Segment an image using various thresholding techniques.
    
    Parameters:
    - image: Grayscale image (2D array)
    - method: The segmentation method to use:
        - "binary": Simple binary thresholding
        - "otsu": Otsu's thresholding
        - "adaptive_mean": Adaptive mean thresholding
        - "adaptive_gaussian": Adaptive Gaussian thresholding
    - threshold: Threshold value for binary segmentation (ignored for Otsu/Adaptive methods)
    - max_value: Maximum value to assign for thresholded pixels
    - block_size: Size of the neighborhood for adaptive methods (must be odd)
    - C: Constant subtracted from the mean in adaptive methods
    
    Returns:
    - Segmented image (binary mask)
    """
    if method == "binary":
        _, segmented = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
    
    elif method == "otsu":
        _, segmented = cv2.threshold(image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif method == "adaptive_mean":
        segmented = cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, block_size, C)
    
    elif method == "adaptive_gaussian":
        segmented = cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, block_size, C)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'binary', 'otsu', 'adaptive_mean', or 'adaptive_gaussian'.")
    
    return segmented

# Example Usage
image = cv2.imread("example.jpg", cv2.IMREAD_GRAYSCALE)

binary_segmented = segment_image(image, method="binary", threshold=128)
otsu_segmented = segment_image(image, method="otsu")
adaptive_mean_segmented = segment_image(image, method="adaptive_mean", block_size=11, C=2)
adaptive_gaussian_segmented = segment_image(image, method="adaptive_gaussian", block_size=11, C=2)




# Matplotlib Code to Visualize All Steps
import matplotlib.pyplot as plt

def plot_all_steps(original, sobel, prewitt, roberts, region, eroded, dilated, homomorphic, 
                   watershed, skeleton, background, flood_filled, glcm, blobs, laplacian, gaussian, contours):
    """
    Plots all steps from the original image to processed results.
    """
    fig, axes = plt.subplots(6, 3, figsize=(15, 20))
    axes = axes.ravel()  # Flatten the grid of axes

    images = [
        (original, "Original Image"),
        (sobel, "Sobel Edge Detection"),
        (prewitt, "Prewitt Edge Detection"),
        (roberts, "Roberts Edge Detection"),
        (region, "Region Growing"),
        (eroded, "Eroded Image"),
        (dilated, "Dilated Image"),
        (homomorphic, "Homomorphic Filtering"),
        (watershed, "Watershed Segmentation"),
        (skeleton, "Skeletonization"),
        (background, "Background Extraction"),
        (flood_filled, "Flood Filled Image"),
        (glcm, "GLCM Visualization"),
        (blobs, "Blob Detection"),
        (laplacian, "Laplacian Filter"),
        (gaussian, "Gaussian Filter"),
        (contours, "Active Contours"),
    ]

    for i, (image, title) in enumerate(images):
        if isinstance(image, np.ndarray):  # For images
            cmap = 'gray' if len(image.shape) == 2 else None
            axes[i].imshow(image, cmap=cmap)
        elif isinstance(image, list):  # For special cases like blobs or contours
            axes[i].imshow(original, cmap='gray')  # Overlay on original
            for point in image:  # Blobs or contour points
                axes[i].plot(point[1], point[0], 'r.', markersize=5)
        else:
            axes[i].text(0.5, 0.5, "Not Visualizable", fontsize=12, ha='center', va='center')
            axes[i].set_facecolor('lightgray')
        
        axes[i].set_title(title)
        axes[i].axis('off')

    # Remove unused axes if there are any
    for j in range(len(images), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Example Usage
plot_all_steps(
    original=image,              # Original grayscale image
    sobel=sobel_result,          # Sobel edge detection result
    prewitt=prewitt_result,      # Prewitt edge detection result
    roberts=roberts_result,      # Roberts edge detection result
    region=region,               # Region growing result
    eroded=eroded,               # Erosion result
    dilated=dilated,             # Dilation result
    homomorphic=homomorphic_result,  # Homomorphic filtering result
    watershed=watershed_result,  # Watershed segmentation result
    skeleton=skeleton,           # Skeletonization result
    background=background,       # Background extraction result
    flood_filled=filled_image,   # Flood-filled result
    glcm=np.log1p(glcm),         # Log-scaled GLCM for visualization
    blobs=blobs,                 # Blob detection (list of coordinates)
    laplacian=laplacian_result,  # Laplacian filter result
    gaussian=gaussian_result,    # Gaussian filter result
    contours=contour             # Active contours (array of points)
)



# Image Loading & Ploting
import cv2
import matplotlib.pyplot as plt

# Load the image using OpenCV
image_path = 'path_to_your_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Display the image using OpenCV
cv2.imshow('Image - OpenCV', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the BGR image (default in OpenCV) to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the image using Matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title('Image - Matplotlib')
plt.axis('off')
plt.show()



# Image Color Conversion
import cv2
import matplotlib.pyplot as plt

# Load the image in BGR format
image_path = 'path_to_your_image.jpg'  # Replace with your image path
image_bgr = cv2.imread(image_path)

# Convert BGR to various color models
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

# Display the converted images using Matplotlib
titles = ['BGR (Original)', 'RGB', 'Grayscale', 'HSV', 'LAB']
images = [image_bgr, image_rgb, image_gray, image_hsv, image_lab]

for i in range(len(images)):
    plt.figure(figsize=(8, 6))
    if len(images[i].shape) == 2:  # Grayscale image
        plt.imshow(images[i], cmap='gray')
    else:  # Color image
        plt.imshow(images[i] if i != 0 else cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
    plt.show()


"""OpenCV contour-based skull stripping, cropping, padding, and image normalization pipeline."""

from typing import Tuple, Optional
import cv2
import numpy as np


class MRIPreprocessor:
    """Preprocesses brain MRI scans via contour detection, background cropping with safety padding, resizing, and normalization."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (150, 150),
        blur_kernel: Tuple[int, int] = (5, 5),
        min_contour_area_ratio: float = 0.05,
        padding: int = 0,
    ):
        """Initializes the MRI preprocessor.

        Args:
            target_size: Target (height, width) for model input. Default: (150, 150).
            blur_kernel: Gaussian blur kernel size for noise reduction. Default: (5, 5).
            min_contour_area_ratio: Minimum ratio of image area for contour to be considered brain tissue.
            padding: Extra bounding-box safety padding (in pixels) added around cropped brain region.
        """
        self.target_size = target_size
        self.blur_kernel = blur_kernel
        self.min_contour_area_ratio = min_contour_area_ratio
        self.padding = max(0, int(padding))

    def crop_brain_contour(self, image: np.ndarray) -> np.ndarray:
        """Finds the extreme outer contours of the brain/skull and crops out surrounding black borders with optional safety padding.

        Order of operations:
        1. Convert image to grayscale.
        2. Apply Gaussian blur to reduce high-frequency scanner noise.
        3. Threshold image (Otsu's thresholding) to create binary brain mask.
        4. Apply morphological closing and opening to seal gaps within cranial boundaries.
        5. Find contours, locate largest contour, and extract bounding coordinates (top, bottom, left, right).
        6. Apply configurable safety padding and clip coordinates strictly inside image boundaries.
        7. Crop original image to bounding box. Fall back to uncropped image if no valid contour is found.

        Args:
            image: Input image array of shape (H, W, 3) or (H, W).

        Returns:
            Cropped image array focused on brain tissue.
        """
        # Convert to grayscale if image has color channels
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            gray = image.squeeze(axis=-1)
        else:
            gray = image.copy()

        # Gaussian blur to filter noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        # Threshold using Otsu's binarization
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations to close minor gaps inside brain tissue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find external contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Select the contour with the largest area (the skull/brain)
        largest_contour = max(contours, key=cv2.contourArea)
        img_h, img_w = image.shape[0], image.shape[1]
        total_area = img_h * img_w

        # Guard against tiny artifact contours
        if cv2.contourArea(largest_contour) < self.min_contour_area_ratio * total_area:
            return image

        # Determine extreme boundary points
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Apply safety padding with boundary clipping
        y1 = max(0, y - self.padding)
        y2 = min(img_h, y + h + self.padding)
        x1 = max(0, x - self.padding)
        x2 = min(img_w, x + w + self.padding)

        # Safe slicing
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return image

        return cropped

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resizes image to target dimensions using INTER_AREA interpolation."""
        return cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)

    def normalize(self, image: np.ndarray, method: str = "zero_one") -> np.ndarray:
        """Normalizes image pixel values.

        Args:
            image: Image array.
            method: 'zero_one' scales to [0.0, 1.0]; 'imagenet' applies standard ImageNet mean & std.

        Returns:
            Normalized float32 array.
        """
        img_float = image.astype(np.float32)

        if method == "zero_one":
            return img_float / 255.0
        elif method == "imagenet":
            # ImageNet mean and std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            scaled = img_float / 255.0
            return (scaled - mean) / std
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def preprocess(self, image: np.ndarray, normalize_method: str = "zero_one") -> np.ndarray:
        """Executes full preprocessing pipeline: Crop -> Resize -> Normalize."""
        cropped = self.crop_brain_contour(image)
        resized = self.resize_image(cropped)
        normalized = self.normalize(resized, method=normalize_method)
        return normalized

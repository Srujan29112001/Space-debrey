"""
Space-specific image preprocessing
Handles star removal, cosmic ray detection, contrast enhancement
"""

import cv2
import numpy as np
from typing import Tuple


class SpaceImagePreprocessor:
    """Preprocess telescope imagery for debris detection"""

    def __init__(self):
        """Initialize preprocessor"""
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline

        Args:
            image: Input telescope image (H, W, C)

        Returns:
            Processed image
        """
        # Remove stars
        star_removed = self.remove_stars(image)

        # Enhance contrast for dim objects
        enhanced = self.enhance_contrast(star_removed)

        # Detect and remove cosmic rays
        cleaned = self.remove_cosmic_rays(enhanced)

        return cleaned

    def remove_stars(self, image: np.ndarray) -> np.ndarray:
        """
        Remove stars using median filtering

        Args:
            image: Input image

        Returns:
            Image with stars removed
        """
        # Median filter to remove point sources
        filtered = cv2.medianBlur(image, 5)
        return filtered

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast for dim objects

        Args:
            image: Input image

        Returns:
            Contrast-enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply CLAHE
        enhanced = self.clahe.apply(gray)

        # Convert back to BGR if needed
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return enhanced

    def remove_cosmic_rays(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and remove cosmic ray hits

        Args:
            image: Input image

        Returns:
            Image with cosmic rays removed
        """
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect cosmic rays (very bright single pixels)
        cosmic_mask = self.detect_cosmic_rays(gray)

        # Inpaint cosmic rays
        if len(image.shape) == 3:
            cleaned = cv2.inpaint(image, cosmic_mask, 3, cv2.INPAINT_TELEA)
        else:
            cleaned = cv2.inpaint(gray, cosmic_mask, 3, cv2.INPAINT_TELEA)

        return cleaned

    def detect_cosmic_rays(self, image: np.ndarray) -> np.ndarray:
        """
        Detect cosmic ray pixels

        Args:
            image: Grayscale image

        Returns:
            Binary mask of cosmic rays
        """
        # Compute median-filtered image
        median = cv2.medianBlur(image, 5)

        # Difference from median
        diff = np.abs(image.astype(float) - median.astype(float))

        # Threshold for cosmic rays (very bright outliers)
        threshold = np.mean(diff) + 5 * np.std(diff)
        cosmic_mask = (diff > threshold).astype(np.uint8) * 255

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cosmic_mask = cv2.morphologyEx(cosmic_mask, cv2.MORPH_OPEN, kernel)

        return cosmic_mask

    def normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize brightness across image

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        # Convert to float
        img_float = image.astype(float)

        # Normalize to 0-255 range
        img_min = np.min(img_float)
        img_max = np.max(img_float)

        if img_max > img_min:
            normalized = ((img_float - img_min) / (img_max - img_min) * 255)
        else:
            normalized = img_float

        return normalized.astype(np.uint8)

from change_detection.changeDetectionStrategy import ChangeDetectionStrategy
import cv2
import numpy as np

class PixelDifferenceChangeDetectionStrategy(ChangeDetectionStrategy):
    """
    Concrete implementation of the ChangeDetectionStrategy (Strategy pattern) which can be used by the ChangeDetector. 
    Performs change detection by taking the difference between the pixels in a given image and comparing it against a threshold.

    Design Pattern:
        Concrete implementation of the Strategy interface of the Strategy design pattern (ChangeDetectionStrategy).
    """
    def __init__(self):
        super().__init__()
        self.previous_frame = None
        self.percentage_of_the_frame_as_count_threshold = 0.0005 # threshold will be set to the (0.05) percent of the total frame size.
        self.diff_threshold = 30 # used to threshold the pixel difference between the two consecutive frames.

    def executeChangeDetection(self, frame):
        """
        Compare each frame to the previous one by calculating the difference between them. 
        If the difference is above a certain threshold, then a change is detected.

        Inputs:
            frame (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3].

        Return:
            True: if change detected.
            False: if no change detected.
        ---
            Pros: Fast, simple, minimal computational cost.
            Cons: Sensitive to noise, may not work well in complex or noisy backgrounds.
        """
        # Convert the frame to grayscale
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.previous_frame is None:
            self.previous_frame = current_gray
            return False

        # Calculate the absolute difference between the current frame and the previous frame
        diff = cv2.absdiff(self.previous_frame, current_gray)

        # Apply a threshold to the difference
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # Count the non-zero pixels in the thresholded difference image
        change_count = np.count_nonzero(thresh)

        # Update previous frame
        self.previous_frame = current_gray

        threshold = np.multiply(*diff.shape) * self.percentage_of_the_frame_as_count_threshold 

        # Set a threshold for the number of changed pixels to trigger detection
        if change_count > threshold:  # Adjust this threshold as needed
            return True
        else:
            return False

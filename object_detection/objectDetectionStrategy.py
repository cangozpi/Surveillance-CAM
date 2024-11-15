from abc import ABC, abstractmethod

class ObjectDetectionStrategy(ABC):
    """
    Implement this class to create an algorithm which can be used interchangeably with other algorithms to perform object detection in images. 

    Design Pattern:
        Strategy interface of the Strategy design pattern (ObjectDetectionStrategy).
    """
    def __init__(self):
        pass

    @abstractmethod
    def executeObjectDetection(frame):
        """
        Performs object detection for the given frame and returns bounding boxes, labels, and scores for the detected objects.

        Inputs:
            frame (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
        Returns:
            boxes (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
            labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                bounding box is in the boxes array's same index.
            scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                score of the detected object at the same index in the boxes and labels arrays.

        Note that this method should return these in the following order as 'return boxes, labels, scores'.
        """
        pass
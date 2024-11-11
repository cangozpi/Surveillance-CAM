from abc import ABC, abstractmethod

class ObjectTrackingStrategy(ABC):
    """
    Implement this class to create an algorithm which can be used interchangeably with other algorithms to perform object tracking. 

    Design Pattern:
        Strategy interface of the Strategy design pattern (ObjectTrackingStrategy).
    """
    def __init__(self):
        pass

    @abstractmethod
    def executeObjectTracking(self, boxes, labels, scores):
        """
        Given detected objects, performs object tracking and associates unique ids (tracker_ids) with the detected objects.

        Inputs:
                boxes (np.ndarray of shape [N, x1, y1, x2, y2]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                    corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
                labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                    bounding box is in the boxes array's same index.
                scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                    score of the detected object at the same index in the boxes and labels arrays.

        Returns: returns the filtered detections
                boxes (np.ndarray of shape [N, x1, y1, x2, y2]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                    corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
                labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                    bounding box is in the boxes array's same index.
                scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                    score of the detected object at the same index in the boxes and labels arrays.
                tracker_ids (np.ndarray of shape [N,]): unique id's assigned by the object trackers. Note that id's are assigned starting from 0 again after no object
                    tracker remains due to all trackers being deleted due to various reasons such as inacitivity.

        Note that this method should return these in the following order as 'return boxes, labels, scores, tracker_ids'.
        """
        pass
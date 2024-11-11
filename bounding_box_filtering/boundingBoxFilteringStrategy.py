from abc import ABC, abstractmethod

class BoundingBoxFilteringStrategy(ABC):
    """
    Implement this class to create an algorithm which can be used interchangeably with other algorithms to perform bounding box filtering 
    as a post-processing step for object detection in images. 

    Design Pattern:
        Strategy interface of the Strategy design pattern (BoundingBoxFilteringStrategy).
    """
    def __init__(self):
        pass

    @abstractmethod
    def executeBoundingBoxFiltering(self, boxes, labels, scores, iou_threshold):
        """
        Filters the detected objects (corresponding bounding boxes, labels, and confidence scores) and returns the filtered results.

        Inputs:
            boxes (np.ndarray of shape [N, x1, y1, x2, y2]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
            labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                bounding box is in the boxes array's same index.
            scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                score of the detected object at the same index in the boxes and labels arrays.
            iou_threshold (float): Intersection Over Uniont (IOU) threshold value which is passed to the BoundingBoxFilteringStrategy's execute()
                method. Whether this value would be used depends solely on the BoundingBoxFilteringStrategy class you chose to use.
        
        Returns: returns the filtered detections
            boxes (np.ndarray of shape [N, x1, y1, x2, y2]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
            labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                bounding box is in the boxes array's same index.
            scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                score of the detected object at the same index in the boxes and labels arrays.
        Note: as an edge case returning a single detection as boxes.shape == (4,) with selected_scores as a single np.value is ok.

        Note that this method should return these in the following order as 'return boxes, labels, scores'.
        """
        pass
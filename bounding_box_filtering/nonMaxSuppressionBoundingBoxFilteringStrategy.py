from bounding_box_filtering.boundingBoxFilteringStrategy import BoundingBoxFilteringStrategy
import torchvision
import torch

class NonMaxSuppressionBoundingBoxFilteringStrategy(BoundingBoxFilteringStrategy):
    """
    Concrete implementation of the BoundingBoxFilteringStrategy (Strategy pattern) which can be used by the BoundingBoxFilter. 
    Performs bounding box filtering using the non-maximum suppression (NMS) algorithm on the bounding box outputs of the object detector.

    Design Pattern:
        Concrete implementation of the Strategy interface of the Strategy design pattern (BoundingBoxFilteringStrategy).
    """
    def __init__(self):
        pass

    def executeBoundingBoxFiltering(self, boxes, labels, scores, iou_threshold):
        """ 
        Perform non-maximum suppression (NMS) to the object detection's bounding boxes.

        Inputs:
            boxes (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
            labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                bounding box is in the boxes array's same index.
            scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                score of the detected object at the same index in the boxes and labels arrays.
            iou_threshold (float): Intersection Over Uniont (IOU) threshold value which is passed to the BoundingBoxFilteringStrategy's execute()
                method. Whether this value would be used depends solely on the BoundingBoxFilteringStrategy class you chose to use.
        
        Returns: returns the filtered detections
            boxes (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
            labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                bounding box is in the boxes array's same index.
            scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                score of the detected object at the same index in the boxes and labels arrays.
        """

        # Apply NMS
        indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)

        # Get the selected boxes and scores
        selected_boxes = boxes[indices]
        selected_scores = scores[indices]
        selected_labels = [labels[i] for i in indices.detach().cpu().numpy()]

        return selected_boxes, selected_labels, selected_scores
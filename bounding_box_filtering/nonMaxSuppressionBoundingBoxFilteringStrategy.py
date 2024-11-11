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
        """ Perform non-maximum suppression (NMS) to the object detection's bounding boxes."""
        # Apply NMS
        indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)

        # Get the selected boxes and scores
        selected_boxes = boxes[indices]
        selected_scores = scores[indices]
        selected_labels = [labels[i] for i in indices.detach().cpu().numpy()]

        return selected_boxes, selected_labels, selected_scores
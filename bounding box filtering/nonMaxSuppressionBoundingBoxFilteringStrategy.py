from boundingBoxFilteringStrategy import BoundingBoxFilteringStrategy

class NonMaxSuppressionBoundingBoxFilteringStrategy(BoundingBoxFilteringStrategy):
    """
    Concrete implementation of the BoundingBoxFilteringStrategy (Strategy pattern) which can be used by the BoundingBoxFilter. 
    Performs bounding box filtering using the non-maximum suppression (NMS) algorithm on the bounding box outputs of the object detector.

    Design Pattern:
        Concrete implementation of the Strategy interface of the Strategy design pattern (BoundingBoxFilteringStrategy).
    """
    def __init__(self):
        pass

    def executeBoundingBoxFiltering(frame):
        # TODO: perform non-maximum suppression to the object detection's bounding boxes
        pass
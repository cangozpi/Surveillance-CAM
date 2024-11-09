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
    def executeBoundingBoxFiltering(frame):
        pass
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
        pass
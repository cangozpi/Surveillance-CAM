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
    def executeObjectTracking(frame):
        pass
from abc import ABC, abstractmethod

class ChangeDetectionStrategy(ABC):
    """
    Implement this class to create an algorithm which can be used interchangeably with other algorithms to detect changes in an image
    which might be worthy of further investigation.

    Design Pattern:
        Strategy interface of the Strategy design pattern (ChangeDetectionStrategy).
    """
    def __init__(self):
        pass

    @abstractmethod
    def executeChangeDetection(frame):
        pass
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
    def executeChangeDetection(self, frame):
        """
        Takes in a single 2D BGR image frame of shape [H, W, C=3], and tries to detect whether a change in the scene has happened.
        The detection of a change should signal that a further investigation of the image should be done by passing the request further
        down the CV Pipeline, else we do not continue further passing the CV Pipeline for the given frame. 
        For example, a change might be detected when a new object has entered the scene which might indicate that we need to run
        object detector on the given frame. A no change might indicate that no object of interest to us is currently in the scene,
        and by not processing this frame further via passing the request down the CV Pipeline we avoid computational costs and time intensive computing.

        Inputs:
            frame (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3].
        
        Returns:
            is_change_detected (bool): True if a change in the scene is detected, else False.
        """
        pass
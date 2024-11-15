from abc import ABC, abstractmethod

class NotificationStrategy(ABC):
    """
    Implement this class to create an algorithm which can be used interchangeably with other algorithms to notify the user.

    Design Pattern:
        Strategy interface of the Strategy design pattern (ChangeDetectionStrategy).
    """
    def __init__(self):
        pass

    @abstractmethod
    def executeNotify(self, msg: str, frame):
        """
        Notifies the user about the given info.
        Inputs:
            msg (str): text message
            frame (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
        """
        pass
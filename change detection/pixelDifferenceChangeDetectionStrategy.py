from changeDetectionStrategy import ChangeDetectionStrategy

class PixelDifferenceChangeDetectionStrategy(ChangeDetectionStrategy):
    """
    Concrete implementation of the ChangeDetectionStrategy (Strategy pattern) which can be used by the ChangeDetector. 
    Performs change detection by taking the difference between the pixels in a given image and comparing it against a threshold.

    Design Pattern:
        Concrete implementation of the Strategy interface of the Strategy design pattern (ChangeDetectionStrategy).
    """
    def __init__(self):
        pass

    def executeChangeDetection(frame):
        # TODO: perform change detection via pixel difference here
        pass
from enum import Enum

class Available_ChangeDetectionStrategies(Enum):
    """
    Enums used by the IPipelineBuilder to set the ChangeDetectionStrategy of the ChangeDetector
    accordingly from the available implementations of the ChangeDetectionStrategy. There should
    be an enum for every concrete implementation of the ChangeDetectionStrategy.
    """
    PIXEL_DIFFERENCE_CHANGE_DETECTION_STRATEGY = 1

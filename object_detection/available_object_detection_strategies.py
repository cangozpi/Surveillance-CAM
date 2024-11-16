from enum import Enum

class Available_ObjectDetectionStrategies(Enum):
    """
    Enums used by the IPipelineBuilder to set the ObjectDetectionStrategy of the ObjectDetector
    accordingly from the available implementations of the ObjectDetectionStrategy. There should
    be an enum for every concrete implementation of the ObjectDetectionStrategy.
    """
    MOBILENET_OBJECT_DETECTION_STRATEGY = 1

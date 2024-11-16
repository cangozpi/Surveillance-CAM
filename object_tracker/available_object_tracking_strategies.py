from enum import Enum

class Available_ObjectTrackingStrategies(Enum):
    """
    Enums used by the IPipelineBuilder to set the ObjectTrackingStrategy of the ObjectTracker
    accordingly from the available implementations of the ObjectTrackingStrategy. There should
    be an enum for every concrete implementation of the ObjectTrackingStrategy.
    """
    SORT_OBJECT_TRACKING_STRATEGY = 1

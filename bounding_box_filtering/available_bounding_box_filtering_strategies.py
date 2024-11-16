from enum import Enum

class Available_BoundingBoxFilteringStrategies(Enum):
    """
    Enums used by the IPipelineBuilder to set the ObjectDetectionStrategy of the BoundingBoxFilter
    accordingly from the available implementations of the BoundingBoxFilteringStrategy. There should
    be an enum for every concrete implementation of the BoundingBoxFilteringStrategy.
    """
    NON_MAX_SUPPRESSION_BOUNDING_BOX_FILTERING_STRATEGY = 1

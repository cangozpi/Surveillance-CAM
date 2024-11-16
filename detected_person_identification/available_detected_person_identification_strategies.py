from enum import Enum

class Available_DetectedPersonIdentificationStrategies(Enum):
    """
    Enums used by the IPipelineBuilder to set the DetectedPersonIdentificationStrategy of the DetectedPersonIdentifier
    accordingly from the available implementations of the DetectedPersonIdentificationStrategy. There should
    be an enum for every concrete implementation of the DetectedPersonIdentificationStrategy.
    """
    KNN_PERSON_IDENTIFICATION_STRATEGY = 1

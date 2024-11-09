from detectedPersonIdentificationStrategy import DetectedPersonIdentificationStrategy

class KNN_DetectedPersonIdentificationStrategy(DetectedPersonIdentificationStrategy):
    """
    Concrete implementation of the DetectedPersonIdentificationStrategy (Strategy pattern) which can be used by the DetectedPersonIdentifier. 
    Performs person identification of detected people by comparing their embeddings vectors against the embedding vectors of the known identities
    in a given identity database. It takes a KNN approach and matches the people with the class with the closest embedding within a threshold value.
    If the distance between the closest embeddings exceeds a threshold then the person is identified as an unknown person.

    Design Pattern:
        Concrete implementation of the Strategy interface of the Strategy design pattern (DetectedPersonIdentificationStrategy).
    """
    def __init__(self):
        pass

    def executePersonIdentification(frame):
        # TODO: perform object tracking using SORT
        pass
from abc import ABC, abstractmethod

class DetectedPersonIdentificationStrategy(ABC):
    """
    Implement this class to create an algorithm which can be used interchangeably with other algorithms to perform person identification of 
    people detected on an image by comparing them against the entries in an available database of identities. 

    Design Pattern:
        Strategy interface of the Strategy design pattern (DetectedPersonIdentificationStrategy).
    """
    def __init__(self):
        pass

    @abstractmethod
    def executePersonIdentification(frame):
        pass
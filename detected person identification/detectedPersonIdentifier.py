from detectedPersonIdentificationStrategy import DetectedPersonIdentificationStrategy
from CVPipelineStep import CVPipelineStep

class DetectedPersonIdentifier(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to identify people detected on an image by comparing them 
    against the entries in an available database of identities.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform object tracking.

    Design Pattern:
        - Context of the Strategy design pattern (DetectedPersonIdentificaitonStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self):
        self.detectedPersonIdentificationStrategy: DetectedPersonIdentificationStrategy = None # strategy to use for identifying people detected on an image
    
    def setPersonIdentificationStrategy(self, strategy: DetectedPersonIdentificationStrategy):
        self.detectedPersonIdentificationStrategy = DetectedPersonIdentificationStrategy
    
    def identifyPeople(self, frame):
        return self.detectedPersonIdentificationStrategy.executePersonIdentification(frame)
    
    def handle(self, request): # handle fn of the chain of responsibility pattern (CVPipelineStep)
        # TODO:
        # perform person identifivaiton by a KNN approach via self.identifyPeople(frame)
        # decide to pass down the chain or not via self.next.handle(request)
        pass
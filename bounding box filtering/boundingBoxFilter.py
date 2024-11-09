from boundingBoxFilteringStrategy import BoundingBoxFilteringStrategy
from CVPipelineStep import CVPipelineStep

class BoundingBoxFilter(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to filter bounding boxes as a post-processing step to the outputs of the object detector.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform bounding box filtering post-processing.

    Design Pattern:
        - Context of the Strategy design pattern (BoundingBoxFilteringStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self):
        self.boundingBoxFilteringStrategy: BoundingBoxFilteringStrategy = None # strategy to use for filtering the bounding box outputs of the object detector
    
    def setBoundingBoxFilteringStrategy(self, strategy: BoundingBoxFilteringStrategy):
        self.boundingBoxFilteringStrategy = BoundingBoxFilteringStrategy
    
    def filterBoundingBoxes(self, frame):
        return self.boundingBoxFilteringStrategy.executeBoundingBoxFiltering(frame)
    
    def handle(self, request): # handle fn of the chain of responsibility pattern (CVPipelineStep)
        # TODO:
        # filter BBs via fiterBoundingBoxes(frame)
        # decide to pass down the chain or not via self.next.handle(request)
        pass
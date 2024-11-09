from changeDetectionStrategy import ChangeDetectionStrategy
from CVPipelineStep import CVPipelineStep

class ChangeDetector(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to detect changes in an image which might be worthy of further investigation.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform change detection.

    Design Pattern:
        - Context of the Strategy design pattern (ChangeDetectionStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self):
        self.changeDetectionStrategy: ChangeDetectionStrategy = None # strategy to use for detecting changes
    
    def setChangeDetectionStrategy(self, strategy: ChangeDetectionStrategy):
        self.changeDetectionStrategy = ChangeDetectionStrategy
    
    def detectChange(self, frame):
        return self.changeDetectionStrategy.executeChangeDetection(frame)
    
    def handle(self, request): # handle fn of the chain of responsibility pattern (CVPipelineStep)
        # TODO:
        # detect change via detectChange()
        # decide to pass down the chain or not via self.next.handle(request)
        pass
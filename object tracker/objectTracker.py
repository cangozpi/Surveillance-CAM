from objectTrackingStrategy import ObjectTrackingStrategy
from CVPipelineStep import CVPipelineStep

class ObjectTracker(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to track objects in an image.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform object tracking.

    Design Pattern:
        - Context of the Strategy design pattern (ObjectTrackingStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self):
        self.objectTrackingStrategy: ObjectTrackingStrategy = None # strategy to use for tracking objects
    
    def setObjectTrackingStrategy(self, strategy: ObjectTrackingStrategy):
        self.objectTrackingStrategy = ObjectTrackingStrategy
    
    def trackObjects(self, frame):
        return self.objectTrackingStrategy.executeObjectTracking(frame)
    
    def handle(self, request): # handle fn of the chain of responsibility pattern (CVPipelineStep)
        # TODO:
        # perform object tracking via self.trackObjects(frame)
        # decide to pass down the chain or not via self.next.handle(request)
        pass
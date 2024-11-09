from objectDetectionStrategy import ObjectDetectionStrategy
from CVPipelineStep import CVPipelineStep

class ObjectDetector(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to detect objects in an image.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform object detection.

    Design Pattern:
        - Context of the Strategy design pattern (ObjectDetectionStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self):
        self.objectDetectionStrategy: ObjectDetectionStrategy = None # strategy to use for detecting objects
    
    def setObjectDetectionStrategy(self, strategy: ObjectDetectionStrategy):
        self.objectDetectionStrategy = ObjectDetectionStrategy
    
    def detectObjects(self, frame):
        return self.objectDetectionStrategy.executeObjectDetection(frame)
    
    def handle(self, request): # handle fn of the chain of responsibility pattern (CVPipelineStep)
        # TODO:
        # detect objects via detectObjects(frame)
        # decide to pass down the chain or not via self.next.handle(request)
        pass
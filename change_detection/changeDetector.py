from change_detection.changeDetectionStrategy import ChangeDetectionStrategy
from CV_pipeline.CVPipelineStep import CVPipelineStep
import numpy as np
from copy import deepcopy

class ChangeDetector(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to detect changes in an image which might be worthy of further investigation.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform change detection.

    Design Pattern:
        - Context of the Strategy design pattern (ChangeDetectionStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self):
        super().__init__()
        self.changeDetectionStrategy: ChangeDetectionStrategy = None # strategy to use for detecting changes
    
    def setChangeDetectionStrategy(self, strategy: ChangeDetectionStrategy):
        assert(isinstance(strategy, ChangeDetectionStrategy))
        self.changeDetectionStrategy = strategy
    
    def detectChange(self, frame):
        """
        Returns True if change is detected in a given frame, else returns False.
        """
        return self.changeDetectionStrategy.executeChangeDetection(frame=frame)
    
    def handle(self, request): # handle fn of the chain of responsibility pattern (CVPipelineStep)
        """
        Inputs:
            request (dict): {
                                'frame' (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
                            }
        
        - Calls self.next.handle() with the following params:
            {
                'frame' (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
            }
        """
        assert((type(request)) == dict and ('frame' in request))

        frame = deepcopy(request['frame'])
        assert(isinstance(frame, np.ndarray) and (len(frame.shape) == 3) and (frame.shape[2] == 3))

        # detect change via detectChange()
        is_change_detected = self.detectChange(frame)
        assert(isinstance(is_change_detected, bool))
        print(f'[ChangeDetector] is_change_detected: {is_change_detected}')

        # decide whether to pass down the chain or not
        if is_change_detected and (self.next is not None):
            self.next.handle({
                'frame': request['frame']
            })
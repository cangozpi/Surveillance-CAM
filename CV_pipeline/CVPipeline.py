from CV_pipeline import CVPipelineStep

class CVPipeline: 
    """
    This class abstracts the chain of responsibility pattern used to create the CV_Pipeline.
    """
    def __init__(self):
        self.cv_PipelineHead: CVPipelineStep = None # points to the head of the CV_Pipeline
        self.cv_PipelineTail: CVPipelineStep = None # points to the tail (end) of the CV_Pipeline
    
    def appendPipelineStep(self, cvPipelineStep: CVPipelineStep):
        """
        Appends the given CVPipelineStep to the end of the Pipeline.
        """
        if self.cv_PipelineHead is None:
            self.cv_PipelineHead = cvPipelineStep
            self.cv_PipelineTail = cvPipelineStep
        else:
            self.cv_PipelineTail.next = cvPipelineStep
            self.cv_PipelineTail = cvPipelineStep
    
    def removePipelineStep(self, cvPipelineStepClass: CVPipelineStep):
        """
        If any PipelineStep of type CVPipelineStepClass exists in the Pipeline then its first instance is removed from the Pipeline. 
        If it does not exist then nothing is done.
        """
        cur_PipelineStep = self.cv_PipelineHead
        if isinstance(cur_PipelineStep, cvPipelineStepClass): # handle if it is the head
            self.cv_PipelineHead = cur_PipelineStep.next
        while cur_PipelineStep.next is not None: # if it is not the head then search for it
            if isinstance(cur_PipelineStep.next, cvPipelineStepClass):
                cur_PipelineStep.next = cur_PipelineStep.next.next
                break
            else:
                cur_PipelineStep = cur_PipelineStep.next
    
    def handle_CVPipeline(self, initial_request):
        """
        Starts the handling of the request through the chaing by passing the given initial_request down the first step in the CV Pipeline.
        In other words it calls the handle() of the first step in the Chain of Responsibility design pattern which will 
        then make the subsequent calls recursively.
        """
        if self.cv_PipelineHead is not None:
            self.cv_PipelineHead.handle(initial_request)

    
    def getCVPipeline(self):
        """
        Returns the head of the CV Pipeline.
        """
        return self.cv_PipelineHead
    
    def resetCVPipeline(self):
        """
        Clears all the elements from the CV Pipeline.
        """
        self.cv_PipelineHead: CVPipelineStep = None # points to the head of the CV_Pipeline
        self.cv_PipelineTail: CVPipelineStep = None # points to the tail (end) of the CV_Pipeline
        


        
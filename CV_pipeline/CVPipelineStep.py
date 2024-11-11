from abc import ABC, abstractmethod

class CVPipelineStep:
    """
    Inherit from this class to create a method that can be added as a step to the Computer Vision (CV) Pipeline.

    Design Pattern:
        Handler interface of the Chain of Responsibility design pattern.
    """
    def __init__(self):
        self.next: CVPipelineStep = None
    
    def setNext(handler):
        self.next = handler # handler: CVPipelineStep
    
    @abstractmethod
    def handle(request):
        pass

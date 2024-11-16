from abc import ABC, abstractmethod
from change_detection.available_change_detection_strategies import Available_ChangeDetectionStrategies
from object_detection.available_object_detection_strategies import Available_ObjectDetectionStrategies
from bounding_box_filtering.available_bounding_box_filtering_strategies import Available_BoundingBoxFilteringStrategies
from object_tracker.available_object_tracking_strategies import Available_ObjectTrackingStrategies
from detected_person_identification.available_detected_person_identification_strategies import Available_DetectedPersonIdentificationStrategies
from notification.available_notification_strategies import Available_NotificationStrategies
from CV_pipeline.CVPipeline import CVPipeline

class IPipelineBuilder(ABC):
    """
    Handles the building of the CV_Pipeline with the appropriate order of the CV_PipelineSteps (Chain of Responsibility design pattern), 
    and the desired strategies (Strategy Design Pattern) used by those steps. Aslo handles the run-time modifications to these 
    strategies.

    Design Pattern:
        - Builder interface of the Builder design pattern.
    """
    def __init__(self):
        self.cv_Pipeline:CVPipeline = None
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def setChangeDetector(self, strategy_type: Available_ChangeDetectionStrategies, **kwargs):
        """
        Sets the ChangeDetector of the CV_Pipeline according to the given strategy_type.
        If the current cv_Pipeline does not have a ChangeDetector PipelineStep already initialized, this method 
        creates and sets it. If the current cv_Pipeline does already have a ChangeDetector PipelineStep, then this
        method changes its ChangeDetectionStrategy according to the passed in strategy_type.

        Inputs:
            strategy_type (Available_ChangeDetectionStrategies): signifies the ChangeDetectionStrategy to be set
                for the current cv_Pipeline.
            **kwargs: this is should contain all the necessary keyword arguments to create both the ChangeDetector and its ChangeDetectionStrategy.
        """
        pass

    @abstractmethod
    def setObjectDetector(self, strategy_type: Available_ObjectDetectionStrategies, **kwargs):
        """
        Sets the ObjectDetector of the CV_Pipeline according to the given strategy_type.
        If the current cv_Pipeline does not have a ObjectDetector PipelineStep already initialized, this method 
        creates and sets it. If the current cv_Pipeline does already have a ObjectDetector PipelineStep, then this
        method changes its ObjectDetectionStrategy according to the passed in strategy_type.

        Inputs:
            strategy_type (Available_ObjectDetectionStrategies): signifies the ObjectDetectionStrategy to be set
                for the current cv_Pipeline.
            **kwargs: this is should contain all the necessary keyword arguments to create both the ObjectDetector and its ObjectDetectionStrategy.
        """
        pass

    @abstractmethod
    def setBoundingBoxFilter(self, strategy_type: Available_BoundingBoxFilteringStrategies, **kwargs):
        """
        Sets the BoundingBoxFilter of the CV_Pipeline according to the given strategy_type.
        If the current cv_Pipeline does not have a BoundingBoxFilter PipelineStep already initialized, this method 
        creates and sets it. If the current cv_Pipeline does already have a BoundingBoxFilter PipelineStep, then this
        method changes its BoundingBoxFilteringStrategy according to the passed in strategy_type.

        Inputs:
            strategy_type (Available_BoundingBoxFilteringStrategies): signifies the BoundingBoxFilteringStrategy to be set
                for the current cv_Pipeline.
            **kwargs: this is should contain all the necessary keyword arguments to create both the BoundingBoxFilter and 
                its BoundingBoxFilteringStrategy.
        """
        pass

    @abstractmethod
    def setObjectTracker(self, strategy_type: Available_ObjectTrackingStrategies, **kwargs):
        """
        Sets the ObjectTracker of the CV_Pipeline according to the given strategy_type.
        If the current cv_Pipeline does not have a ObjectTracker PipelineStep already initialized, this method 
        creates and sets it. If the current cv_Pipeline does already have a ObjectTracker PipelineStep, then this
        method changes its ObjectTrackingStrategy according to the passed in strategy_type.

        Inputs:
            strategy_type (Available_ObjectTrackingStrategies): signifies the ObjectTrackingStrategy to be set
                for the current cv_Pipeline.
            **kwargs: this is should contain all the necessary keyword arguments to create both the ObjectTracker and 
                its ObjectTrackingStrategy.
        """
        pass

    @abstractmethod
    def setDetectedPersonIdentifier(self, strategy_type: Available_DetectedPersonIdentificationStrategies, **kwargs):
        """
        Sets the DetectedPersonIdentifier of the CV_Pipeline according to the given strategy_type.
        If the current cv_Pipeline does not have a DetectedPersonIdentifier PipelineStep already initialized, this method 
        creates and sets it. If the current cv_Pipeline does already have a DetectedPersonIdentifier PipelineStep, then this
        method changes its DetectedPersonIdentificationStrategy according to the passed in strategy_type.

        Inputs:
            strategy_type (Available_DetectedPersonIdentificationStrategies): signifies the DetectedPersonIdentificationStrategy to be set
                for the current cv_Pipeline.
            **kwargs: this is should contain all the necessary keyword arguments to create both the DetectedPersonIdentifier and 
                its DetectedPersonIdentificationStrategy.
        """
        pass

    @abstractmethod
    def setNotifier(self, strategy_type: Available_NotificationStrategies, **kwargs):
        """
        Sets the Notifier of the CV_Pipeline according to the given strategy_type.
        If the current cv_Pipeline does not have a Notifier PipelineStep already initialized, this method 
        creates and sets it. If the current cv_Pipeline does already have a Notifier PipelineStep, then this
        method changes its NotificationStrategy according to the passed in strategy_type.

        Inputs:
            strategy_type (Available_NotificationStrategies): signifies the NotificationStrategy to be set
                for the current cv_Pipeline.
            **kwargs: this is should contain all the necessary keyword arguments to create both the Notifier and 
                its NotificationStrategy.
        """
        pass

from pipeline_builder.IPipelineBuilder import IPipelineBuilder
from CV_pipeline.CVPipelineStep import CVPipelineStep
from CV_pipeline.CVPipeline import CVPipeline

from change_detection.available_change_detection_strategies import Available_ChangeDetectionStrategies
from object_detection.available_object_detection_strategies import Available_ObjectDetectionStrategies
from bounding_box_filtering.available_bounding_box_filtering_strategies import Available_BoundingBoxFilteringStrategies
from object_tracker.available_object_tracking_strategies import Available_ObjectTrackingStrategies
from detected_person_identification.available_detected_person_identification_strategies import Available_DetectedPersonIdentificationStrategies
from notification.available_notification_strategies import Available_NotificationStrategies

from change_detection.changeDetector import ChangeDetector
from change_detection.changeDetectionStrategy import ChangeDetectionStrategy
from change_detection.pixelDifferenceChangeDetectionStrategy import PixelDifferenceChangeDetectionStrategy
from object_detection.objectDetector import ObjectDetector
from object_detection.objectDetectionStrategy import ObjectDetectionStrategy
from object_detection.mobileNetObjectDetectionStrategy import MobileNetObjectDetectionStrategy
from bounding_box_filtering.boundingBoxFilter import BoundingBoxFilter
from bounding_box_filtering.boundingBoxFilteringStrategy import BoundingBoxFilteringStrategy
from bounding_box_filtering.nonMaxSuppressionBoundingBoxFilteringStrategy import NonMaxSuppressionBoundingBoxFilteringStrategy
from object_tracker.objectTracker import ObjectTracker
from object_tracker.objectTrackingStrategy import ObjectTrackingStrategy
from object_tracker.SORT_ObjectTrackingStrategy import SORT_ObjectTrackingStrategy
from detected_person_identification.detectedPersonIdentifier import DetectedPersonIdentifier
from detected_person_identification.detectedPersonIdentificationStrategy import DetectedPersonIdentificationStrategy
from detected_person_identification.KNN_PersonIdentificationStrategy import KNN_DetectedPersonIdentificationStrategy
from detected_person_identification.knownPersonDb import KnownPersonDb
from detected_person_identification.knownPersonDb import KnownPersonFolderDb
from notification.notifier import Notifier
from notification.notificationStrategy import NotificationStrategy
from notification.emailNotificationStrategy import EmailNotificationStrategy

class PipelineBuilder(IPipelineBuilder):
    """
    A concrete implementation of the IPipelineBuilder.
    Handles the building of the CV_Pipeline with the appropriate order of the CV_PipelineSteps (Chain of Responsibility design pattern), 
    and the desired strategies (Strategy Design Pattern) used by those steps. Aslo handles the run-time modifications to these 
    strategies.

    Design Pattern:
        - Concrete implementation of the builder interface of the Builder design pattern. (IPipelineBuilder)
    """
    def __init__(self):
        super().__init__()
        self.cv_Pipeline = CVPipeline()

    def reset(self):
        self.cv_Pipeline = CVPipeline()

    def build(self):
        """
        Returns:
            cv_pipeline (CVPipeLine): the built CVPipeline
        """
        return self.cv_Pipeline

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
        assert(isinstance(strategy_type, Available_ChangeDetectionStrategies))

        if strategy_type == Available_ChangeDetectionStrategies.PIXEL_DIFFERENCE_CHANGE_DETECTION_STRATEGY:
            # Create the Strategy to set
            pixelDifferenceChangeDetectionStrategy: ChangeDetectionStrategy = PixelDifferenceChangeDetectionStrategy()

            # Check if ChangeDetector step has been set already
            step_ref: ChangeDetector = self.cv_Pipeline.getCVPipelineStepIfExists(ChangeDetector)
            
            # If ChangeDetector has not been set
            if step_ref is None:
                # Create a ChangeDetector, set its Strategy, and add it to the CV Pipeline:
                changeDetector: ChangeDetector = ChangeDetector()
                changeDetector.setChangeDetectionStrategy(pixelDifferenceChangeDetectionStrategy)

                self.cv_Pipeline.appendPipelineStep(changeDetector)
            else: # If ChangeDetector has been already set
                step_ref.setChangeDetectionStrategy(pixelDifferenceChangeDetectionStrategy)


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
        assert(isinstance(strategy_type, Available_ObjectDetectionStrategies))

        if strategy_type == Available_ObjectDetectionStrategies.MOBILENET_OBJECT_DETECTION_STRATEGY:
            # Create the Strategy to set
            mobileNetObjectDetectionStrategy: ObjectDetectionStrategy = MobileNetObjectDetectionStrategy()

            # Check if ObjectDetector step has been set already
            step_ref: ObjectDetector = self.cv_Pipeline.getCVPipelineStepIfExists(ObjectDetector)
            
            # If ObjectDetector has not been set
            if step_ref is None:
                # Create a ObjectDetector, set its Strategy, and add it to the CV Pipeline:
                objectDetector: ObjectDetector = ObjectDetector()
                objectDetector.setObjectDetectionStrategy(mobileNetObjectDetectionStrategy)


                self.cv_Pipeline.appendPipelineStep(objectDetector)
            else: # If ObjectDetector has been already set
                step_ref.setObjectDetectionStrategy(mobileNetObjectDetectionStrategy)

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
        assert(isinstance(strategy_type, Available_BoundingBoxFilteringStrategies))

        if strategy_type == Available_BoundingBoxFilteringStrategies.NON_MAX_SUPPRESSION_BOUNDING_BOX_FILTERING_STRATEGY:
            # Create the Strategy to set
            nonMaxSuppressionBoundingBoxFilteringStrategy: BoundingBoxFilteringStrategy = NonMaxSuppressionBoundingBoxFilteringStrategy()

            # Check if BoundingBoxFilter step has been set already
            step_ref: BoundingBoxFilter = self.cv_Pipeline.getCVPipelineStepIfExists(BoundingBoxFilter)
            
            # If BoundingBoxFilter has not been set
            if step_ref is None:
                # Create a ObjectDetector, set its Strategy, and add it to the CV Pipeline:
                boundingBoxFilter: BoundingBoxFilter = BoundingBoxFilter(**kwargs)
                boundingBoxFilter.setBoundingBoxFilteringStrategy(nonMaxSuppressionBoundingBoxFilteringStrategy)

                self.cv_Pipeline.appendPipelineStep(boundingBoxFilter)
            else: # If ObjectDetector has been already set
                step_ref.setBoundingBoxFilteringStrategy(nonMaxSuppressionBoundingBoxFilteringStrategy)

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
        assert(isinstance(strategy_type, Available_ObjectTrackingStrategies))

        if strategy_type == Available_ObjectTrackingStrategies.SORT_OBJECT_TRACKING_STRATEGY:
            # Create the Strategy to set
            sort_ObjectTrackingStrategy: ObjectTrackingStrategy = SORT_ObjectTrackingStrategy(**kwargs)

            # Check if ObjectTracker step has been set already
            step_ref: ObjectTracker = self.cv_Pipeline.getCVPipelineStepIfExists(ObjectTracker)
            
            # If ObjectTracker has not been set
            if step_ref is None:
                # Create a ObjectTracker, set its Strategy, and add it to the CV Pipeline:
                objectTracker: ObjectTracker = ObjectTracker()
                objectTracker.setObjectTrackingStrategy(sort_ObjectTrackingStrategy)

                self.cv_Pipeline.appendPipelineStep(objectTracker)
            else: # If ObjectTracker has been already set
                step_ref.setObjectTrackingStrategy(sort_ObjectTrackingStrategy)

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
        assert(isinstance(strategy_type, Available_DetectedPersonIdentificationStrategies))

        if strategy_type == Available_DetectedPersonIdentificationStrategies.KNN_PERSON_IDENTIFICATION_STRATEGY:
            # Create the Strategy to set
            knn_DetectedPersonIdentificationStrategy: DetectedPersonIdentificationStrategy = KNN_DetectedPersonIdentificationStrategy(**kwargs)

            # Check if DetectedPersonIdentifier step has been set already
            step_ref: DetectedPersonIdentifier = self.cv_Pipeline.getCVPipelineStepIfExists(DetectedPersonIdentifier)
            
            # If DetectedPersonIdentifier has not been set
            if step_ref is None:
                # Create a Detected Person Identifier, set its Strategy and the db it uses, and add it to the CV Pipeline:
                knownPersonFolderDb:KnownPersonDb = KnownPersonFolderDb()
                detectedPersonIdentifier: DetectedPersonIdentifier = DetectedPersonIdentifier(db=knownPersonFolderDb)
                detectedPersonIdentifier.setPersonIdentificationStrategy(knn_DetectedPersonIdentificationStrategy)

                self.cv_Pipeline.appendPipelineStep(detectedPersonIdentifier)
            else: # If DetectedPersonIdentifier has been already set
                step_ref.setPersonIdentificationStrategy(knn_DetectedPersonIdentificationStrategy)

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
        assert(isinstance(strategy_type, Available_NotificationStrategies))

        if strategy_type == Available_NotificationStrategies.EMAIL_NOTIFICATION_STRATEGY:
            # Create the Strategy to set
            emailNotificationStrategy: NotificationStrategy = EmailNotificationStrategy()

            # Check if Notifier step has been set already
            step_ref: Notifier = self.cv_Pipeline.getCVPipelineStepIfExists(Notifier)
            
            # If Notifier has not been set
            if step_ref is None:
                # Create a Notifier, set its Strategy, and add it to the CV Pipeline:
                notifier: Notifier = Notifier(**kwargs)
                notifier.setNotificationStrategy(emailNotificationStrategy)

                self.cv_Pipeline.appendPipelineStep(notifier)
            else: # If Notifier has been already set
                step_ref.setNotificationStrategy(emailNotificationStrategy)
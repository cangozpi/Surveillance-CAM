from pipeline_builder.IPipelineBuilder import IPipelineBuilder

from change_detection.available_change_detection_strategies import Available_ChangeDetectionStrategies
from object_detection.available_object_detection_strategies import Available_ObjectDetectionStrategies
from bounding_box_filtering.available_bounding_box_filtering_strategies import Available_BoundingBoxFilteringStrategies
from object_tracker.available_object_tracking_strategies import Available_ObjectTrackingStrategies
from detected_person_identification.available_detected_person_identification_strategies import Available_DetectedPersonIdentificationStrategies
from notification.available_notification_strategies import Available_NotificationStrategies

class PipelineDirector:
    """
    Director to be used with a concrete implementation of the IPipelineBuilder.
    It defines the order in which to call construction steps of the passed in IPipelineBuilder to build appropriate
    a CV Pipeline.

    Design Pattern:
        - Director for the builder interface of the Builder design pattern. (IPipelineBuilder)
    """
    def __init__(self, builder: IPipelineBuilder = None):
        self.pipelineBuilder: IPipelineBuilder = builder
    
    def changeBuilder(self, builder: IPipelineBuilder):
        assert(isinstance(builder, IPipelineBuilder))
        self.pipelineBuilder = builder

    def make(self, **kwargs):
        """
        Initializes and returns a CV Pipeline with the following CVPipelineSteps with the specified

        Strategies in the given order below:
            1. ChangeDetector --(ChangeDetectionStrategy)--> PixelDifferenceChangeDetectionStrategy
            2. ObjectDetector --(ObjectDetectionStrategy)--> MobileNetObjectDetectionStrategy
            3. BoundingBoxFilter --(BoundingBoxFilteringStrategy)--> NonMaxSuppressionBoundingBoxStrategy
            4. ObjectTracker --(ObjectTrackingStrategy)--> SORT_ObjectTrackingStrategy
            5. DetectedPersonIdentifier --(DetectedPersonIdentificationStrategy)--> KNN_PersonIdentificationStrategy
            6. Notifier --(NotificationStrategy)--> EmailNotificationStrategy

        Inputs:
            **kwargs: any parameter required to create the CV Pipeline with the corresponding CVPipelineSteps and Strategies should be
                pass to this function using this argument.
        
        Returns:
            cv_pipeline (CVPipeline): created CVPipeline.
        """
        # 1. Set the ChangeDetector
        self.pipelineBuilder.setChangeDetector(Available_ChangeDetectionStrategies.PIXEL_DIFFERENCE_CHANGE_DETECTION_STRATEGY)

        # 2. Set the ObjectDetector
        self.pipelineBuilder.setObjectDetector(Available_ObjectDetectionStrategies.MOBILENET_OBJECT_DETECTION_STRATEGY)

        # 3. Set the BoundingBoxFilter
        self.pipelineBuilder.setBoundingBoxFilter(Available_BoundingBoxFilteringStrategies.NON_MAX_SUPPRESSION_BOUNDING_BOX_FILTERING_STRATEGY, conf_threshold=kwargs['conf_threshold'], iou_threshold=kwargs['iou_threshold']) 

        # 4. Set the ObjectTracker
        self.pipelineBuilder.setObjectTracker(Available_ObjectTrackingStrategies.SORT_OBJECT_TRACKING_STRATEGY, 
            max_age=kwargs['max_age'],
            min_hits=kwargs['min_hits'], 
            iou_threshold=kwargs['iou_threshold'])

        # 5. Set the DetectedPersonIdentifier
        self.pipelineBuilder.setDetectedPersonIdentifier(Available_DetectedPersonIdentificationStrategies.KNN_PERSON_IDENTIFICATION_STRATEGY, threshold_norm_dist=kwargs['threshold_norm_dist']) 

        # 6. Set the Notifier
        self.pipelineBuilder.setNotifier(Available_NotificationStrategies.EMAIL_NOTIFICATION_STRATEGY, notifierHistory_windowSize=kwargs['notifierHistory_windowSize'], forgetting_threshold=kwargs['forgetting_threshold']) 

        return self.pipelineBuilder.build()
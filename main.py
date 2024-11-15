from time import sleep
from videoStream import VideoStream
from CV_pipeline.CVPipelineStep import CVPipelineStep
from CV_pipeline.CVPipeline import CVPipeline
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

#-----  Manually create the CV Pipeline:
cv_Pipeline: CVPipeline = CVPipeline()

# Create a Change Detector, set its Strategy:
changeDetector: ChangeDetector = ChangeDetector()
pixelDifferenceChangeDetectionStrategy: ChangeDetectionStrategy = PixelDifferenceChangeDetectionStrategy()
changeDetector.setChangeDetectionStrategy(pixelDifferenceChangeDetectionStrategy)

cv_Pipeline.appendPipelineStep(changeDetector)


# Create a Object Detector, set its Strategy, and add it to the CV Pipeline:
objectDetector: ObjectDetector = ObjectDetector()
mobileNetObjectDetectionStrategy: ObjectDetectionStrategy = MobileNetObjectDetectionStrategy()
objectDetector.setObjectDetectionStrategy(mobileNetObjectDetectionStrategy)

cv_Pipeline.appendPipelineStep(objectDetector)


# Create a Bounding Box Filter, set its Strategy, and add it to the CV Pipeline:
boundingBoxFilter: BoundingBoxFilter = BoundingBoxFilter(conf_threshold=0.2, iou_threshold=0.5) #TODO: Set these constructor parameters dynamically
nonMaxSuppressionBoundingBoxFilteringStrategy: BoundingBoxFilteringStrategy = NonMaxSuppressionBoundingBoxFilteringStrategy()
boundingBoxFilter.setBoundingBoxFilteringStrategy(nonMaxSuppressionBoundingBoxFilteringStrategy)

cv_Pipeline.appendPipelineStep(boundingBoxFilter)


# Create a Object Tracker, set its Strategy, and add it to the CV Pipeline:
objectTracker: ObjectTracker = ObjectTracker()
sort_ObjectTrackingStrategy: ObjectTrackingStrategy = SORT_ObjectTrackingStrategy(max_age=1, min_hits=5, iou_threshold=0.3)
objectTracker.setObjectTrackingStrategy(sort_ObjectTrackingStrategy)

cv_Pipeline.appendPipelineStep(objectTracker)


# Create a Detected Person Identifier, set its Strategy and the db it uses, and add it to the CV Pipeline:
knownPersonFolderDb:KnownPersonDb = KnownPersonFolderDb()
detectedPersonIdentifier: DetectedPersonIdentifier = DetectedPersonIdentifier(db=knownPersonFolderDb)
knn_DetectedPersonIdentificationStrategy: DetectedPersonIdentificationStrategy = KNN_DetectedPersonIdentificationStrategy(threshold_norm_dist=100) #TODO: tune this threshold value dynamically
detectedPersonIdentifier.setPersonIdentificationStrategy(knn_DetectedPersonIdentificationStrategy)

cv_Pipeline.appendPipelineStep(detectedPersonIdentifier)



# ----- Simulate video stream and the CV Pipeline:
video_path = './test/test video.mp4'
videoStream = VideoStream(video_path)
fps = 1/30

while True:
    frame = videoStream.getNextFrame()

    if frame is not None:
        # VideoStream.displayFrame(frame)
        sleep(fps)

        # pass the frame through the CV Pipeline
        cv_Pipeline.handle_CVPipeline(initial_request={
            'frame': frame
        })
    else:
        exit()


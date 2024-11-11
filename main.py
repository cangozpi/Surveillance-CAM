from time import sleep
from videoStream import VideoStream
from CV_pipeline.CVPipelineStep import CVPipelineStep
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

#-----  Manually create the CV Pipeline:
cv_PipelineHead: CVPipelineStep = None
cv_PipelineTail: CVPipelineStep = None

# Create a Change Detector, set its Strategy:
pixelDifferenceChangeDetectionStrategy: ChangeDetectionStrategy = PixelDifferenceChangeDetectionStrategy()
changeDetector: ChangeDetector = ChangeDetector()
changeDetector.setChangeDetectionStrategy(pixelDifferenceChangeDetectionStrategy)

cv_PipelineHead = changeDetector # set the head of the CV Pipeline
cv_PipelineTail = changeDetector


# Create a Object Detector, set its Strategy, and add it to the CV Pipeline:
mobileNetObjectDetectionStrategy: ObjectDetectionStrategy = MobileNetObjectDetectionStrategy()
objectDetector: ObjectDetector = ObjectDetector()
objectDetector.setObjectDetectionStrategy(mobileNetObjectDetectionStrategy)

cv_PipelineTail.next = objectDetector
cv_PipelineTail = objectDetector


# Create a Bounding Box Filter, set its Strategy, and add it to the CV Pipeline:
nonMaxSuppressionBoundingBoxFilteringStrategy: BoundingBoxFilteringStrategy = NonMaxSuppressionBoundingBoxFilteringStrategy()
boundingBoxFilter: BoundingBoxFilter = BoundingBoxFilter(conf_threshold=0.2, iou_threshold=0.5) #TODO: Set these constructor parameters dynamically
boundingBoxFilter.setBoundingBoxFilteringStrategy(nonMaxSuppressionBoundingBoxFilteringStrategy)

cv_PipelineTail.next = boundingBoxFilter
cv_PipelineTail = boundingBoxFilter


# Create a Bounding Box Filter, set its Strategy, and add it to the CV Pipeline:
sort_ObjectTrackingStrategy: ObjectTrackingStrategy = SORT_ObjectTrackingStrategy(max_age=1, min_hits=5, iou_threshold=0.3)
objectTracker: ObjectTracker = ObjectTracker()
objectTracker.setObjectTrackingStrategy(sort_ObjectTrackingStrategy)

cv_PipelineTail.next = objectTracker
cv_PipelineTail = objectTracker




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
        if cv_PipelineHead is not None:
            cv_PipelineHead.handle({
                'frame': frame
            })
    else:
        exit()


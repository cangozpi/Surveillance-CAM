from time import sleep
from videoStream import VideoStream
from CV_pipeline.CVPipelineStep import CVPipelineStep
from CV_pipeline.CVPipeline import CVPipeline
from pipeline_builder.pipelineBuilder import PipelineBuilder
from pipeline_builder.pipelineDirector import PipelineDirector


# ----- Simulate video stream:
video_path = './test/test video.mp4'
videoStream = VideoStream(video_path)
fps = 1/30


# -----------------
# Create the CV Pipeline:
pipelineBuilder: PipelineBuilder = PipelineBuilder()
pipelineDirector: PipelineDirector = PipelineDirector(pipelineBuilder)
cv_Pipeline = pipelineDirector.make(**{
    # params for NonMaxSuppressionBoundingBoxFilteringStrategy: #TODO: Set these constructor parameters dynamically
    'conf_threshold': 0.2, 
    'iou_threshold': 0.5,

    # param for KNN_PersonIdentificationStrategy:
    'threshold_norm_dist': 100, #TODO: tune this threshold value dynamically

    'notifierHistory_windowSize': int(2*(1/fps)), # param for Notifier
    'forgetting_threshold': 1.0 # param for EmailNotificationStrategy #TODO: tune these parameters
})

# Start processing the stream using the CV Pipeline:
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
# -----------------

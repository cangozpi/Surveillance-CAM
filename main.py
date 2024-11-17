import argparse
from time import sleep
from videoStream import VideoStream
from CV_pipeline.CVPipelineStep import CVPipelineStep
from CV_pipeline.CVPipeline import CVPipeline
from pipeline_builder.pipelineBuilder import PipelineBuilder
from pipeline_builder.pipelineDirector import PipelineDirector

if __name__ == '__main__':
    # Get the video_path from the user as a command line argument:
    parser = argparse.ArgumentParser(description="Smart Surveillance System")
    parser.add_argument(
        '--video_path', 
        type=str,
        required=True,  # Make sure it's a required argument
        help="Path to the video source file or RTSP video stream URL. It cannot be empty."
    )
    args = parser.parse_args()
    

    # ----- Simulate video stream:
    video_path: str = args.video_path
    # If the video_path is a RTSP video stream URL then process it in real time by always processing the newest frame available.
    # If video_path is a video file than process it frame by frame. 
    if video_path.lower().startswith(('rtsp://', 'rtsps://')):
        read_video_stream_non_blocking = True
    else: 
        read_video_stream_non_blocking = False
    videoStream = VideoStream(video_path, non_blocking=read_video_stream_non_blocking)


    # -----------------
    # Create the CV Pipeline:
    pipelineBuilder: PipelineBuilder = PipelineBuilder()
    pipelineDirector: PipelineDirector = PipelineDirector(pipelineBuilder)
    cv_Pipeline = pipelineDirector.make(**{
        # params for NonMaxSuppressionBoundingBoxFilteringStrategy: #TODO: Set these constructor parameters dynamically
        'conf_threshold': 0.2, 
        'iou_threshold': 0.5,

        # params for SORT_ObjectTrackingStrategy #TODO: tune these parameters
        'max_age': 1,
        'min_hits': 5, 
        'iou_threshold': 0.3,

        # param for KNN_PersonIdentificationStrategy:
        'threshold_norm_dist': 100, #TODO: tune this threshold value dynamically

        'notifierHistory_windowSize': int(2*(1/30)), # param for Notifier #TODO: tune these parameters
        'forgetting_threshold': 1.0 # param for EmailNotificationStrategy
    })

    # Start processing the stream using the CV Pipeline:
    while True:
        frame = videoStream.getNextFrame()

        if frame is not None:
            # VideoStream.displayFrame(frame)
            # sleep(fps)

            # pass the frame through the CV Pipeline
            cv_Pipeline.handle_CVPipeline(initial_request={
                'frame': frame
            })
        else:
            exit()
    # -----------------

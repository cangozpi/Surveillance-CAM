from object_detection.objectDetectionStrategy import ObjectDetectionStrategy
from CV_pipeline.CVPipelineStep import CVPipelineStep
from copy import deepcopy
import cv2
import numpy as np

class ObjectDetector(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to detect objects in an image.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform object detection.

    Design Pattern:
        - Context of the Strategy design pattern (ObjectDetectionStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self):
        super().__init__()
        self.objectDetectionStrategy: ObjectDetectionStrategy = None # strategy to use for detecting objects
    
    def setObjectDetectionStrategy(self, strategy: ObjectDetectionStrategy):
        assert(isinstance(strategy, ObjectDetectionStrategy))
        self.objectDetectionStrategy = strategy
    
    def detectObjects(self, frame):
        """
        Performs object detection for the given frame and returns bounding boxes, labels, and scores for the detected objects.

        Inputs:
            frame (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
        Returns:
            boxes (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
            labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                bounding box is in the boxes array's same index.
            scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                score of the detected object at the same index in the boxes and labels arrays.

        Note that this method should return these in the following order as 'return boxes, labels, scores'.
        """
        return self.objectDetectionStrategy.executeObjectDetection(frame)
    
    def handle(self, request): # handle fn of the chain of responsibility pattern (CVPipelineStep)
        """
        Inputs:
            request (dict): {
                                'frame' (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
                            }
        
        - Calls self.next.handle() with the following params:
            {
                'boxes' (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                    corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
                'labels' (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                    bounding box is in the boxes array's same index.
                'scores' (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                    score of the detected object at the same index in the boxes and labels arrays.
                'frame' (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
            }
        """
        assert((type(request)) == dict and ('frame' in request))
        frame = deepcopy(request['frame']) # make a deepcopy because self.detectObjects(frame) ends up changing the oritinal (unannotated) frame object via the pointer
        assert(isinstance(frame, np.ndarray) and (len(frame.shape) == 3) and (frame.shape[2] == 3))

        # detect objects via detectObjects(frame)
        boxes, labels, scores = self.detectObjects(frame)
        assert(isinstance(boxes, np.ndarray) and (len(boxes.shape) == 2) and (boxes.shape[1] == 4))
        assert(isinstance(labels, list) and (len(labels) == boxes.shape[0]))
        assert(isinstance(scores, np.ndarray) and (len(scores.shape) == 1) and (scores.shape == (boxes.shape[0],)))

        print(f'[ObjectDetector] num_detections: {len(boxes)}, detected_classes: {list(set(labels))}')
        # from videoStream import VideoStream
        # annotated_frame = self.getAnnotatedFrame(boxes, labels, scores, deepcopy(frame), threshold=0.1)
        # VideoStream.displayFrame(annotated_frame)
        # VideoStream.displayFrame(frame)

        # decide whether to pass down the chain or not
        if self.next is not None:
            self.next.handle({
                'boxes': boxes, # detected bounding boxes
                'labels': labels, # detection class labels as string
                'scores': scores, # detection confidence scores
                'frame': request['frame'] # original unannotated frame
            })
        
    
    def getAnnotatedFrame(self, boxes, labels, scores, frame, threshold=0.1):
        """
        A utility function. Annotates the frame with bounding boxes and class labels of detections of confidence score higher than self.threshold.
        Note that the input frame will be modified as np.array images are passed by reference and any modifications made in this method would
        reflect everywhere else.
        """
        for i in range(len(scores)):
            if scores[i] > threshold:
                box = boxes[i]
                class_name = labels[i]
                score = scores[i]

                # Draw the bounding box and label on the image
                cv2.rectangle(frame, 
                            (int(box[0]), int(box[1])),  # Top-left corner
                            (int(box[2]), int(box[3])),  # Bottom-right corner
                            (0, 255, 0), 2)  # Green color for the bounding box

                cv2.putText(frame, 
                            f"{class_name}: {score:.2f}",  # Class name and confidence score
                            (int(box[0]), int(box[1]) - 10),  # Position the label above the box
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 2)  # White text with a thickness of 2

        return frame
from object_tracker.objectTrackingStrategy import ObjectTrackingStrategy
from CV_pipeline.CVPipelineStep import CVPipelineStep
import numpy as np

class ObjectTracker(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to track objects in an image.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform object tracking.

    Design Pattern:
        - Context of the Strategy design pattern (ObjectTrackingStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self):
        super().__init__()
        self.objectTrackingStrategy: ObjectTrackingStrategy = None # strategy to use for tracking objects
    
    def setObjectTrackingStrategy(self, strategy: ObjectTrackingStrategy):
        assert(isinstance(strategy, ObjectTrackingStrategy))
        self.objectTrackingStrategy = strategy
    
    def trackObjects(self, boxes, labels, scores):
        """
        Given detected objects, performs object tracking and associates unique ids (tracker_ids) with the detected objects.

        Inputs:
                boxes (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                    corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
                labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                    bounding box is in the boxes array's same index.
                scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                    score of the detected object at the same index in the boxes and labels arrays.

        Returns: returns the filtered detections
                boxes (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                    corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
                labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                    bounding box is in the boxes array's same index.
                scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                    score of the detected object at the same index in the boxes and labels arrays.
                tracker_ids (np.ndarray of shape [N,]): unique id's assigned by the object trackers. Note that id's are assigned starting from 0 again after no object
                    tracker remains due to all trackers being deleted due to various reasons such as inacitivity.

        Note that this method should return these in the following order as 'return boxes, labels, scores, tracker_ids'.
        """
        return self.objectTrackingStrategy.executeObjectTracking(boxes, labels, scores)
    
    def handle(self, request): # handle fn of the chain of responsibility pattern (CVPipelineStep)
        """
        Inputs:
            {
                'boxes' (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                    corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
                'labels' (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                    bounding box is in the boxes array's same index.
                'scores' (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                    score of the detected object at the same index in the boxes and labels arrays.
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
                'tracker_ids' (np.ndarray of shape [N,]): unique id's assigned by the object trackers. Note that id's are assigned starting from 0 again after no object
                    tracker remains due to all trackers being deleted due to various reasons such as inacitivity.
                'frame' (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
            }
        """
        assert((type(request)) == dict and 
        ('boxes' in request) and ('labels' in request) and ('scores' in request) and ('frame' in request))
        boxes = request['boxes']
        labels = request['labels']
        scores = request['scores']
        frame = request['frame']
        assert(isinstance(boxes, np.ndarray) and (len(boxes.shape) == 2) and (boxes.shape[1] == 4))
        assert(isinstance(labels, list) and (len(labels) == boxes.shape[0]))
        assert(isinstance(scores, np.ndarray) and (len(scores.shape) == 1) and (scores.shape == (boxes.shape[0],)))
        assert(isinstance(frame, np.ndarray) and (len(frame.shape) == 3) and (frame.shape[2] == 3))

        # Perform object tracking via self.trackObjects(frame)
        boxes, labels, scores, tracker_ids = self.trackObjects(boxes, labels, scores)
        assert(isinstance(boxes, np.ndarray) and (len(boxes.shape) == 2) and (boxes.shape[1] == 4))
        assert(isinstance(labels, list) and (len(labels) == boxes.shape[0]))
        assert(isinstance(scores, np.ndarray) and (len(scores.shape) == 1) and (scores.shape == (boxes.shape[0],)))
        assert(isinstance(tracker_ids, np.ndarray) and (len(scores.shape) == 1) and (scores.shape == (boxes.shape[0],)))
        print(f'[ObjectTracker] num_tracked_objects: {boxes.shape[0]}')

        # from videoStream import VideoStream
        # from copy import deepcopy
        # annotated_frame = self.getAnnotatedFrame(boxes, labels, scores, deepcopy(request['frame']), tracker_ids, threshold=0.0)
        # VideoStream.displayFrame(annotated_frame)

        # decide whether to pass down the chain or not
        if self.next is not None:
            self.next.handle({
                'boxes': boxes, # detected bounding boxes after filtering
                'labels': labels, # detection class labels as string after filtering
                'scores': scores, # detection confidence scores after filtering
                'tracker_ids': tracker_ids, # tracker ids (unique ids) assigned for the detected objects
                'frame': request['frame'] # original unannotated frame
            })
    
    def getAnnotatedFrame(self, boxes, labels, scores, frame, tracker_ids, threshold=0.1):
        """
        A utility function. Annotates the frame with bounding boxes and class labels of detections of confidence score higher than self.threshold.
        Note that the input frame will be modified as np.array images are passed by reference and any modifications made in this method would
        reflect everywhere else.
        """
        import cv2
        for i in range(len(scores)):
            if scores[i] > threshold:
                box = boxes[i]
                class_name = labels[i]
                score = scores[i]
                tracker_id = tracker_ids[i]

                # Draw the bounding box and label on the image
                cv2.rectangle(frame, 
                            (int(box[0]), int(box[1])),  # Top-left corner
                            (int(box[2]), int(box[3])),  # Bottom-right corner
                            (0, 255, 0), 2)  # Green color for the bounding box

                cv2.putText(frame, 
                            f"{class_name}: tracker_id_{tracker_id} : {score:.2f}",  # Class name and confidence score
                            (int(box[0]), int(box[1]) - 10),  # Position the label above the box
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 2)  # White text with a thickness of 2

        return frame
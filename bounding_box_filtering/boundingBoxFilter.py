from bounding_box_filtering.boundingBoxFilteringStrategy import BoundingBoxFilteringStrategy
from CV_pipeline.CVPipelineStep import CVPipelineStep
from copy import deepcopy
import cv2
import numpy as np

class BoundingBoxFilter(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to filter bounding boxes as a post-processing step to the outputs of the object detector.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform bounding box filtering post-processing.
    Also by default filters the detections by comparing against a given confidence threshold value (conf_threshold).

    Design Pattern:
        - Context of the Strategy design pattern (BoundingBoxFilteringStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self, conf_threshold=0.5, iou_threshold=0.5):
        """
        Params:
            conf_threshold (float): Object detections with confidence scores lower than this value are filtered by default. 
                You can pass 0 to turn the confidence thresholding based filtering off.
            iou_threshold (float): Intersection Over Uniont (IOU) threshold value which is passed to the BoundingBoxFilteringStrategy's execute()
                method. Whether this value would be used depends solely on the BoundingBoxFilteringStrategy class you chose to use.
        """
        super().__init__()
        self.boundingBoxFilteringStrategy: BoundingBoxFilteringStrategy = None # strategy to use for filtering the bounding box outputs of the object detector
        assert((0 <= conf_threshold <= 1) and (0 <= iou_threshold <= 1))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def setBoundingBoxFilteringStrategy(self, strategy: BoundingBoxFilteringStrategy):
        assert(isinstance(strategy, BoundingBoxFilteringStrategy))
        self.boundingBoxFilteringStrategy = strategy
    
    def filterBoundingBoxes(self, boxes, labels, scores, iou_threshold):
        """
        Filters the detected objects (corresponding bounding boxes, labels, and confidence scores).
        By default as a first step a filtering by comparing the confidence scores of the detections against a threshold value is applied.
        Then the filtering is done by passing detections to the selected BoundingBoxFilteringStrategy's execute() method.

        Inputs:
            boxes (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
            labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                bounding box is in the boxes array's same index.
            scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                score of the detected object at the same index in the boxes and labels arrays.
            iou_threshold (float): Intersection Over Uniont (IOU) threshold value which is passed to the BoundingBoxFilteringStrategy's execute()
                method. Whether this value would be used depends solely on the BoundingBoxFilteringStrategy class you chose to use.
        
        Returns: returns the filtered detections
            boxes (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
            labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                bounding box is in the boxes array's same index.
            scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                score of the detected object at the same index in the boxes and labels arrays.
        Note: as an edge case returning a single detection as boxes.shape == (4,) with selected_scores as a single np.value is ok.

        Note that this method should return these in the following order as 'return boxes, labels, scores'.
        """
        # filter by confidence threshold
        filtered_boxes, filtered_labels, filtered_scores = self.confidence_threshold_filtering(self.conf_threshold, boxes, labels, scores)
        # filter according to the strategy
        return self.boundingBoxFilteringStrategy.executeBoundingBoxFiltering(filtered_boxes, filtered_labels, filtered_scores, iou_threshold)
    
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
                'frame' (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
            }
            Note that 'boxes', 'labels', 'scores' are filtered versions of their input equivalents.
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

        # filter detected bounding boxes via fiterBoundingBoxes()
        selected_boxes, selected_labels, selected_scores = self.filterBoundingBoxes(boxes, labels, scores, self.iou_threshold)

        # handle edge case of a single detection being returned with shape (4,) and convert it to (1,4) to keep the batch dimension
        if selected_boxes.shape == (4,) or len(selected_boxes.shape) < 2: 
            selected_boxes = np.expand_dims(selected_boxes, axis=0)
            selected_scores = np.expand_dims(np.array(selected_scores), axis=0)
            assert(selected_boxes.shape == (1,4) and selected_scores.shape == (1,))
        assert(isinstance(selected_boxes, np.ndarray) and (len(selected_boxes.shape) == 2) and (selected_boxes.shape[1] == 4))
        assert(isinstance(selected_labels, list) and (len(selected_labels) == selected_boxes.shape[0]))
        assert(isinstance(selected_scores, np.ndarray) and (len(selected_scores.shape) == 1) and (selected_scores.shape == (selected_boxes.shape[0],)))
        
        print(f'[BoundingBoxFilter] num_detections_filtered: {len(selected_boxes)}, detected_classes_filtered: {list(set(selected_labels))}')

        # from videoStream import VideoStream
        # annotated_frame = self.getAnnotatedFrame(selected_boxes, selected_labels, selected_scores, deepcopy(request['frame']), threshold=0.0)
        # VideoStream.displayFrame(annotated_frame)
        # VideoStream.displayFrame(request['frame'])

        # decide whether to pass down the chain or not
        if self.next is not None:
            self.next.handle({
                'boxes': selected_boxes, # detected bounding boxes after filtering
                'labels': selected_labels, # detection class labels as string after filtering
                'scores': selected_scores, # detection confidence scores after filtering
                'frame': frame # original unannotated frame
            })
    
    def confidence_threshold_filtering(self, conf_threshold, boxes, labels, scores):
        """
        Filter detected objects by their confidence scores and a given threshold.
        """
        filtered_indices = np.argwhere(scores > conf_threshold).squeeze(axis=-1)
        filtered_boxes = boxes[filtered_indices]
        filtered_labels = [labels[i] for i in filtered_indices.tolist()]
        filtered_scores = scores[filtered_indices]
        return filtered_boxes, filtered_labels, filtered_scores


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
    
    def set_conf_threshold(self, conf_threshold):
        self.conf_threshold = conf_threshold

    def set_iou_threshold(self, iou_threshold):
        self.iou_threshold = iou_threshold
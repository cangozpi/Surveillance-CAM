from notification.notificationStrategy import NotificationStrategy
from CV_pipeline.CVPipelineStep import CVPipelineStep
import numpy as np
from copy import deepcopy
import cv2
from collections import deque
from collections import Counter
from datetime import datetime

class Notifier(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to notify the user.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform change detection.

    Design Pattern:
        - Context of the Strategy design pattern (NotificationStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self, notifierHistory_windowSize:int=30, forgetting_threshold:float=1.0):
        """
        Constructor Parameters:
            notifierHistory_windowSize (int): Sliding window size used for smoothing the object detections before deciding whether to send 
                a notification to the user. This is also the number of consecutive frames that a tracker_id should appear in order to have a 
                notification sent about it.
            forgetting_threshold (float): Minutes (time) after which to remove a tracker_id from the self.notification_history if it has not been 
                detected consecutively for the past self.shouldUpdate_windowSize frames.
        """
        super().__init__()
        self.notificationStrategy: NotificationStrategy = None # strategy to use for notifying the user
        
        self.notifierHistory_windowSize = notifierHistory_windowSize  # sliding window size used for smoothing the object detections before deciding whether to send a notification to the user. 
                                            # This is also the number of consecutive frames that a tracker_id should appear in order to have a notification sent about it.
        self.tracker_id_history = deque(maxlen=self.notifierHistory_windowSize) # holds tracker_ids obtained during the last windowSize frames
        self.notification_history = {} # records when a notification regarding a tracker_id was sent {'tracker_id': notification_time}
        self.forgetting_threshold = forgetting_threshold  # minutes (time) after which to remove a tracker_id from the self.notification_history if it has not been 
                                        #detected consecutively for the past self.shouldUpdate_windowSize frames.
    
    def setNotificationStrategy(self, strategy: NotificationStrategy):
        assert(isinstance(strategy, NotificationStrategy))
        self.notificationStrategy = strategy
    
    def notify(self, msg: str, frame):
        """
        Notifies the user about the given info.
        Inputs:
            msg (str): text message
            frame (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
        """
        self.notificationStrategy.executeNotify(msg, frame)
    
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
                'tracker_ids' (np.ndarray of shape [N,]): unique id's assigned by the object trackers. Note that id's are assigned starting from 0 again after no object
                    tracker remains due to all trackers being deleted due to various reasons such as inacitivity.
                'frame' (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
                'identified_person_names' (list of strings of length N): (list of strings of length N) same length as boxes, so if a detection 
                    in a certain index in 'boxes' is a person, and it was matched with a known person in the KnownPersonDb, 
                    then that index in this array will contain the corresponding matched person's name (str). If no match was found for a 
                    person detection or a person was not matched with any person in the db then it will contain the value None (not as string). 
                    For example, if 'boxes'[idx] is a person and it was matched with the known person 'Kevin' in the KnownPersonDb, 
                    then 'identified_person_names'[i] == 'Kevin'.
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
                'identified_person_names' (list of strings of length N): (list of strings of length N) same length as boxes, so if a detection 
                    in a certain index in 'boxes' is a person, and it was matched with a known person in the KnownPersonDb, 
                    then that index in this array will contain the corresponding matched person's name (str). If no match was found for a 
                    person detection or a person was not matched with any person in the db then it will contain the value None (not as string). 
                    For example, if 'boxes'[idx] is a person and it was matched with the known person 'Kevin' in the KnownPersonDb, 
                    then 'identified_person_names'[i] == 'Kevin'.
            }
        """
        assert((type(request)) == dict and 
        ('boxes' in request) and ('labels' in request) and ('scores' in request) and ('tracker_ids' in request) and ('frame' in request) and ('identified_person_names' in request))
        boxes = request['boxes'].copy()
        labels = request['labels']
        scores = request['scores']
        tracker_ids = request['tracker_ids']
        frame = deepcopy(request['frame'])
        identified_person_names = request['identified_person_names']
        assert(isinstance(boxes, np.ndarray) and (len(boxes.shape) == 2) and (boxes.shape[1] == 4))
        assert(isinstance(labels, list) and (len(labels) == boxes.shape[0]))
        assert(isinstance(scores, np.ndarray) and (len(scores.shape) == 1) and (scores.shape == (boxes.shape[0],)))
        assert(isinstance(tracker_ids, np.ndarray) and (len(tracker_ids.shape) == 1) and (tracker_ids.shape == (boxes.shape[0],)))
        assert(isinstance(frame, np.ndarray) and (len(frame.shape) == 3) and (frame.shape[2] == 3))
        assert(isinstance(identified_person_names, list) and (len(identified_person_names) == boxes.shape[0]))

        # Check if user should be notified
        new_objects_to_notify_about = self.getNewObjectsToNotifyAbout(tracker_ids)

        # If there is any object which the user should be notified about then send a notification to the user
        if len(new_objects_to_notify_about) > 0: 
            notification_msg = ''
            notification_msg += f'detected objects: {labels}, \n'
            notification_msg += f'identified_people_names: {list(filter(lambda x: x is not None, identified_person_names))}, \n'
            notification_msg += f'new_objects_to_notify_about: {new_objects_to_notify_about}, \n'
            self.notify(msg=notification_msg 
            , frame=self.getAnnotatedFrame(boxes, labels, scores, frame, tracker_ids, identified_person_names))

        print(f'[Notifier] new_objects_to_notify_about:{new_objects_to_notify_about}')

        # decide whether to pass down the chain or not
        if self.next is not None:
            self.next.handle({
                'boxes': request['boxes'], # detected bounding boxes after filtering
                'labels': labels, # detection class labels as string after filtering
                'scores': scores, # detection confidence scores after filtering
                'tracker_ids': tracker_ids, # tracker ids (unique ids) assigned for the detected objects
                'frame': request['frame'], # original unannotated frame
                'identified_person_names': identified_person_names, # a list of people names that were matched from the db
            })
    
    def getAnnotatedFrame(self, boxes, labels, scores, frame, tracker_ids, identified_person_names):
        """
        A utility function. Annotates the given frame with the passed in information and returns it.
        Note that the input frame will be modified as np.array images are passed by reference and any modifications made in this method would
        reflect everywhere else.
        """
        for i in range(len(scores)):
            box = boxes[i]
            class_name = labels[i]
            score = scores[i]
            tracker_id = tracker_ids[i]
            identified_person_name = identified_person_names[i]

            # Draw the bounding box and label on the image
            cv2.rectangle(frame, 
                        (int(box[0]), int(box[1])),  # Top-left corner
                        (int(box[2]), int(box[3])),  # Bottom-right corner
                        (0, 255, 0), 2)  # Green color for the bounding box

            cv2.putText(frame, 
                        f"{class_name}: tracker_id_{tracker_id}: {score:.2f}" if identified_person_name is None else f"{class_name}: tracker_id_{tracker_id}: [{identified_person_name}] : {score:.2f}",  # Class name and confidence score
                        (int(box[0]), int(box[1]) - 10),  # Position the label above the box
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2)  # White text with a thickness of 2

        return frame
    
    def getNewObjectsToNotifyAbout(self, tracker_ids):
        """
        Checks whether the user should be notified by using a sliding window approach, finds the object which the user 
        should be notified about and returns those objects tracker_ids in a list.

        Inputs:
                tracker_ids (np.ndarray of shape [N,]): unique id's assigned by the object trackers. Note that id's are assigned starting from 0 again after no object
                    tracker remains due to all trackers being deleted due to various reasons such as inacitivity.
        
        Returns:
            new_objects_to_notify_about (list): list of tracker_ids of objects which the user should be notified about.
        """
        new_objects_to_notify_about = []

        # Update the history with the current frame
        self.tracker_id_history.append(tracker_ids)

        # Count how many times each tracker_id appeared in the history:
        tracker_id_counts = {} # like {'tracker_id': count}
        for h in self.tracker_id_history:
            for tracker_id in h:
                tracker_id_counts[tracker_id] = tracker_id_counts.get(tracker_id, 0) + 1
        
        for tracker_id, count in tracker_id_counts.items():
            # Check if tracker_id_has_been notified before
            if tracker_id not in self.notification_history:
                # Check if it has appeared enough times over the history (i.e. was it stable)
                if count >= self.notifierHistory_windowSize:
                    self.notification_history[tracker_id] = datetime.now()
                    new_objects_to_notify_about.append(tracker_id)

        tracker_ids_to_remove = [] # tracker_ids to remove
        for tracker_id, last_notification_time in self.notification_history.items():
            # Check if the tracker_id has been not seen in any frame of the recent history
            if tracker_id not in [tracker_id for sublist in self.tracker_id_history for tracker_id in sublist]:
                # Check if enough time has passed over the last time the object was notified so that we can forget about that object ever being notified
                if ((datetime.now() - last_notification_time).total_seconds() / 60) > self.forgetting_threshold: # calculate time difference in minutes
                    tracker_ids_to_remove.append(tracker_id)
        
        for tracker_id in tracker_ids_to_remove:
            del self.notification_history[tracker_id]

        return new_objects_to_notify_about


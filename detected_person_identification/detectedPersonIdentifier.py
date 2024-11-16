from detected_person_identification.detectedPersonIdentificationStrategy import DetectedPersonIdentificationStrategy
from CV_pipeline.CVPipelineStep import CVPipelineStep
import numpy as np
from copy import deepcopy
import cv2
from detected_person_identification.knownPersonDb import KnownPersonDb

class DetectedPersonIdentifier(CVPipelineStep):
    """
    Set the strategy (algorithm) you want to use to identify people detected on an image by comparing them 
    against the entries in an available database of identities.
    Can be added to the Computer Vision Pipeline (CVPipelineStep) as a step to perform object tracking.

    Design Pattern:
        - Context of the Strategy design pattern (DetectedPersonIdentificaitonStrategy).
        - Concrete Handler of the Chain of Responsibility design pattern pattern (CVPipelineStep).
    """
    def __init__(self, db:KnownPersonDb):
        super().__init__()
        self.detectedPersonIdentificationStrategy: DetectedPersonIdentificationStrategy = None # strategy to use for identifying people detected on an image

        self.db = db # database of known people
        self.db_dict =  self.db.get_db_dict() # db_dict (dict | keys: string | values: array of np arrays): persons name as the key, 
            # and the array of feature vectors (np arrays)extracted from the images that person had in the database.
        assert (isinstance(self.db_dict, dict))
        if len(self.db_dict.keys()) > 0:
            assert(isinstance(list(self.db_dict.keys())[0], str) and isinstance(list(self.db_dict.values())[0], list))
    
    def setPersonIdentificationStrategy(self, strategy: DetectedPersonIdentificationStrategy):
        assert(isinstance(strategy, DetectedPersonIdentificationStrategy))
        self.detectedPersonIdentificationStrategy = strategy
    
    def identifyPeople(self, cropped_person_images):
        """
        Given a list of cropped out person detection and a dictionary containing known people names and paths to their associated
        images in the KnownPersonDb, it checks if the cropped person detections match any of the known people from the db. If
        it matches then a name is assigned to that detection, if not no name is assigned. The final list of assigned names are
        returned by this function as a list in the end.

        Inputs:
            cropped_person_images (list of image 2D BGR image of shape [H,W,C=3]): contains the cut out images of the detected people in the given frame.

        Returns:
            cropped_person_names (list of string): a list where each index holds the name (str) assigned to the person detection given
                at the same index in the cropped_person_images. It holds the value of <None> (not string 'None') when the detection
                was not assigned a name (i.e. when detection did not match any known person in the db_dict (KnownPersonDb)).
        """
        return self.detectedPersonIdentificationStrategy.executePersonIdentification(cropped_person_images, self.db_dict)
    
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
        ('boxes' in request) and ('labels' in request) and ('scores' in request) and ('frame' in request))
        boxes = request['boxes'].copy()
        labels = request['labels']
        scores = request['scores']
        tracker_ids = request['tracker_ids']
        frame = deepcopy(request['frame'])
        assert(isinstance(boxes, np.ndarray) and (len(boxes.shape) == 2) and (boxes.shape[1] == 4))
        assert(isinstance(labels, list) and (len(labels) == boxes.shape[0]))
        assert(isinstance(scores, np.ndarray) and (len(scores.shape) == 1) and (scores.shape == (boxes.shape[0],)))
        assert(isinstance(tracker_ids, np.ndarray) and (len(tracker_ids.shape) == 1) and (tracker_ids.shape == (boxes.shape[0],)))
        assert(isinstance(frame, np.ndarray) and (len(frame.shape) == 3) and (frame.shape[2] == 3))

        # extract detected people from the given frame
        detected_people_indices, cropped_person_images = self.extractPeopleFromFrame(boxes, labels, scores, tracker_ids, frame)

        # Perform person identificaiton
        if len(detected_people_indices) > 0:
            cropped_person_names  = self.identifyPeople(cropped_person_images) # list of matched person names (str), with None for no matches
            assert(len(cropped_person_names) == len(cropped_person_images))
        
        identified_person_names = [None for i in range(boxes.shape[0])] # (list of strings of length N) same length as boxes so if a detection
                                                                                # at a certain index in boxes is a person and it was matched with a
                                                                                # known person in the KnownPersonDb, then that index in this
                                                                                # array will contain the name of the matched person's name (str)
        for i in range(len(cropped_person_images)):
            cur_matched_person_name = cropped_person_names[i]
            identified_person_names[detected_people_indices[i]] = cur_matched_person_name
        assert(isinstance(identified_person_names, list) and len(identified_person_names) == boxes.shape[0])
        print(f'[DetectedPersonIdentifier] identified_people: {list(filter(lambda x: x is not None, identified_person_names))}')

        from videoStream import VideoStream
        annotated_frame = self.getAnnotatedFrame(boxes, labels, scores, frame, tracker_ids, identified_person_names, threshold=0.1)
        VideoStream.displayFrame(annotated_frame)

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
    
    def extractPeopleFromFrame(self, boxes, labels, scores, tracker_ids, frame):
        """
        Extracts the detected people from the given frame and returns them as separate images along with their corresponding 
        indices in the boxes np array. People are cut from the given frame by their corresponding bounding boxes.

        Inputs:
                'boxes' (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                    corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
                'labels' (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                    bounding box is in the boxes array's same index.
                'scores' (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                    score of the detected object at the same index in the boxes and labels arrays.
                'tracker_ids' (np.ndarray of shape [N,]): unique id's assigned by the object trackers. Note that id's are assigned starting from 0 again after no object
                    tracker remains due to all trackers being deleted due to various reasons such as inacitivity.
                'frame' (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]

        Returns:
                detected_people_indices (list): a list of integers where each int corresponds to a person detected in the boxes np array.
                cropped_person_images (list of image 2D BGR image of shape [H,W,C=3]): contains the cut out images of the detected people in the given frame.
        """
        # Find the detections that only correspond to person
        detected_people_indices = [i for i, l in enumerate(labels) if l == 'person']
        person_boxes = np.clip(np.floor(boxes[detected_people_indices, :]), a_min=0, a_max=None).astype(np.uint)

        assert(person_boxes.shape == (len(detected_people_indices), 4))

        # cut out the detected people according to their detection bounding boxes
        cropped_person_images = []
        for (x1, y1, x2, y2) in person_boxes:
            # Crop the region of interest (ROI) from the frame
            cropped_image = frame[y1:y2, x1:x2] # 2D BGR image of shape [H, W, C=3]
            if cropped_image.shape[1] == 0:
                print("NOOOOOOOOOOOOOOOOO")
                breakpoint()
            cropped_person_images.append(cropped_image)

            # from videoStream import VideoStream
            # VideoStream.displayFrame(cropped_image)
        if len(cropped_person_images) > 0:
            assert((len(cropped_person_images) == len(detected_people_indices)) and (len(cropped_person_images[0].shape) == 3))
        
        return detected_people_indices, cropped_person_images
    
    def getAnnotatedFrame(self, boxes, labels, scores, frame, tracker_ids, identified_person_names, threshold=0.1):
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
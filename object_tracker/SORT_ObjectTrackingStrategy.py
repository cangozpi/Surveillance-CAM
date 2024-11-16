from object_tracker.objectTrackingStrategy import ObjectTrackingStrategy

from filterpy.kalman import KalmanFilter
import numpy as np

class SORT_ObjectTrackingStrategy(ObjectTrackingStrategy):
    """
    Concrete implementation of the ObjectTrackingStrategy (Strategy pattern) which can be used by the ObjectTracker. 
    Performs object tracking using the SORT (Simple Online and Realtime Tracking) algorithm.
    SORT (Simple Object Realtime Tracking) is an algorithm which combines Kalman Filter with Hungarian Algorithm to 
    tackle the object tracking task.

    Design Pattern:
        Concrete implementation of the Strategy interface of the Strategy design pattern (BoundingBoxFilteringStrategy).
    """
    def __init__(self, max_age=1, min_hits=5, iou_threshold=0.3):
        """
        Params:
            max_age (int): Maximum number of frames to keep alive a track without associated detections.
            min_hits (int): Minimum number of associated detections before track is initialised.
            iou_threshold (float): Minimum IOU (Intersection Over Union) for match.
        """
        super().__init__()
        self.SORT_tracker = SORT(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold) #create instance of the SORT tracker


    def executeObjectTracking(self, boxes, labels, scores):
        """
        Given detected objects, performs object tracking using the SORT algorithm and associates unique ids (tracker_ids) with the detected objects.

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
        # self.SORT_tracker.update() expects inputs as np array of [x1, y1, x2, y2, confidence_score]
        boxes_and_scores_appended = np.hstack((boxes, np.expand_dims(scores, axis=-1)))

        trackers, labels, scores = self.SORT_tracker.update(boxes_and_scores_appended, labels) # np array of elements [x1, y1, x2, y2, track_i], and the corresponding labels
        boxes = trackers[:,:-1]
        tracker_ids = trackers[:, -1]

        # for d in trackers:
        #     print('id:%d, x1:%.2f, y1:%.2f, w:%.2f, h:%.2f'%(d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]), labels)
        
        return boxes, labels, np.array(scores), tracker_ids




# -----
# ---- Helper classes below:
# (SORT implementation is modified from: https://github.com/abewley/sort/blob/master/sort.py#L42)

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
  
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)  


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    Implements a Kalman Filter which is used for tracking the 2D positions of object detection's bounding boxes.

    Tracked state is as follows:
        X_t = [x1, y1, s (scale (i.e. w*h)), r (i.e. aspect ratio (w/h)), vel_x1, vel_y1, delta_s (i.e. change in scale per unit time)]
        Y_t = [x1, y1, s (scale (i.e. w*h)), r (i.e. aspect ratio (w/h))]
            where w and h correspond to the width and the height of the detected object's bounding box.

    Note that:
        F is constructed such at X_t+1 = F X_t follows the following equations:
            x1_t+1 = x1_t + vel_x1 * delta_t
            y1_t+1 = y1_t + vel_y1 * delta_t
            s1_t+1 = s1_t + delta_s * delta_t
            r_t+1 = r_t
            vel_x1_t+1 = vel_x1_t
            vel_y1_t+1 = vel_y1_t
            delta_s_t+1 = delta_s_t
    
        H is constructed so that it extracts the [x1, y1, s, r] from the state vector X ([x1, y1, s, r, vel_x1, vel_y1, delta_s])
    """

    count = 0 # made static to give unique track_id's
    def __init__(self,bbox, label):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 

        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]])

        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox) # X_t = [x1, y1, s (scale (i.e. w*h)), r (i.e. aspect ratio (w/h)), vel_x1, vel_y1, delta_s (i.e. change in scale per unit time)]
        self.time_since_update = 0
        self.detection_confidence_score = bbox[-1] # corresponding confidence score of the detected object
        self.label = label # corresponding class (category) label of the detected object
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = [] # past track/path of the object's locations
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class SORT(object):
    """
    Implements the SORT (Simple Online Object Tracking) algorithm for 2D object tracking using the detected bounding box coordinates.
    It uses Kalman Filter and the Hungarian Algorithm to track the objects.
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age #TODO: set these dynamically
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets=np.empty((0, 5)), detection_labels=[]):
        """
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).

        Inputs:
            dets: (np.array of shape [num_detections, 5]) - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]. 
                It is a concatenation of the bounding box coordinates and the confidence scores of the detections.
            detection_labels (list): list that carries the detected objects class labels. Each label in the list 
                corresponds to a bounding box with same index in the dets np array.
        Returns:
            bounding_boxes_and_track_ids (np array of shape [num_detections, 5]): a numpy array of detections in the format [[x1, y1, x2, y2, track_id], [x1, y1, x2, y2, track_id], ...].
                In other words the arrays a[:4] (i.e. first 4 columns) contains the [x1, y1, x2, y2] coordinates of the bounding boxes, and the a[:-1] (i.e. last column) contains the unique tracking ids.
            labels (list of strings): contains class labels corresponding to the same index in the bounding_boxes np array.
            confidence_scores (list of float): contains confidence scores of the detections correesponding to the same index in the bounding boxes np array.

        NOTE: The number of detections returned may differ from the number of detections provided as it also 
            performs IOU (Intersection Over Union) thresholding.
        """
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        ret_labels = []
        ret_confidence_scores = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold) 
        # matched is of type [[detection_index, tracker_index], [detection_index, tracker_index], ...]
        # unmatched_dets is of type [unmatched_detection_index, unmatched_detection_index, ...]
        # unmatched_trks is of type [unmatched_tracker_index, unmatched_tracker_index, ...]

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:], detection_labels[i])
            if len(self.trackers) == 0: # if no trackers are left then zero the count so that tracker_ids would not keep increasing and overflow in the future
                KalmanBoxTracker.count = 0
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
                ret_labels.append(trk.label)
                ret_confidence_scores.append(trk.detection_confidence_score)
            i -= 1
            # remove dead tracker
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret), ret_labels, ret_confidence_scores
        return np.empty((0,5)), ret_labels, ret_confidence_scores
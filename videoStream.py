import time
import cv2
import multiprocessing


class VideoStream:
    def __init__(self, video_path, non_blocking=True):
        """
        Constructor Params:
            video_path (str): Something like a file path to a video, or a rtsp link for a video feed which
                will be used by cv2.VideoCapture(video_path).
            non_blocking (bool): If False then each frame of the video is read by blocking the main frame and every frame
                is returned without missing any. If True then video is read in a subprocess in a non-blocking way, and 
                every frame is continuously being read hence when you ask for the latest frame you might miss some frames in 
                between which the VideoStream consumed before you asked for a new frame.
        """
        # Open the video file
        self.video_path = video_path
        self.non_blocking = non_blocking 
        
        # Consume video feed continuously in a subprocess
        self.frame_queue = None
        self.video_feed_consumer_process = None
        self.getNextFrame = None
        self.cap = None
        if self.non_blocking == True:
            self.video_consuming_delay = 0.01 # sleep time between reading frames
            multiprocessing.set_start_method('spawn', force=True) # ensure spawn is used as the start method
            self.frame_queue = multiprocessing.Queue(maxsize=1)  # Queue to store only the latest frame
            self.video_feed_consumer_process = multiprocessing.Process(target=self._read_frames)
            self.video_feed_consumer_process.daemon = True  # daemonize the process, so it exits when the main program exits
            self.video_feed_consumer_process.start()
            self.getNextFrame = self.getNextFrameNonBlocking # change the fn pointer so that it consumes the latest frame from the Qeueu
        else:
            self.getNextFrame = self.getNextFrameBlocking # change the fn pointer so that it consumes frame by frame by blocking the main thread

            # Check if the video opened successfully
            self.cap = cv2.VideoCapture(self.video_path)
            try:
                if not self.cap.isOpened():
                    raise Exception("Error: Could not open video.")
            except:
                self.close()
    
    def _read_frames(self):
        """ 
        Worker function to read frames from the video and put them in the queue in a subprocess. 
        It continuously consumed the video feed and puts the latest consumed frame into the queue.

        Puts None to the queue if the end of the video is reached, else it puts the latest frame.
        """
        # Create a new VideoCapture instance inside the subprocess
        cap = cv2.VideoCapture(self.video_path)

        while True:
            if not cap.isOpened():
                print("Error: Could not open video in subprocess.", flush=True)
                self.frame_queue.put(None) # put None to signal the end of the video
                break

            ret, frame = cap.read()

            # Check if the video has finished:
            if not ret:
                print("Reached the end of the video.", flush=True)
                self.frame_queue.put(None) # put None to signal the end of the video
                break  # exit the loop if video ends

            try:
                # If the queue is full, get the oldest frame to make space for the new one
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()  # Remove the oldest frame from the queue

                # Put the latest frame into the queue (this will overwrite the previous one)
                self.frame_queue.put(frame, block=False) # Note that put() method does not replace the last element if the 
                                                            # queue is full hence we have to check if the queue is full and remove the element before putting it here

                # print(f"Frame read and added to queue", flush=True)
            except:
                # This block shouldn't be hit if block=False is used, as we already checked if the queue is full
                print(f'Exception caught in VideoStream._read_frames() in videoStream.py.')
                pass

            time.sleep(self.video_consuming_delay)  # small delay to prevent CPU over-utilization

        # When video ends, release resources
        cap.release()
        cv2.destroyAllWindows()
    
    def getNextFrameNonBlocking(self):
        """ 
        Get the latest frame from the queue read by a subprocess in a non-blocking way.

        Returns:
            'frame' (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
        
        Note: it returns None if the end of the video is reached.
        """
        frame =  self.frame_queue.get(block=True, timeout=None)
        if frame is not None: # None signals that the end of the video is reached
            return frame
        else:
            return None

    def getNextFrameBlocking(self):
        """
        Reads the consecutive frame by blocking the main and returns it.

        Returns:
            'frame' (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
        
        Note: it returns None if the end of the video is reached.
        """
        ret, frame = self.cap.read()

        # If frame is read correctly, ret will be True
        if not ret:
            print("Reached the end of the video.")
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
            return None

        return frame
    

    @staticmethod
    def displayFrame(frame):
        # Display the current frame
        cv2.imshow('Video Frame', frame)

        # Wait for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

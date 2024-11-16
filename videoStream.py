#TODO: this is a temporary implementation used for debugging
import cv2


class VideoStream:
    def __init__(self, video_path):
        # Open the video file
        self.cap = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        try:
            if not self.cap.isOpened():
                raise Exception("Error: Could not open video.")
        except:
            self.close()


    def getNextFrame(self): # TODO: add the option to run this on a separate thread
        ret, frame = self.cap.read()

        # If frame is read correctly, ret will be True
        if not ret:
            print("Reached the end of the video.")
            return None

        return frame
    

    @staticmethod
    def displayFrame(frame):
        # Display the current frame
        cv2.imshow('Video Frame', frame)

        # Wait for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()


    def close(self):
        # Release the video capture object and close any OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()
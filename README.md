# Surveillance CAM
 
Many people have installed surveillance cameras to their homes. Despite the vast opportunities these systems present many people just use them to watch and record surveillance videos. __Surveillance CAM__ aims to extend these capabilities by connecting to the already available RTSP streams of the installed cameras. Below are a list of the additional features provided by this project.

__Features__:

* <u>_Change Detection </u>_: <br>
    Determines if a change has occurred, such as the presence of an object of interest, which might require further investigation of the current frame.
* <u>_Object Detection </u>_: <br>
    Performs object detection and produces class labels, bounding boxes, confidence scores.
* <u>_Object Tracking </u>_: <br>
    Performs object tracking on the detected objects.
* <u>_Detected Person Identification </u>_: <br>
    Performs object identification. Given a database of person names and their corresponding images, it tries to identify people detected by the object detector on the current frame.
* <u>_Notification </u>_: <br>
    Keeps track of the past notifications sent to the user and if something worthy of informing the user has come up, it sends a notification to the user.
---

### Usage:
* __Development Environment:__ <br>
&emsp; Python 3.10.13 <br>
&emsp; Ubuntu 22.04.5 LTS x86_64

* __Installing requirements:__
    ```bash
    pip install -r requirements.txt
    ```

* __Running:__ <br>
    ```bash
    EMAIL=yourMailAddress@gmail.com  EMAIL_PASSWORD=yourEmailPassword python3 main.py --video_path='./test/test video.mp4'
    ```
    __--video_path__ can either be the relative path of the video file you want to process frame by frame, or the URL of the RTSP video stream that you want to process in real time by always grabbing the newest available frame.

    If you wish to get notificaiton via email then set _EMAIL_ and _EMAIL\_PASSWORD_ accordingly. _EMAIL_ should be a gmail account and it will be used to send notification from itself to itself. _EMAIL\_PASSWORD_ should be the password of the _EMAIL_ you have specified. If wrong credentials are provided then no notification will be sent and system will continue to function without sending email notifications. Note that to use gmail one should set and use a less secure app password and pass it as the _EMAIL_PASSWORD=xxx_ (see the accepted answer in this [discussion](https://stackoverflow.com/questions/16512592/login-credentials-not-working-with-gmail-smtp)).

    * __Sample Output:__ <br>
    Downloading and using the following [video](https://www.youtube.com/watch?v=yWFv9eA72LA) produces the following outputs (Note that the __KnownPersonDb__ contains a few images of the delivery man present in the image associated with the name Kevin along with other random people names and pictures):

        <img src="./assets/sample email notification ss.png" width="100%" alt="A sample email notification.">
        A sample email notification.

        <img src="./assets/sample annotated frame.jpg" width="100%" alt="Sample annotated frame which was sent as an attachment with the email notification above.">
        Sample annotated frame which was sent as an attachment with the email notification above.
---

### Architecture
It aims to be easily extensible with new features and allow for the switching between different algorithms (i.e. different implementations of a functionality) that underly a feature.

* __Image Processing Flow (CV Pipeline)__: <br>
A high level overview of how the system works is as follows. 

    Current frame of the video is obtained and passed into the <u>ChangeDetector</u>. It determines whether a change has occurred in the scene which might require further investigation of the current frame. If no change is detected then the image is not processed any further and the system waits for the next frame to arrive. But, if a change is detected then that frame is passed down to the ObjectDetector.

    <u>ObjectDetector</u> performs object detection in the passed down (current) frame. It produces class labels, bounding box coordinates, and confidence scores. Along with the current frame these detected objects are passed down to the BoundingBoxFilter.

    <u>BoundingBoxFilter</u> filters the bounding box predictions of the object detector. It aims to get rid of redundant and low confidence object predictions. The filtererd object detections and the current frame are passed down further to the ObjectTracker.

    <u>ObjectTracker</u> performs multi-object tracking and assigns unique tracker_ids to the object detections. Then these unique tracker_ids are passed to the DetectedPersonIdentifier along with the current frame and object detections.

    <u>DetectedPersonIdentifier</u> tries to identify detected people on the current frame by comparing them against a supplied database of known people which contains people's names and their corresponding reference images. Then the detected people which were identified with a known name from the database are passed down to the Notifier along with the tracker_ids, object detections, and the current frame.

    <u>Notifier</u> keeps track of the notifications that has been sent to the user and by taking that into consideration decides whether a new notificaiton should be sent to the user. If so it sends a new notification to the user, else it does not send. The medium that the notification is sent depends on the underlying implementation of the used _NotificationStrategy_. By default it supports email notification via the _EmailNotificationStrategy_.

    <img src="./assets/Image Processing Flow Chart.png" width="100%" alt="Image Processing Flow Diagram.">

    _Figure 1: Flow Diagram of the image processing architecture. This chain of image processing steps constitutes the _CV Pipeline_._

    ---
* __Extending the Code__: <br>
    * __Changing the Underlying Implementation of a Certain Image Processing Step (CVPipelineStep)__: <br>
    The image processing steps (e.g. _ChangeDetector_, ..., _Notifier_) make use of the Strategy design pattern. This allows one to treat a faimly of different algorithms as one and switch between them dynamically. One can use a different algorithm by implementing a concrete implementation of the corresponding strategy interface using their choice of algorithm, and configuring the code to use it. For example, the __Notifier__ uses a implementation of the __NotificationStrategy__ interface to determine how the notificaitons are sent. By default it uses the __EmailNotificationStrategy__ hence it sends notification via an email. In order to use a different medium to send notifications one can create a new implementation of the __NotificaitonStrategy__ interface and configure __Notifier__ to use it instead of the __EmailNotificaitonStrategy__ to send notifications according to their underlying implementation.

    * __Adding a New Processing Step (CVPipelineStep) to the Image Processing Flow (CVPipeline)__: <br>
    The image processing steps (__CVPipelineStep__) are implemented using the Chain of Responsibility design pattern. This allows one to add and remove steps dynamically and easily. One can add a new step to the image processing pipeline (__CVPipeline__) by extending the __CVPipelineStep__ in the class definition of their new image processing step, and then adding a new instance of it to the image processing chain (__CVPipeline__) as a step. One should note that in order for the chain to function properly one should make sure that the inputs and the returned values from the \'handle()\' method of the __CVPipelineStep__ should match between the consecutive image processing steps (__CVPipelineStep__).
    
    <img src="./assets/class diagram.png" width="100%" alt="Class Diagram">

    _Figure 2: Class Diagram with an emphasis on the Strategy, the Chain of Responsiblity, and the Builder Design patterns used. I suggest viewing it by write clicking  on the image and choosing \'open image in new tab\' which allows one to zoom in._
---

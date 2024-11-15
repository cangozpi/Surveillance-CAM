from object_detection.objectDetectionStrategy import ObjectDetectionStrategy
import torch
import torchvision
import cv2
import numpy as np
from torchvision import transforms


class MobileNetObjectDetectionStrategy(ObjectDetectionStrategy):
    """
    Concrete implementation of the ObjectDetectionStrategy (Strategy pattern) which can be used by the ObjectDetector. 
    Performs object detection using a pretrained MobileNetV3 (refer to https://pytorch.org/vision/0.9/models.html).

    Note that: The torchvision detection models (from torchvision model_zoo) expect inputs to be in a 0-1 range. The input images are 
    automatically scaled and normalized during the processing pipeline inside the model.
    In other words, the models expect a list of Tensor[C, H, W], in the range 0-1. The models internally resize the images so that 
    they have a minimum size of 800. This option can be changed by passing the option min_size to the constructor of the models.

    Design Pattern:
        Concrete implementation of the Strategy interface of the Strategy design pattern (ObjectDetectionStrategy).
    """
    def __init__(self):
        super().__init__()

        # Load the pretrained MobileNet model with Faster R-CNN from torchvision
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_320_fpn(pretrained=True)
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        # self.model = quantized = torchvision.models.quantization.mobilenet_v3_large(pretrained=True)
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(self.device)

        # Define the COCO class labels (object detector models from torchvision model_zoo are pretrained on ImageNet then 
        # finetuned on MSCOCO object detetion task hence they use the COCO class labels below)
        self.COCO_CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.transform = transforms.Compose([
            # Convert the image to a PyTorch tensor 
            # (Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] )
            transforms.ToTensor(),
        ])
    

    def executeObjectDetection(self, frame):
        """
        Performs object detection on the passed frame and returns the detections.

        Inputs:
            frame (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
        Returns:
            boxes (np.ndarray of shape [N, 4=[x1, y1, x2, y2]]): returns the bounding box coordinates for the given detections, where (x1, y1)  
                corresponds to the top left corner of the bounding box, and (x2, y2) corresponds to the bottom right corner of the bounding box.
            labels (list of strings): contains labels (categories) for the detected objects as strings (not class id's). Each label's corresponding
                bounding box is in the boxes array's same index.
            scores (np.ndarray of shape [N,]): contains the confidence scores for the detections. Each index in the array has the confidence
                score of the detected object at the same index in the boxes and labels arrays.
        """
        # Preprocess the input image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing with the model [H, W, C]
        input_tensor = self.transform(image_rgb) # [C, H, W]
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension [N, C, H, W]
        assert(input_batch.shape == (1, 3, *input_tensor.shape[1:3]))

        # Check if a GPU is available and move the model to GPU if possible
        input_batch = input_batch.to(self.device)

        # Perform inference
        with torch.no_grad():  # Disable gradient calculation for inference
            output = self.model(input_batch)

        # The model returns a dictionary containing:
        # - boxes: list of bounding box coordinates for detected objects
        # - labels: list of class indices for the detected objects
        # - scores: list of confidence scores for the detected objects
        boxes = output[0]['boxes'].detach().cpu().numpy()  # Bounding boxes
        label_ids = output[0]['labels'].detach().cpu().numpy()  # Class labels as id's
        labels = list(map(lambda x: self.COCO_CLASSES[x], label_ids)) # convert class label id's to strings labels
        scores = output[0]['scores'].detach().cpu().numpy()  # Confidence scores

        return boxes, labels, scores


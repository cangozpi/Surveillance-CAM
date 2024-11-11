from detected_person_identification.detectedPersonIdentificationStrategy import DetectedPersonIdentificationStrategy
import torch
from torch import nn
import torchvision
from torchvision import transforms
import cv2
import numpy as np


class KNN_DetectedPersonIdentificationStrategy(DetectedPersonIdentificationStrategy):
    """
    Concrete implementation of the DetectedPersonIdentificationStrategy (Strategy pattern) which can be used by the DetectedPersonIdentifier. 
    Performs person identification of detected people by comparing their embeddings vectors against the embedding vectors of the known identities
    in a given identity database. It takes a KNN approach and matches the people with the class with the closest embedding within a threshold value.
    If the distance between the closest embeddings exceeds a threshold then the person is identified as an unknown person.

    Design Pattern:
        Concrete implementation of the Strategy interface of the Strategy design pattern (DetectedPersonIdentificationStrategy).
    """
    def __init__(self, threshold_norm_dist=100):
        super().__init__()
        self.embedding_model = MobileNetImageFeaturesNetwork() # pretrained model to use as feature extractor

        self.db_embed_dict = None  # db_embed_dict (dict | keys: strin | values: np array): person's name as the key, and the np array of 
                                    #     shape [num_pictures_of_the_person, embed_dim] where indexing by each row returns the embedding vector extracted from the person's
                                    #     image at the same index in the db_dict's value.

        self.threshold_norm_dist:float = threshold_norm_dist # if the norm distance between a person detections embedding vector and a known persons embedding vector is
                                        # smaller than this threshold value then that name is assigned to that person detection, else no name is assigned.


    def executePersonIdentification(self, cropped_person_images, db_dict):
        """
        Given a list of cropped out person detection and a dictionary containing known people names and paths to their associated
        images in the KnownPersonDb, it checks if the cropped person detections match any of the known people from the db. If
        it matches then a name is assigned to that detection, if not no name is assigned. The final list of assigned names are
        returned by this function as a list in the end.

        Inputs:
            cropped_person_images (list of image 2D BGR image of shape [H,W,C=3]): contains the cut out images of the detected 
                people in the given frame.
            db_dict (dict | keys: string | values: array of np arrays): person's name as the key, and the array of path names (string)
                to the images that person had in the database.
        
        Returns:
            cropped_person_names (list of string): a list where each index holds the name (str) assigned to the person detection given
                at the same index in the cropped_person_images. It holds the value of <None> (not string 'None') when the detection
                was not assigned a name (i.e. when detection did not match any known person in the db_dict (KnownPersonDb)).
        """
        # If embedding vectors of the known people have not been extracted before, then first compute and cache them for future use.
        if self.db_embed_dict is None:
            self.initDbEmbedDict(db_dict)

        # Extract feature embeddings for the detected cropped_person_images
        cropped_person_embeddings = np.zeros((len(cropped_person_images), self.embedding_model.embedding_dim), dtype=np.float32)
        for i, cropped_image in enumerate(cropped_person_images):
            try:
                embedding_vector = self.getEmbeddingVector(cropped_image) # embedding_vector (np array of shape [1, embedding_dim]):
            except:
                breakpoint()
            assert(embedding_vector.shape == (1, self.embedding_model.embedding_dim))
            cropped_person_embeddings[i,:] = embedding_vector[0]
        
        # Find best matching known embedding vector for each embedding of the cropped_person_image
        cropped_person_names = [] # contains assigned names for the person detection in the cropped_person_images. When a name is not assigned
                                    # its index has the value None
        for i in range(cropped_person_embeddings.shape[0]):
            cur_cropped_person_embedding = np.expand_dims(cropped_person_embeddings[i], axis=0) # [1, embedding_dim]
            assert(cur_cropped_person_embedding.shape == (1, self.embedding_model.embedding_dim))

            # Find difference of the current cropped person embedding with all of the embeddings available in the known person db
            cur_cropped_persons_min_norm = None
            cur_cropped_persons_name = None
            for cur_person, cur_person_embeddings in self.db_embed_dict.items():
                embedding_diff_norms = np.linalg.norm(cur_person_embeddings - cur_cropped_person_embedding, axis=1) # [num_embeddings_of_the_person, ]
                assert(embedding_diff_norms.shape == (cur_person_embeddings.shape[0],)) 
                # get the minimum norm value among the embedding vectors of the current person
                cur_min_norm = np.min(embedding_diff_norms)
                if (cur_cropped_persons_min_norm is None) or (cur_cropped_persons_min_norm > cur_min_norm):
                    cur_cropped_persons_min_norm = cur_min_norm
                    cur_cropped_persons_name = cur_person

            if cur_cropped_persons_min_norm < self.threshold_norm_dist:  # assign person name to the person detection
                cropped_person_names.append(cur_cropped_persons_name)
            else: # no match for the detected person
                cropped_person_names.append(None)

        assert(len(cropped_person_names) == len(cropped_person_images))
        return cropped_person_names

    
    def initDbEmbedDict(self, db_dict):
        """
        Inputs:
            db_dict (dict | keys: string | values: array of np arrays): person's name as the key, and the array of path names (string)
                to the images that person had in the database.

        Returns:
            db_embed_dict (dict | keys: strin | values: np array): person's name as the key, and the np array of 
                shape [num_pictures_of_the_person, embed_dim] where indexing by each row returns the embedding vector extracted from the person's
                image at the same index in the db_dict's value.
        """
        self.db_embed_dict = {} # like db_dict but for its values it contanis np array of extracted embeddings for the pictures instead of the image absolute paths of a person.
        for person_name, person_images_array in db_dict.items():
            cur_embeddings_arr = np.zeros((len(person_images_array) ,self.embedding_model.embedding_dim), dtype=np.float32) # [num_images, embedding_dim]
            for i, img_path in enumerate(person_images_array):
                cur_img_arr = cv2.imread(img_path) # BGR image np array with shape [H, W, C=3]
                assert(isinstance(cur_img_arr, np.ndarray) and (len(cur_img_arr.shape) == 3) and (cur_img_arr.shape[2] == 3))
                cur_img_embedding = self.getEmbeddingVector(cur_img_arr) # embedding_vector (np array of shape [1, embedding_dim]):
                assert(cur_img_embedding.shape == (1, self.embedding_model.embedding_dim))
                cur_embeddings_arr[i, :] = cur_img_embedding[0,:]
            self.db_embed_dict[person_name] = cur_embeddings_arr
        assert(set(self.db_embed_dict.keys()) == set(db_dict.keys()))
                

    def getEmbeddingVector(self, cropped_person_image):
        """
        Inputs:
            frame (a 2D BGR image of shape [H,W,C=3]): contains the cut out image of the detected person.

        Returns:
            embedding_vector (np array of shape [1, embedding_dim]):
        """
        frame = cv2.cvtColor(cropped_person_image, cv2.COLOR_BGR2RGB) # convert BGR image ot RGB image
        embedding_vector = self.embedding_model(frame)
        assert(embedding_vector.shape == (1, self.embedding_model.embedding_dim) and (isinstance(embedding_vector, np.ndarray)))
        return embedding_vector



# ---
# ----- Helper classes/functions below:


class MobileNetImageFeaturesNetwork(nn.Module):
    embedding_dim = None # dimension of the embedding vectors
    def __init__(self):
        """
        Uses a pretrained MobileNet model from the torchvision model_zoo (refer to: https://pytorch.org/vision/0.9/models.html) to 
        extract features embeddings (representations) for given images which can then be used for person identification by comparing the
        extracted feature embeddings from the two images. The output of the last fully connected layer of the MobileNet model is used as
        the extracted features.
        """
        super().__init__()
        self.mobilenet_v3_small = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.mobilenet_v3_small.fc = torch.nn.Identity() # remove the classification head
        MobileNetImageFeaturesNetwork.embedding_dim = self.mobilenet_v3_small.classifier[-1].out_features

        self.eval() # this model will only be used for inference
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mobilenet_v3_small = self.mobilenet_v3_small.to(self.device)

        self.transform = transforms.Compose([ # normalize according to https://pytorch.org/vision/0.9/models.html
            transforms.ToTensor(), # convert numpy.ndarray [H, W, C=3] in the range [0, 255] to a torch.FloatTensor of shape [C=3, H, W] in the range [0.0, 1.0]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def forward(self, x):
        """
        Takes in a single 2D BGR image and returns its corresponding feature (embedding) vector.

        Inputs:
            x (np array 2D BGR image of shape [H,W,C=3]): contains the cut out images of the detected people in the given frame.
        
        Returns:
            embedding (np array of shpae [1, embedding_dim]): embedding vector extracted from the input image x
        """

        x = self.transform(x)
        x = torch.unsqueeze(x, axis=0) # add batch dimension [1, C, H, W]
        x = x.to(self.device)
        with torch.no_grad():
            embedding = self.mobilenet_v3_small(x)
            embedding = embedding.detach().cpu().numpy()
        assert (embedding.shape == (1, MobileNetImageFeaturesNetwork.embedding_dim))
        return embedding
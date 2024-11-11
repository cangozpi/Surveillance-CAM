from abc import ABC, abstractmethod

class DetectedPersonIdentificationStrategy(ABC):
    """
    Implement this class to create an algorithm which can be used interchangeably with other algorithms to perform person identification of 
    people detected on an image by comparing them against the entries in an available database of identities. 

    Design Pattern:
        Strategy interface of the Strategy design pattern (DetectedPersonIdentificationStrategy).
    """
    def __init__(self):
        pass

    @abstractmethod
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
        pass
from abc import ABC, abstractmethod
import os

class KnownPersonDb(ABC):
    def __init__(self):
        """
        This class abstracts reading and writing of known person to the database
        Subclass this class to have an implementation of the db.
        This class is instantiated and used by the DetectedPersonIdentifier class. This class is also Singleton which means that its 
        constructor should not be called directly to instantiate this class. Instead, KnownPersonDb.getInstance() should be used to get an
        instance.

        Design Pattern:
            - This class uses Singleton pattern as there should only be 1 instance of this class at runtime
        """
        self.singleton_instance = None # holds reference to the only instance of this class
 
    @abstractmethod
    def getInstance(self):
        """
        If there is no instance of this object it instantiates a new object with the passed in parameters and returns a reference to it.
        If there is an instance from before then it is returned instead. Note that in the case there was already an instance, the passed in parameters
        would not be used for object instantiation and hence the returned object might not be initialized according to the passed in parameters.

        Returns:
            db_instance (a concrete implementation of KnownPersonDb): a concrete implementation of KnownPersonDb class
        """
        pass


    @abstractmethod
    def get_db_dict(self):
        """
        Returns:
            db_dict (dict | keys: string | values: array of np arrays): persons name as the key, and the array of feature vectors (np arrays)
                extracted from the images that person had in the database.
        """



# ---
# ----- concrete implementations of KnownPersonDb are below:

class KnownPersonFolderDb(KnownPersonDb):
    def __init__(self, db_path='known_people_db'):
        """
        This class abstracts reading and writing of known person to the database (not a real db, but a folder with specific structure).
        The db is implemented as structured folder containing subfolders with known people's names along with their corresponding images.
        This class is instantiated and used by the DetectedPersonIdentifier class. This class is also Singleton which means that its 
        constructor should not be called directly to instantiate this class. Instead, KnownPersonDb.getInstance() should be used to get an
        instance.

        Params:
            db_path (str): relative path from this file (without the trailing './' prefix) to the folder that contains the known people images. The db should contain 
                the known person's name as the name of the subfolder which includes any number of images that belong to that person.

        Design Pattern:
            - This class uses Singleton pattern as there should only be 1 instance of this class at runtime
        """
        super().__init__()
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.db_path = os.path.join(script_dir, db_path) 

        # read in the db
        self.db_dict = self.read_db_paths() # dict with key(string): person name, value (array of string): contains absolute paths to the images of the corresponding person.
    
    def getInstance(self, db_path='./known_people_db'):
        """
        If there is no instance of this object it instantiates a new object with the passed in parameters and returns a reference to it.
        If there is an instance from before then it is returned instead. Note that in the case there was already an instance, the passed in parameters
        would not be used for object instantiation and hence the returned object might not be initialized according to the passed in parameters.

        Inputs:
            db_path (str): relative path from this file (without the trailing './' prefix) to the folder that contains the known people images. The db should contain 
                the known person's name as the name of the subfolder which includes any number of images that belong to that person.
        
        Returns:
            db_instance (KnownPersonFolderDb): an instance of the KnownPersonFolderDb
        """
        if self.singleton_instance is None:
            self.singleton_instance = KnownPersonFolderDb(db_path)
        return self.singleton_instance

    def read_db_paths(self):
        """
        Read in the db specified by the self.db_path parameter and returns a dict (db) which has the name of the known people
        as its key, and the corresponding array of absolute path names to images (string) as values where each path corresponds to 
        an image that person had registered in the db_path. 

        Returns:
            db_dict (dict | keys: string | values: array of np arrays): persons name as the key, and the array of path names (string)
                to the images that person had in the database.
        """
        db_dict = {}
        for root, dirs, files in os.walk(self.db_path):
            cur_person_name = root.split(f'{os.sep}')[-1]# name of the known person
            cur_image_paths = []
            for file in files:
                # Check if the file is an image (based on extension)
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Construct the full path of the image file
                    cur_image_paths.append(os.path.join(root, file))
            if len(cur_image_paths) > 0:
                db_dict[cur_person_name] = cur_image_paths
        return db_dict
    
    def get_db_dict(self):
        """
        Returns:
            db_dict (dict | keys: string | values: array of np arrays): persons name as the key, and the array of path names (string)
                to the images that person had in the database.
        """
        return self.db_dict

    
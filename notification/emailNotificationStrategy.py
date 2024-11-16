from notification.notificationStrategy import NotificationStrategy
import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import os
from io import BytesIO
from time import gmtime, strftime
import os

class EmailNotificationStrategy(NotificationStrategy):
    """
    Concrete implementation of the NotificationStrategy (Strategy pattern) which can be used by the Notifier. 
    Notifies the user via an email.

    Design Pattern:
        Concrete implementation of the Strategy interface of the Strategy design pattern (NotificationStrategy).
    """
    def __init__(self):
        super().__init__()
        # Email details
        self.sender_email = os.getenv('EMAIL')
        self.receiver_email = os.getenv('EMAIL')
        self.password = os.getenv('EMAIL_PASSWORD')  # Or app-specific password (if using Gmail, for example)

        # Setup the SMTP server
        self.smtp_server = "smtp.gmail.com"  # For Gmail, use Gmail's SMTP server
        self.smtp_port = 587  # For TLS (587), 465 for SSL

    def executeNotify(self, msg: str, frame):
        """
        Notifies the user about the given info.
        Inputs:
            msg (str): text message
            frame (np.ndarray of shape [H, W, C=3]): input image 2D BGR image of shape [H,W,C=3]
        """
        # Set up the MIME (Multi-Purpose Internet Mail Extensions)
        message = MIMEMultipart()
        message['From'] = self.sender_email
        message['To'] = self.receiver_email
        message['Subject'] = f'Surveillance CAM Notification [{strftime("%Y-%m-%d %H:%M:%S", gmtime())}]'

        # Email body (text)
        body = msg
        message.attach(MIMEText(body, 'plain'))

        # Convert the BGR NumPy ndarray to an image in memory
        image_bytes = self.convert_bgr_to_image_bytes(frame, image_format="JPEG")

        # Set image filename
        image_filename = 'Annotated Surveillance Footage'

        # Attach the image inline (instead of as an attachment)
        image_part = MIMEImage(image_bytes.read(), name="Annotated Frame.jpg")
        image_part.add_header('Content-ID', '<image1>')  # This is the reference used in the email body
        image_part.add_header('Content-Disposition', 'inline')  # Ensure it's inline

        # Add headers to the attachment part
        image_part.add_header('Content-Disposition', f'attachment; filename={image_filename}')

        # Attach the image to the email message
        message.attach(image_part)

        try:
            # Create a secure connection with the SMTP server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()  # Start TLS (transport layer security)
    
            # Login to the SMTP server
            server.login(self.sender_email, self.password)
    
            # Send the email
            text = message.as_string()
            server.sendmail(self.sender_email, self.receiver_email, text) #TODO: send email async or in a new thread
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")
        finally:
            server.quit()  # Close the connection

    def convert_bgr_to_image_bytes(self, bgr_image, image_format="JPEG"):
        """
        Function to convert a NumPy ndarray (BGR) to a file-like object in PNG or JPEG format (i.e. converts to bytes).
        """
        _, buffer = cv2.imencode(f".{image_format.lower()}", bgr_image)
        image_bytes = BytesIO(buffer.tobytes())
        return image_bytes

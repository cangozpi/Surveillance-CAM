from enum import Enum

class Available_NotificationStrategies(Enum):
    """
    Enums used by the IPipelineBuilder to set the NotificationStrategy of the Notifier
    accordingly from the available implementations of the NotificationStrategy. There should
    be an enum for every concrete implementation of the NotificationStrategy.
    """
    EMAIL_NOTIFICATION_STRATEGY = 1

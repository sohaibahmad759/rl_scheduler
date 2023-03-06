
class ConfigException(Exception):
    """Exception raised for errors in experiment configuration file.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

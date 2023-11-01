class BusinessException(Exception):

    def __init__(self, message, status_code=409):
        self.message = message
        self.status_code = status_code
        super(BusinessException, self).__init__(message)

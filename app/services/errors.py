class ServiceError(Exception):
    status = 400


class ValidationFailure(ServiceError):
    status = 422


class Conflict(ServiceError):
    status = 409

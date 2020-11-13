class Status:
    NOT_STARTED = "not_started"
    RUNNING = "running"
    EXCEPTION = "exception"
    INTERRUPTED = "interrupted"
    FINISHED = "finished"

    @staticmethod
    def all():
        return (
            Status.NOT_STARTED,
            Status.RUNNING,
            Status.EXCEPTION,
            Status.INTERRUPTED,
            Status.FINISHED,
        )

    @staticmethod
    def validate(value: str):
        options = Status.all()
        if value not in options:
            raise ValueError(f"Invalid Status {value}: must be one of {options}")

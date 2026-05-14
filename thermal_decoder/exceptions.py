"""Application-specific exceptions."""


class System_OCV_Vis_Temp_Error(Exception):
    """Raised when the temperature scale gradient is discontinuous or invalid."""

    default_message = (
        "System_OCV_Vis_Temp_Error: Ошибка анализа шкалы температур"
    )

    def __init__(self, message: str | None = None, *, detail: str | None = None):
        if message is not None:
            super().__init__(message)
            self.detail = detail
        elif detail:
            super().__init__(f"{self.default_message}\n\n{detail}")
            self.detail = detail
        else:
            super().__init__(self.default_message)
            self.detail = None

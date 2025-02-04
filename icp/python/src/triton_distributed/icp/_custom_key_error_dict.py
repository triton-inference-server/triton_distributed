from typing import Type


class CustomKeyErrorDict(dict):
    def __init__(
        self,
        from_name: str,
        to_name: str,
        *args,
        exception: Type[Exception] = ValueError,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._to_name = to_name
        self._from_name = from_name
        self._exception = exception

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise self._exception(
                f"Unsupported {self._from_name}. Can't convert {key} to {self._to_name}"
            ) from None

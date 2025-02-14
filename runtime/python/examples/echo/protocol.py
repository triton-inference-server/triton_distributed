import msgspec


class User(msgspec.Struct):
    user: str


class Id(msgspec.Struct):
    id: str

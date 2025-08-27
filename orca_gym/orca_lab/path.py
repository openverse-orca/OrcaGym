class Path:

    def __init__(self, p: str = "/"):
        if not self.is_valid_path(p):
            raise Exception("Invalid path.")
        self._p = p

    def append(self, name: str):
        if not isinstance(name, str):
            raise Exception("Invalid argument.")

        if not self.is_valid_name(name):
            raise Exception("Invalid name.")

        if self == self.root_path():
            return Path(self._p + name)

        return Path(self._p + "/" + name)

    def __truediv__(self, other):
        return self.append(other)

    def string(self):
        return self._p

    def __str__(self):
        return self._p

    def __repr__(self):
        return f'Path("{self._p}")'

    def __hash__(self):
        return hash(self._p)

    def __eq__(self, other):
        return self._p == other._p

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    @classmethod
    def is_valid_name(cls, name: str) -> bool:
        # An identifier can only contain alphanumeric characters (a-z, A-Z, 0-9) and underscores (_)
        # We ignore Unicode here.
        return name.isascii() and name.isidentifier()

    @classmethod
    def is_valid_path(cls, path: str) -> bool:
        if path == "/":
            return True

        if not path.startswith("/"):
            return False

        rest = path[1:]
        tokens = rest.split("/")

        for token in tokens:
            if not cls.is_valid_name(token):
                return False

        return True

    @classmethod
    def root_path(cls):
        return Path("/")

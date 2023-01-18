from houghttranformation import HoughtTransformation


class Image(HoughtTransformation):
    def __init__(self, path: str = None) -> None:
        super().__init__(path)
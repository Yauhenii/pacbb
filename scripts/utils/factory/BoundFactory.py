from core.bound import AbstractBound, KLBound, McAllesterBound
from scripts.utils.factory import AbstractFactory


class BoundFactory(AbstractFactory[AbstractBound]):
    def __init__(self) -> None:
        super().__init__()
        self.register_creator("kl", KLBound)
        self.register_creator("mcallester", McAllesterBound)

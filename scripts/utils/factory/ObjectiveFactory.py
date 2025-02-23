from core.objective import (
    AbstractObjective,
    BBBObjective,
    FClassicObjective,
    FQuadObjective,
    McAllesterObjective,
    TolstikhinObjective,
)
from scripts.utils.factory import AbstractFactory


class ObjectiveFactory(AbstractFactory[AbstractObjective]):
    def __init__(self) -> None:
        super().__init__()
        self.register_creator("bbb", BBBObjective)
        self.register_creator("fclassic", FClassicObjective)
        self.register_creator("fquad", FQuadObjective)
        self.register_creator("mcallester", McAllesterObjective)
        self.register_creator("tolstikhin", TolstikhinObjective)

from abc import ABC, abstractmethod

class BaseSelector(ABC):
    @abstractmethod
    def select(self, dataset, model, budget):
        raise NotImplementedError

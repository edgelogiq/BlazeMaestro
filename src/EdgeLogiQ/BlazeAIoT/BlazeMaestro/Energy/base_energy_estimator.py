from abc import ABC, abstractmethod


class BaseEnergyEstimator(ABC):
    @abstractmethod
    def estimate_service_energy(self, service, node):
        pass

    @abstractmethod
    def apply_service(self, service, node):
        pass

    @abstractmethod
    def get_battery_level(self):
        pass

    @abstractmethod
    def get_battery_history(self):
        pass

    @abstractmethod
    def reset(self, battery_level=None):
        pass

"""
Service Provider abstraction layer.

This module provides a base ServiceProvider interface and implementations for both
simulator (XML-based) and real Kubernetes cluster environments.
"""

from abc import ABC, abstractmethod
from typing import List
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service import Service


class ServiceProvider(ABC):
    """
    Abstract base class for service providers.

    A ServiceProvider is responsible for discovering and providing service
    definitions. Implementations can:
    - Load from XML files
    - Query from Kubernetes namespaces
    - Read from configuration databases
    - Generate programmatically
    """

    @abstractmethod
    def get_services(self) -> List[Service]:
        """
        Get all service definitions.

        Returns:
            List of Service objects
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get a human-readable name for this service provider.

        Returns:
            Provider name (e.g., "XML File", "Kubernetes namespace:production")
        """
        pass


class ServiceManager:
    """
    Manages services through dependency injection.

    This class acts as a facade that accepts any ServiceProvider implementation
    and provides a unified interface for service management.
    """

    def __init__(self, service_provider: ServiceProvider):
        """
        Initialize the service manager with a service provider.

        Args:
            service_provider: ServiceProvider implementation (XML, Kubernetes, etc.)
        """
        self.service_provider = service_provider
        self.services = None

    def load_services(self) -> List[Service]:
        """
        Load services from the injected provider.

        Returns:
            List of Service objects
        """
        self.services = self.service_provider.get_services()
        return self.services

    def get_services(self) -> List[Service]:
        """
        Get all services (loads if not already loaded).

        Returns:
            List of Service objects
        """
        if self.services is None:
            self.load_services()
        return self.services

    def get_provider_name(self) -> str:
        """
        Get the name of the service provider.

        Returns:
            Provider name
        """
        return self.service_provider.get_name()

    def __str__(self):
        """String representation."""
        services_count = len(self.services) if self.services else 0
        return f"ServiceManager({self.get_provider_name()}: {services_count} services)"


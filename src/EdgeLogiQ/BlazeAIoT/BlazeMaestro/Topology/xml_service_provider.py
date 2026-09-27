"""
XML-based Service Provider implementation.

Loads service definitions from XML files using the existing service_loader.
"""

from typing import List
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service_provider import ServiceProvider
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service import Service


class XMLServiceProvider(ServiceProvider):
    """
    Service provider that loads services from XML files.

    Uses the existing ServiceDefinitionLoader to parse XML.
    """

    def __init__(self, xml_path: str, simpy_env=None):
        """
        Initialize XML service provider.

        Args:
            xml_path: Path to services XML file
            simpy_env: Optional SimPy environment for ServiceSimulator instances
        """
        self.xml_path = xml_path
        self.simpy_env = simpy_env

    def get_services(self) -> List[Service]:
        """
        Load services from XML file.

        Returns:
            List of Service objects (or ServiceSimulator in simulator mode)
        """
        from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service_loader import load_services_from_xml

        return load_services_from_xml(self.xml_path, simpy_env=self.simpy_env)

    def get_name(self) -> str:
        """Get provider name."""
        import os
        return f"XML File ({os.path.basename(self.xml_path)})"


class ProgrammaticServiceProvider(ServiceProvider):
    """
    Service provider that returns a pre-defined list of services.

    Useful for hardcoded scenarios or testing.
    """

    def __init__(self, services: List[Service], name: str = "Programmatic"):
        """
        Initialize with a list of services.

        Args:
            services: List of Service objects
            name: Provider name
        """
        self.services = services
        self.provider_name = name

    def get_services(self) -> List[Service]:
        """
        Return the pre-defined list of services.

        Returns:
            List of Service objects
        """
        return self.services

    def get_name(self) -> str:
        """Get provider name."""
        return self.provider_name


"""
BlazeAIoT Maestro Application Module

This module contains the main BlazeMaestroApp class that encapsulates
the orchestration logic, service management, and Kubernetes integration.
"""

from BlazeAIoT.Core.NodeManager import ServiceManager
from BlazeAIoT.Core.Utils import BlazeLogger
from BlazeAIoT.Messages.Platform import BlazeMessage
from BlazeAIoT.Core.NodeManager import BlazeService

from .blaze_maestro import BlazeMaestro

class BlazeMaestroApp(BlazeService):
    """Main BlazeAIoT Maestro application."""

    def __init__(self, args):
        ServiceManager.create_instance(self, args.layer_config_path, args.node_config_path, args.service_config_path)
        # Initialize logger after service manager is created (requires configuration)
        BlazeLogger.create_instance(ServiceManager.get_instance().configuration)
        self._logger = BlazeLogger.get_instance()

        ServiceManager.get_instance().config_manager.register_node_callback(self._delete_node_callback,
                                                                            self._update_node_callback)
        ServiceManager.get_instance().config_manager.register_service_callback(self._delete_service_callback,
                                                                               self._update_service_callback)
        ServiceManager.get_instance().config_manager.register_layer_callback(self._delete_layer_callback,
                                                                             self._update_layer_callback)
        
        # Get config file path from service config parameters
        config_file = ServiceManager.get_instance().configuration.service_config['parameters']['configFile']
        
        self.service = BlazeMaestro(config_file)

        # Wire CLI flags to config-driven update path
        self._update_configs = bool(args.update_configs)
        self._update_configs_only = bool(args.update_configs_only)
        if self._update_configs or self._update_configs_only:
            # Enable Prometheus-driven updates for this run
            self.service.config_loader.config['update_configs_from_prometheus'] = True

    def stop(self):
        self.service.stop()
        ServiceManager.get_instance().dispose()

    async def start(self):
        # Initialize BlazeMaestro before starting
        init_ok = self.service.initialize()
        if not init_ok:
            self._logger.error("Initialization failed; aborting")
            return

        # If only updating configs, exit after initialize (the update happens inside initialize)
        if self._update_configs_only:
            print("\n✓ Configuration update complete (not starting orchestration)")
            return

        await self.service.start()

    def manage_new_config(self):
        BlazeLogger.get_instance().info("Config updated")

    def _delete_node_callback(self):
        BlazeLogger.get_instance().info("Node deleted")

    def _update_node_callback(self, command: BlazeMessage):
        BlazeLogger.get_instance().info("Node updated")

    def _delete_service_callback(self):
        BlazeLogger.get_instance().info("Service deleted")

    def _update_service_callback(self, command: BlazeMessage):
        BlazeLogger.get_instance().info("Service updated")

    def _delete_layer_callback(self):
        BlazeLogger.get_instance().info("Layer deleted")

    def _update_layer_callback(self, command: BlazeMessage):
        BlazeLogger.get_instance().info("Layer updated")

# Configuration Files

This directory contains the configuration files for BlazeAIoT Maestro.

## Files

### platform.xml
Platform topology definition for the simulator.

Defines:
- Node infrastructure (cloud, fog, edge, mobile layers)
- Node resources (CPU, RAM, GPU)
- Network topology and links
- Battery and power configurations

**Usage**:
```python
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.simulator_platform import SimulatorPlatform

platform = SimulatorPlatform(env, "config/platform.xml")
topology_manager = PlatformTopologyManager(platform)
layers, nodes = topology_manager.load_topology()
```

### services.xml
Service definitions for the Blaze scenario.

Defines:
- Service workloads (infrastructure and robot services)
- Container specifications (CPU, RAM, GPU requirements)
- Service dependencies
- Placement constraints (layer locks, node locks, daemonsets)
- Execution rates

**Usage**:
```python
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.xml_service_provider import XMLServiceProvider

service_provider = XMLServiceProvider("config/services.xml")
service_manager = ServiceManager(service_provider)
services = service_manager.load_services()
```

## Customization

### For Development
Edit these XML files to customize your simulation:
- Add/remove nodes in platform.xml
- Modify node resources
- Add/remove services in services.xml
- Change service resource requirements
- Adjust placement constraints

### For Production
In production Kubernetes environments, you typically:
- Use MultiKubernetesPlatform (`Topology/kubernetes_platform.py`) instead of XML for nodes
- Discover services through the Kubernetes service discovery (`Executors/kubernetes_service_discovery.py`)
- Or use these XML files as templates/overrides

# Service Allocation Module

## Overview

The ServiceAllocation module provides a **Facade pattern** implementation for service allocation in BlazeAIoT Maestro. It enables dynamic service movement across nodes and layers in both Kubernetes clusters and simulation environments.

### Key Features

- **Unified Interface**: Single API for both Kubernetes and Simulator modes
- **Event-Driven**: Publishes allocation events to ServiceManager Broker for tracking
- **Graceful Allocation Flow**: Multi-step process with broker confirmation support
- **Object-Oriented Design**: Clean separation of concerns with abstract base class
- **Easy Integration**: Simple integration into BlazeMaestroApp

## Architecture

### Components

```
ServiceAllocationFacade (Entry Point)
    ├── KubernetesAllocator (kubectl-based implementation)
    ├── SimulatorAllocator (in-memory implementation)
    └── BaseAllocator (abstract interface)
        └── AllocationEvent (event model for broker)
```

### Allocation Flow

```
1. ALLOCATION_INITIATED
   ↓
2. Scale up target deployment to 1 replica
   ↓
3. ALLOCATION_IN_PROGRESS
   ↓
4. Wait for ServiceManager Broker confirmation (PLACEHOLDER)
   ↓
5. Scale down source deployment to 0 replicas
   ↓
6. ALLOCATION_COMPLETED
```

## Usage

### Basic Usage - Kubernetes Mode

```python
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.ServiceAllocation import ServiceAllocationFacade

# Initialize facade for Kubernetes
facade = ServiceAllocationFacade.create_for_kubernetes(
    kubeconfig_path='/path/to/kubeconfig',
    context='prod-context'
)

# Allocate service
success = facade.allocate(
    service_id='vision-service',
    target_node='edge-gpu1',
    target_layer='edge',
    source_node='fog-cpu1',
    source_layer='fog',
    namespace='workloads'
)

if success:
    print("Service allocated successfully")
```

### Basic Usage - Simulator Mode

```python
# Initialize facade for Simulator
facade = ServiceAllocationFacade.create_for_simulator()

# Allocate service (no namespace needed in simulator)
success = facade.allocate(
    service_id='robot-slam',
    target_node='edge-cpu1',
    target_layer='edge',
    source_node='fog-gpu1',
    source_layer='fog'
)
```

### Integration with BlazeMaestroApp

```python
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.blaze_maestro_app import BlazeMaestroApp

# Initialize Maestro
maestro = BlazeMaestroApp(config)
maestro.initialize()

# Use allocation through Maestro app
success = maestro.allocate_service(
    service_id='my-service',
    target_node='edge-gpu1',
    target_layer='edge',
    source_node='fog-cpu1',
    source_layer='fog',
    namespace='default'
)
```

### With Broker Integration

```python
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.ServiceAllocation import ServiceAllocationFacade

# Assume you have a broker instance
broker = YourServiceManagerBroker()

# Create facade with broker
facade = ServiceAllocationFacade(
    mode='kubernetes',
    service_manager_broker=broker,
    verbose=True
)

# Allocation events will be published to broker
success = facade.allocate(
    service_id='service-x',
    target_node='edge-gpu1',
    target_layer='edge',
    source_node='fog-cpu1',
    source_layer='fog',
    namespace='default'
)
```

## Classes

### ServiceAllocationFacade

Main entry point for service allocation. Routes operations to appropriate backend.

**Methods:**
- `allocate()` - Move service to target node/layer
- `deallocate()` - Remove service from node/layer
- `get_replica_count()` - Check current replica count
- `get_allocator()` - Access underlying allocator instance

**Factory Methods:**
- `create_for_kubernetes()` - Create Kubernetes-backed facade
- `create_for_simulator()` - Create Simulator-backed facade

### BaseAllocator (Abstract)

Defines the interface all allocators must implement.

**Abstract Methods:**
- `allocate()` - Allocate service (implement allocation flow)
- `deallocate()` - Deallocate service
- `get_replica_count()` - Get current replica count

**Protected Methods:**
- `_publish_event()` - Publish event to broker
- `_create_event()` - Create AllocationEvent

### KubernetesAllocator

Kubernetes implementation using kubectl to manage Deployment replicas.

**Features:**
- Uses `kubectl patch` to modify replica counts
- Supports custom kubeconfig and context
- Implements full allocation flow with broker integration

**Assumptions:**
- Deployments exist with initial replica count of 0 (inactive)
- kubectl is available on the system
- Kubernetes credentials are configured

### SimulatorAllocator

In-memory simulator implementation for testing and development.

**Features:**
- Tracks allocations in-memory
- No external dependencies
- Useful for testing allocation logic without Kubernetes
- `get_all_allocations()` - Access all current allocations
- `clear_allocations()` - Reset state (for testing)

### AllocationEvent

Event model for broker communication.

**Event Types:**
- `ALLOCATION_INITIATED` - Allocation starting
- `ALLOCATION_IN_PROGRESS` - Target deployment scaled up
- `ALLOCATION_COMPLETED` - Allocation finished successfully
- `ALLOCATION_FAILED` - Allocation failed
- `DEALLOCATION_INITIATED` - Deallocation starting
- `DEALLOCATION_COMPLETED` - Deallocation finished

**Attributes:**
- `event_type` - Type of event
- `service_id` - Service identifier
- `source_node/layer` - Original placement
- `target_node/layer` - Destination placement
- `timestamp` - Event creation time
- `namespace` - Kubernetes namespace
- `deployment_name` - Kubernetes deployment name
- `error_message` - Error details if failed

## Broker Integration

The module includes a placeholder for ServiceManager Broker integration. Users should implement actual broker communication:

```python
class YourServiceManagerBroker:
    def publish(self, topic: str, event_dict: dict):
        """Publish allocation event."""
        # TODO: Implement actual broker communication
        # Example with message queue:
        # self.queue.put({
        #     'topic': topic,
        #     'event': event_dict
        # })
        pass
    
    def wait_for_confirmation(self, service_id: str, timeout: float):
        """Wait for allocation confirmation from service manager."""
        # TODO: Implement confirmation waiting logic
        # Example with event listener:
        # return self.listener.wait_for(
        #     f'allocation.confirmed.{service_id}',
        #     timeout=timeout
        # )
        pass

# Use with facade
broker = YourServiceManagerBroker()
facade = ServiceAllocationFacade(
    mode='kubernetes',
    service_manager_broker=broker
)
```

## Configuration

### Kubernetes Mode Configuration

```yaml
environment:
  mode: kubernetes
  kubernetes:
    clusters:
      - name: edge-cluster
        context: edge-context
        kubeconfig_path: /path/to/kubeconfig
        layer_id: edge
```

### Simulator Mode Configuration

```yaml
environment:
  mode: simulator
  simulator:
    platform_xml: src/config/platform_sim.xml
```

## Examples

See `examples.py` in this directory for complete working examples including:

- Kubernetes allocation
- Simulator allocation  
- Broker integration
- Integration with BlazeMaestroApp

## Best Practices

1. **Always check return value** - Verify allocation succeeded before proceeding
2. **Use appropriate namespace** - Specify correct Kubernetes namespace when available
3. **Log events** - Enable verbose mode for debugging complex allocation scenarios
4. **Handle failures gracefully** - Implement retry logic or fallback allocation strategies
5. **Broker integration** - Implement proper broker communication for production use

## Testing

Run the example scenarios:

```bash
# Simulate allocation in-memory
python3 src/EdgeLogiQ/BlazeAIoT/BlazeMaestro/ServiceAllocation/examples.py

# Test with Maestro app
python3 src/blaze_maestro_main.py --config src/config/blaze_maestro_simulator.yaml
```

## Future Enhancements

- [ ] Support for StatefulSets and DaemonSets
- [ ] Allocation policies (affinity, anti-affinity)
- [ ] Batch allocation operations
- [ ] Allocation history and audit trail
- [ ] Advanced broker integration patterns
- [ ] Multi-cluster allocation coordination
- [ ] Service mesh integration (Istio, Linkerd)

## Troubleshooting

### "ServiceManager Broker not available"
- Broker is optional. This message appears when publishing events without a configured broker.
- To enable broker integration, pass a broker instance to facade initialization.

### "Failed to patch deployment"
- Verify kubectl is installed: `kubectl version`
- Check kubeconfig: `kubectl config view`
- Verify deployment exists: `kubectl get deployment <name> -n <namespace>`

### Replicas not scaling
- Check deployment exists with 0 initial replicas
- Verify resources available on target node
- Check node selectors and affinity rules

## License

Part of BlazeAIoT Maestro orchestration framework.

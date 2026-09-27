"""
Orchestration Strategy Adapters

These adapters wrap existing orchestrator implementations to work with the Strategy pattern.
This allows existing orchestrator code to work without major rewrites.
"""

from typing import Dict, Sequence, Any, Optional, List
import os
from BlazeAIoT.Core.Utils import BlazeLogger
from .orchestration_strategy import OrchestrationStrategy, OrchestrationContext


class OrchestratorStrategyAdapter(OrchestrationStrategy):
    """
    Base adapter that wraps existing orchestrator implementations.
    Converts old API to new Strategy API by maintaining backward compatibility.
    """
    
    def __init__(
        self,
        orchestrator_class,
        context: OrchestrationContext,
        checkpoint_path: Optional[str] = None,
        **init_kwargs,
    ):
        """
        Initialize adapter.
        
        Args:
            orchestrator_class: The orchestrator class to instantiate
            context: OrchestrationContext with services, nodes, metrics
            **init_kwargs: Additional kwargs for orchestrator initialization
        """
        self.orchestrator_class = orchestrator_class
        self.context = context
        self.init_kwargs = init_kwargs
        self._orchestrator = None
        self._initialized = False
        self._last_allocation: Dict[str, str] = {}
        self._checkpoint_path = checkpoint_path
    
    def _ensure_initialized(self):
        """Lazy-initialize orchestrator with context data."""
        if self._initialized:
            return
        
        self._orchestrator = self.orchestrator_class(
            services=self.context.services,
            nodes=self.context.nodes,
            metrics=self.context.metrics,
            **self.init_kwargs
        )
        # Load checkpoint for orchestrators that support it (DQN, AutoEncoder, Unsupervised)
        if self._checkpoint_path and hasattr(self._orchestrator, "load_checkpoint"):
            if os.path.exists(self._checkpoint_path):
                try:
                    self._orchestrator.load_checkpoint(self._checkpoint_path)
                    BlazeLogger.get_instance().info(
                        f"Loaded checkpoint for {self.orchestrator_class.__name__}: {self._checkpoint_path}"
                    )
                except Exception as exc:
                    BlazeLogger.get_instance().warning(
                        f"Failed to load checkpoint for {self.orchestrator_class.__name__}: {exc}"
                    )
            else:
                BlazeLogger.get_instance().warning(
                    f"Checkpoint not found for {self.orchestrator_class.__name__}: {self._checkpoint_path}"
                )

        self._initialized = True
    
    def propose_assignment(self, context: OrchestrationContext, 
                          services: Optional[Sequence[Any]] = None) -> Dict[str, str]:
        """Generate assignment using wrapped orchestrator."""
        self._ensure_initialized()
        self._refresh_metrics(context)
        try:
            allocation = self._orchestrator._propose_assignment(services)
        except RuntimeError as exc:
            # Violation errors from strict validation should signal a failed placement attempt
            # Return empty dict so hysteresis rejects the proposal
            if "violation" in str(exc).lower():
                BlazeLogger.get_instance().warning(
                    f"Orchestrator {self.orchestrator_class.__name__} proposed assignment with violations: {exc}"
                )
                return {}
            else:
                # Other RuntimeErrors are unexpected
                BlazeLogger.get_instance().error(
                    f"Orchestrator {self.orchestrator_class.__name__} failed to propose assignment: {exc}"
                )
                return dict(self._last_allocation)
        except Exception as exc:
            BlazeLogger.get_instance().error(
                f"Orchestrator {self.orchestrator_class.__name__} failed to propose assignment: {exc}"
            )
            return dict(self._last_allocation)

        if allocation:
            self._last_allocation = dict(allocation)
        return allocation

    def _refresh_metrics(self, context: OrchestrationContext) -> None:
        """Refresh metrics in the wrapped orchestrator from the current context."""
        if not context or not self._orchestrator:
            return
        if getattr(context, 'metrics', None) is None:
            return

        self._orchestrator.metrics = context.metrics

        if hasattr(self._orchestrator, 'latency_matrix') and hasattr(context.metrics, 'get_latency_matrix'):
            try:
                try:
                    latency_matrix = context.metrics.get_latency_matrix(context.nodes)
                except TypeError:
                    latency_matrix = context.metrics.get_latency_matrix()
                if latency_matrix is not None:
                    self._orchestrator.latency_matrix = latency_matrix
            except Exception:
                return
    
    def run_single_episode_training(self, all_nodes: Sequence[Any], 
                                   services: Sequence[Any], 
                                   latency: List[List[float]]) -> int:
        """Delegate to wrapped orchestrator."""
        self._ensure_initialized()
        return self._orchestrator.run_single_episode_training(all_nodes, services, latency)


class NSGA2Strategy(OrchestratorStrategyAdapter):
    """Strategy adapter for NSGA2Orchestrator."""
    
    def __init__(self, context: OrchestrationContext, population_size: int = 30, generations: int = 50):
        from .nsga2_orchestrator import NSGA2Orchestrator
        super().__init__(
            NSGA2Orchestrator,
            context,
            population_size=population_size,
            generations=generations
        )


class DQNStrategy(OrchestratorStrategyAdapter):
    """Strategy adapter for DQNOrchestrator."""
    
    def __init__(
        self,
        context: OrchestrationContext,
        learning_rate: float = 0.001,
        episodes: int = 50,
        checkpoint_path: Optional[str] = None,
    ):
        from .dqn_orchestrator import DQNOrchestrator
        super().__init__(
            DQNOrchestrator,
            context,
            checkpoint_path=checkpoint_path,
            lr=learning_rate,  # DQNOrchestrator expects 'lr' not 'learning_rate'
            episodes=episodes
        )


class UnsupervisedStrategy(OrchestratorStrategyAdapter):
    """Strategy adapter for GreedyWeightedOrchestrator (formerly UnsupervisedOrchestrator)."""
    
    def __init__(self, context: OrchestrationContext, checkpoint_path: Optional[str] = None):
        from .unsupervised_orchestrator import GreedyWeightedOrchestrator
        super().__init__(GreedyWeightedOrchestrator, context, checkpoint_path=checkpoint_path)


class AutoEncoderStrategy(OrchestratorStrategyAdapter):
    """Strategy adapter for AutoEncoderOrchestrator."""
    
    def __init__(
        self,
        context: OrchestrationContext,
        latent_dim: int = 16,
        training_epochs: int = 50,
        checkpoint_path: Optional[str] = None,
    ):
        from .autoencoder_orchestrator import AutoEncoderOrchestrator
        super().__init__(
            AutoEncoderOrchestrator,
            context,
            checkpoint_path=checkpoint_path,
            latent_dim=latent_dim,
            epochs=training_epochs  # AutoEncoderOrchestrator expects 'epochs' not 'training_epochs'
        )


class TestStrategy(OrchestratorStrategyAdapter):
    """Strategy adapter for TestOrchestrator."""
    
    def __init__(self, context: OrchestrationContext, service_id: str, 
                 target_node_id: Optional[str] = None, 
                 target_layer_id: Optional[str] = None):
        from .test_orchestrator import TestOrchestrator
        super().__init__(
            TestOrchestrator,
            context,
            service_id=service_id,
            target_node_id=target_node_id,
            target_layer_id=target_layer_id
        )
        self._current_allocation_cache = None
    
    def propose_assignment(self, context: OrchestrationContext, 
                          services: Optional[Sequence[Any]] = None) -> Dict[str, str]:
        """Generate assignment using wrapped orchestrator with current allocation."""
        self._ensure_initialized()
        
        # Update orchestrator's current allocation cache if we have fresh data
        # The context doesn't directly expose current allocation, so we track it ourselves
        result = self._orchestrator._propose_assignment(services)
        
        # Cache this for next iteration
        self._current_allocation_cache = result
        
        return result
    
    def _ensure_initialized(self):
        """Lazy-initialize orchestrator with context data and current allocation."""
        if self._initialized:
            # Update current allocation on each call
            if self._current_allocation_cache is not None:
                self._orchestrator._current_allocation = self._current_allocation_cache
            return
        
        # First initialization
        self._orchestrator = self.orchestrator_class(
            services=self.context.services,
            nodes=self.context.nodes,
            metrics=self.context.metrics,
            current_allocation={},  # Empty on first call
            **self.init_kwargs
        )
        self._initialized = True


def create_strategy(strategy_type: str, context: OrchestrationContext, 
                   config: Dict[str, Any] = None) -> OrchestrationStrategy:
    """
    Factory function to create orchestration strategies.
    
    Args:
        strategy_type: Type of strategy ('nsga2', 'dqn', 'unsupervised', 'autoencoder', 'test')
        context: OrchestrationContext with services, nodes, metrics
        config: Configuration dict for the strategy
    
    Returns:
        OrchestrationStrategy instance
    
    Raises:
        ValueError: If strategy type is unknown
    """
    config = config or {}
    
    if strategy_type == 'nsga2':
        return NSGA2Strategy(
            context,
            population_size=config.get('population_size', 30),
            generations=config.get('generations', 50)
        )
    elif strategy_type == 'dqn':
        return DQNStrategy(
            context,
            learning_rate=config.get('learning_rate', 0.001),
            episodes=config.get('episodes', 50),
            checkpoint_path=config.get('checkpoint_path')
        )
    elif strategy_type == 'unsupervised':
        return UnsupervisedStrategy(
            context,
            checkpoint_path=config.get('checkpoint_path')
        )
    elif strategy_type == 'autoencoder':
        return AutoEncoderStrategy(
            context,
            latent_dim=config.get('latent_dim', 16),
            training_epochs=config.get('training_epochs', 50),
            checkpoint_path=config.get('checkpoint_path')
        )
    elif strategy_type == 'test':
        return TestStrategy(
            context,
            service_id=config.get('service_id'),
            target_node_id=config.get('target_node_id'),
            target_layer_id=config.get('target_layer_id')
        )
    else:
        BlazeLogger.get_instance().warning(
            f"Unknown strategy type '{strategy_type}', defaulting to 'nsga2'"
        )
        return NSGA2Strategy(
            context,
            population_size=config.get('population_size', 30),
            generations=config.get('generations', 50)
        )

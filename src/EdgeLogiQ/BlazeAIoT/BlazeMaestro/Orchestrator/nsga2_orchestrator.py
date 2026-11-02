import random
from typing import List, Tuple, Any, Sequence, Optional, Dict
from .base_nsga2_orchestrator import BaseNSGA2Orchestrator


class NSGA2Orchestrator(BaseNSGA2Orchestrator):
    """NSGA-II orchestration using pure Python implementation with true non-dominated sorting.

    Implements the full NSGA-II algorithm (Deb et al., 2002):
    - Fast non-dominated sorting: groups solutions into Pareto fronts
    - Crowding distance: maintains diversity within fronts
    - Constraint handling: treats constraint violations as an objective
    
    This orchestrator uses genetic algorithm operations (crossover, mutation)
    and inherits common logic from BaseNSGA2Orchestrator.
    """

    def _initialize_population(self) -> List[List[int]]:
        population = []
        attempts = 0
        while len(population) < self.population_size:
            # Randomized near-feasible sampling: each service draws among the nodes allowed by
            # its locks, layers and GPU need; capacity and period are then checked as a whole.
            assignment = [random.choice(self._allowed_nodes(s_idx)) for s_idx in range(self.num_services)]
            attempts += 1
            if self._is_assignment_valid(assignment):
                population.append(assignment)
        return population

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        if self.num_services < 2:
            return parent1[:], parent2[:]
        point = random.randint(1, self.num_services - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        # Enforce locked-node and locked-layer constraints after crossover
        for i in range(self.num_services):
            svc = self.services[i]
            locked_nid = getattr(svc, 'locked_node_id', None)
            locked_layer = getattr(svc, 'locked_layer_id', None)
            
            if locked_nid is not None:
                node_idx = next((j for j, n in enumerate(self.nodes) if str(getattr(n, 'node_id', '')) == str(locked_nid)), 0)
                child1[i] = node_idx
                child2[i] = node_idx
            elif locked_layer is not None:
                # Find first node in the locked layer
                node_idx = next((j for j, n in enumerate(self.nodes) if getattr(n, 'layer_id', None) == locked_layer), 0)
                child1[i] = node_idx
                child2[i] = node_idx
        
        return child1, child2

    def _mutate(self, assignment: List[int], mutation_rate: float = 0.1) -> List[int]:
        for i in range(self.num_services):
            # Skip mutation for locked services
            svc = self.services[i]
            locked_nid = getattr(svc, 'locked_node_id', None)
            locked_layer = getattr(svc, 'locked_layer_id', None)
            
            if locked_nid is not None:
                # Ensure locked node is correctly assigned
                node_idx = next((j for j, n in enumerate(self.nodes) if str(getattr(n, 'node_id', '')) == str(locked_nid)), assignment[i])
                assignment[i] = node_idx
                continue
            elif locked_layer is not None:
                # Ensure node is in locked layer
                current_node = self.nodes[assignment[i]]
                current_layer = getattr(current_node, 'layer_id', None)
                if current_layer != locked_layer:
                    # Find first node in locked layer
                    node_idx = next((j for j, n in enumerate(self.nodes) if getattr(n, 'layer_id', None) == locked_layer), assignment[i])
                    assignment[i] = node_idx
                # Don't mutate further for layer-locked services
                continue
            
            # Only mutate unlocked services, to a node allowed by their layers and GPU need
            if random.random() < mutation_rate:
                assignment[i] = random.choice(self._allowed_nodes(i))
        return assignment

    def _allowed_nodes(self, s_idx: int) -> List[int]:
        """Node indices compatible with the locks, allowed layers and GPU need of a service."""
        cache = getattr(self, '_allowed_cache', None)
        if cache is None or cache[0] is not self.services:
            cache = self._allowed_cache = (self.services, {})
        if s_idx in cache[1]:
            return cache[1][s_idx]
        svc = self.services[s_idx]
        locked_nid = getattr(svc, 'locked_node_id', None)
        locked_layer = getattr(svc, 'locked_layer_id', None)
        layers = getattr(svc, 'layers', []) if getattr(svc, 'multi_layer', False) else []
        gpu = sum(float(getattr(c, 'gpu_gb', 0.0) or 0.0) for c in getattr(svc, 'containers', []))
        allowed = []
        for j, n in enumerate(self.nodes):
            if locked_nid is not None and str(getattr(n, 'node_id', '')) != str(locked_nid):
                continue
            if locked_layer is not None and getattr(n, 'layer_id', None) != locked_layer:
                continue
            if layers and getattr(n, 'layer_id', None) not in layers:
                continue
            if gpu > 0 and (float(n.gpu.capacity) if getattr(n, 'gpu', None) else 0.0) < gpu:
                continue
            allowed.append(j)
        cache[1][s_idx] = allowed or list(range(self.num_nodes))
        return cache[1][s_idx]

    def _generate_offspring(self, population: List[List[int]]) -> List[List[int]]:
        """Generate offspring through crossover and mutation.
        
        Returns:
            List of valid offspring assignments
        """
        offspring = []
        while len(offspring) < self.population_size:
            p1, p2 = random.sample(population, 2)
            c1, c2 = self._crossover(p1, p2)
            c1 = self._mutate(c1)
            c2 = self._mutate(c2)
            if self._is_assignment_valid(c1):
                offspring.append(c1)
            if len(offspring) < self.population_size and self._is_assignment_valid(c2):
                offspring.append(c2)
        return offspring

    def _dominates(self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]) -> bool:
        """Check if objective vector obj1 dominates obj2.
        
        Minimization: obj1 dominates obj2 if obj1 is <= obj2 on all objectives
        and strictly < on at least one objective.
        
        Objectives are: (violations, energy, latency) - all minimize
        """
        # Constrained domination (Deb et al., 2002): a feasible solution (V(x)=0) always
        # dominates an infeasible one; two infeasible solutions compare on violation count;
        # two feasible solutions compare on Pareto dominance over (energy, latency).
        v1, v2 = obj1[0], obj2[0]
        if v1 == 0 and v2 > 0:
            return True
        if v1 > 0 and v2 == 0:
            return False
        if v1 > 0 and v2 > 0:
            return v1 < v2
        at_least_one_better = False
        for i in (1, 2):
            if obj1[i] > obj2[i]:
                return False  # obj1 worse on this objective
            elif obj1[i] < obj2[i]:
                at_least_one_better = True
        return at_least_one_better

    def _fast_nondominated_sort(self, scored_individuals: List[Tuple[Tuple[float, float, float], List[int]]]) -> List[List[Tuple[Tuple[float, float, float], List[int]]]]:
        """Fast non-dominated sorting algorithm (Deb et al., 2000).
        
        Returns list of fronts, where each front is a list of (objectives, assignment) tuples.
        """
        fronts = []
        domination_count = [0] * len(scored_individuals)
        dominated_solutions = [[] for _ in range(len(scored_individuals))]
        
        # Build domination relationships
        for i in range(len(scored_individuals)):
            for j in range(i + 1, len(scored_individuals)):
                obj_i = scored_individuals[i][0]
                obj_j = scored_individuals[j][0]
                
                if self._dominates(obj_i, obj_j):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(obj_j, obj_i):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        # Form first front (non-dominated solutions)
        current_front = [i for i in range(len(scored_individuals)) if domination_count[i] == 0]
        
        # Successively form fronts
        while current_front:
            fronts.append([scored_individuals[i] for i in current_front])
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance_assignment(self, front: List[Tuple[Tuple[float, float, float], List[int]]]) -> List[Tuple[float, Tuple[Tuple[float, float, float], List[int]]]]:
        """Calculate crowding distance for individuals in a front.
        
        Returns list of (crowding_distance, (objectives, assignment)) tuples.
        """
        n = len(front)
        if n == 0:
            return []
        
        # Initialize distances
        distances = [0.0] * n
        
        # Objective dimension indices: 0=violations, 1=energy, 2=latency
        num_objectives = 3
        
        for m in range(num_objectives):
            # Sort by m-th objective
            sorted_indices = sorted(range(n), key=lambda i: front[i][0][m])
            
            # Boundary points get infinite distance (always selected)
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[n-1]] = float('inf')
            
            # Calculate objective range
            obj_max = front[sorted_indices[n-1]][0][m]
            obj_min = front[sorted_indices[0]][0][m]
            obj_range = obj_max - obj_min
            
            # Avoid division by zero
            if obj_range < 1e-10:
                continue
            
            # Assign crowding distance contribution
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                idx_prev = sorted_indices[i - 1]
                idx_next = sorted_indices[i + 1]
                
                distance_contribution = (front[idx_next][0][m] - front[idx_prev][0][m]) / obj_range
                distances[idx] += distance_contribution
        
        # Return with distances (for sorting)
        return [(distances[i], front[i]) for i in range(n)]

    def _select_next_generation(self, population: List[List[int]], offspring: List[List[int]]) -> List[List[int]]:
        """Select next generation using true NSGA-II: non-dominated sorting + crowding distance.
        
        Combines population and offspring, applies fast non-dominated sort,
        calculates crowding distances, and selects top population_size individuals.
        
        Returns:
            Next generation population
        """
        combined = population + offspring
        
        # Score all individuals: (period_violations, energy, latency)
        scored = []
        for assignment in combined:
            energy, latency, violations, is_valid = self._evaluate_assignment(assignment)
            # Include all individuals (valid and invalid) but prioritize violations
            scored.append(((float(violations), energy, latency), assignment))
        
        # Fast non-dominated sort
        fronts = self._fast_nondominated_sort(scored)
        
        # Select individuals from fronts using crowding distance
        next_gen = []
        for front in fronts:
            if len(next_gen) >= self.population_size:
                break
            
            # How many slots available in this front?
            remaining_slots = self.population_size - len(next_gen)
            
            if len(front) <= remaining_slots:
                # Take entire front
                next_gen.extend([assignment for _, assignment in front])
            else:
                # Rank by crowding distance and take top
                ranked = self._crowding_distance_assignment(front)
                ranked.sort(key=lambda x: x[0], reverse=True)  # Higher crowding distance = better
                next_gen.extend([assignment for _, (_, assignment) in ranked[:remaining_slots]])
        
        # Fallback: ensure we have a population
        if not next_gen:
            # If no fronts created, fallback to lexicographic (shouldn't happen)
            scored.sort(key=lambda x: x[0])
            next_gen = [assignment for _, assignment in scored[:self.population_size]]
        
        return next_gen

    def _select_final(self, population: List[List[int]]) -> List[int]:
        scored = [(self._evaluate_assignment(a), a) for a in population]
        best_v = min(v for (_, _, v, _), _ in scored)
        front = [((e, l), a) for (e, l, v, _), a in scored if v == best_v]
        e_min, e_max = min(o[0] for o, _ in front), max(o[0] for o, _ in front)
        l_min, l_max = min(o[1] for o, _ in front), max(o[1] for o, _ in front)

        def dist(o):
            de = (o[0] - e_min) / (e_max - e_min) if e_max > e_min else 0.0
            dl = (o[1] - l_min) / (l_max - l_min) if l_max > l_min else 0.0
            return de * de + dl * dl
        return min(front, key=lambda t: dist(t[0]))[1]

    def _assignment_to_result_dict(self, assignment: List[int]) -> Dict[str, str]:
        """Convert assignment list to service_id -> node_id mapping.
        
        Args:
            assignment: List of node indices for each service
            
        Returns:
            Dict mapping service_id (str) to node_id (str)
        """
        placement = {}
        for i, svc in enumerate(self.services):
            node_idx = int(assignment[i]) if i < len(assignment) else 0
            placement[str(svc.service_id)] = str(self.nodes[node_idx].node_id)
        return placement

    def optimize(self, services: Optional[Sequence[Any]] = None,
                 current_assignment: Optional[Dict[str, str]] = None,
                 hysteresis_threshold: float = 0.0) -> Dict[str, str]:
        """
        Generate a service-to-node assignment using NSGA-II algorithm.
        
        Args:
            services: Optional services list. If provided, uses these instead of self.services.
            current_assignment: Current assignment (unused, for API compatibility)
            hysteresis_threshold: Hysteresis threshold (unused, for API compatibility)
        
        Returns:
            Dict mapping service_id (str) to node_id (str)
        """
        if self.num_nodes == 0:
            return {}
        
        # Temporarily override services if provided
        orig_services = self.services
        if services is not None:
            self.services = list(services)
            self.num_services = len(self.services)
        
        if self.num_services == 0:
            self.services = orig_services
            self.num_services = len(self.services)
            return {}
        
        # Initialize population and evolve
        population = self._initialize_population()
        
        for _ in range(self.generations):
            offspring = self._generate_offspring(population)
            population = self._select_next_generation(population, offspring)
        
        # Final deployment: among the individuals with the fewest violations (the first front
        # under constrained domination), the one closest to the ideal point after min-max
        # normalization of energy and latency.
        best_assignment = self._select_final(population) if population else [0 for _ in range(self.num_services)]
        result = self._assignment_to_result_dict(best_assignment)
        
        # Restore original services
        self.services = orig_services
        self.num_services = len(self.services)
        
        return result

    def run_single_episode_training(self, all_nodes, services, latency) -> int:
        pass
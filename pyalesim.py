#################################
# ALE Simulaton with variable   #
# mutation rate scheduling      #
#                               #
# Geoff Taghon                  #
# NIST-MML / Nov 2024           #
#################################

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from keras.src.optimizers.schedules.learning_rate_schedule import (
    CosineDecay,
    CosineDecayRestarts, 
    ExponentialDecay,
    PiecewiseConstantDecay
)
from tqdm import tqdm
import sys

@dataclass
class CellLine:
    """Represents a distinct cell lineage with its own growth rate"""
    growth_rate: float
    cell_count: int
    mutations: List[Tuple[str, float]]  # List of (mutation_type, effect_size) pairs

class PopulationTracker:
    """Tracks multiple competing subpopulations"""
    def __init__(self):
        self.subpopulations: Dict[float, CellLine] = {}  # growth_rate -> CellLine
        self.total_cells = 0
        
    def add_cells(self, growth_rate: float, count: int, mutation: Optional[Tuple[str, float]] = None):
        """Add cells to a subpopulation"""
        count = int(count)  # Ensure integer count
        if count <= 0:
            return
            
        if growth_rate in self.subpopulations:
            self.subpopulations[growth_rate].cell_count += count
            if mutation:
                self.subpopulations[growth_rate].mutations.append(mutation)
        else:
            mutations = [mutation] if mutation else []
            self.subpopulations[growth_rate] = CellLine(growth_rate, count, mutations)
        self.total_cells += count
    
    def scale_to_size(self, target_size: int):
        """Scale population to target size while maintaining proportions"""
        if self.total_cells == 0:
            return
            
        scale_factor = target_size / self.total_cells
        new_population = PopulationTracker()
        
        for growth_rate, cell_line in self.subpopulations.items():
            new_count = int(cell_line.cell_count * scale_factor)
            if new_count > 0:
                new_population.add_cells(growth_rate, new_count)
        
        self.subpopulations = new_population.subpopulations
        self.total_cells = new_population.total_cells

class ALESimulator:
    def __init__(
            self,
            initial_growth_rate: float = 0.7,  # Initial growth rate in h^-1
            max_growth_rate: float = 2.0,      # Maximum achievable growth rate
            population_size: int = 1000,
            initial_mutation_rate: float = 1e-3,
            deleterious_mutation_rate: float = 1e-8,
            passage_size: float = 0.1,
            time_steps: int = 50,
            scheduler_type: str = 'constant',
            scheduler_kwargs: dict = None,
            name: str = 'Default'  # Added name for plotting
        ):
            self.initial_growth_rate = initial_growth_rate
            self.max_growth_rate = max_growth_rate
            self.population_size = population_size
            self.passage_size = passage_size
            self.time_steps = time_steps
            self.name = name
            
            # Mutation parameters
            self.beneficial_effect_mean = 0.15
            self.beneficial_effect_std = 0.05
            self.deleterious_effect_mean = -0.1
            self.deleterious_effect_std = 0.03
            
            # Initialize mutation rate scheduler
            if scheduler_kwargs is None:
                scheduler_kwargs = {}
                
            self.mutation_scheduler = self._setup_scheduler(
                initial_mutation_rate,
                time_steps,
                scheduler_type,
                scheduler_kwargs
            )
            
            self.deleterious_mutation_rate = deleterious_mutation_rate
            
            # Population tracking
            self.population = PopulationTracker()
            self.population.add_cells(initial_growth_rate, population_size)
            
            # History tracking
            self.mean_fitness_history = []
            self.max_fitness_history = []
            self.mutation_rate_history = []
            self.population_size_history = []

    def _setup_scheduler(self, initial_rate, total_steps, scheduler_type, scheduler_kwargs):
        """Set up mutation rate scheduler"""
        if scheduler_type == 'constant':
            rate = scheduler_kwargs.get('rate', 0.0)
            return lambda x: rate
        elif scheduler_type == 'cosine':
            alpha = scheduler_kwargs.get('alpha', 0.0)
            return CosineDecay(
                initial_learning_rate=initial_rate,
                decay_steps=total_steps,
                alpha=alpha
            )
        elif scheduler_type == 'cosineRestarts':
            first_decay_steps = scheduler_kwargs.get('first_decay_steps', 1000)
            t_mul = scheduler_kwargs.get('t_mul', 2.0)
            m_mul = scheduler_kwargs.get('m_mul', 1.0)
            alpha = scheduler_kwargs.get('alpha', 0.0)
            return CosineDecayRestarts(
                initial_learning_rate=initial_rate,
                first_decay_steps=first_decay_steps,
                t_mul=t_mul,
                m_mul=m_mul,
                alpha=alpha
            )
        elif scheduler_type == 'exponential':
            decay_rate = scheduler_kwargs.get('decay_rate', 0.9)
            return ExponentialDecay(
                initial_learning_rate=initial_rate,
                decay_steps=total_steps // 10,
                decay_rate=decay_rate
            )
        elif scheduler_type == 'stepwise':
            boundaries = scheduler_kwargs.get('boundaries', [total_steps // 2])
            values = scheduler_kwargs.get('values', [initial_rate, initial_rate * 0.1])
            return PiecewiseConstantDecay(boundaries=boundaries, values=values)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def simulate_step(self, current_step: int):
            """Simulate one time step of growth and mutations"""
            # Create new population for this generation
            new_population = PopulationTracker()
            
            # Process each subpopulation's growth
            current_subpops = list(self.population.subpopulations.items())
            
            # First calculate growth for all subpopulations
            for growth_rate, cell_line in current_subpops:
                # Calculate growth
                growth_time = 1.0  # 1 hour
                new_count = int(cell_line.cell_count * np.exp(growth_rate * growth_time))
                
                # Add grown cells to new population
                new_population.add_cells(growth_rate, new_count)
            
            # Now apply mutations to the grown population
            mutated_population = PopulationTracker()
            current_subpops = list(new_population.subpopulations.items())
            
            for growth_rate, cell_line in current_subpops:
                self._apply_mutations(growth_rate, cell_line.cell_count, current_step)
            
            # Scale population back to target size using weighted random selection
            # This creates selection pressure favoring faster-growing cells
            if self.population.total_cells > 0:
                growth_rates = list(self.population.subpopulations.keys())
                weights = [self.population.subpopulations[r].cell_count * np.exp(r) 
                        for r in growth_rates]
                weights = np.array(weights) / sum(weights)
                
                # Select cells for next generation based on their fitness-weighted frequency
                selected_indices = np.random.choice(
                    len(growth_rates),
                    size=self.population_size,
                    p=weights,
                    replace=True
                )
                
                next_population = PopulationTracker()
                for idx in selected_indices:
                    next_population.add_cells(growth_rates[idx], 1)
                
                self.population = next_population
            
            # Record statistics
            if len(self.population.subpopulations) > 0:
                growth_rates = np.array(list(self.population.subpopulations.keys()))
                counts = np.array([cl.cell_count for cl in self.population.subpopulations.values()])
                
                mean_fitness = np.average(growth_rates, weights=counts)
                max_fitness = np.max(growth_rates)
            else:
                mean_fitness = 0
                max_fitness = 0
                
            current_mutation_rate = float(self.mutation_scheduler(current_step))
            
            self.mean_fitness_history.append(mean_fitness)
            self.max_fitness_history.append(max_fitness)
            self.mutation_rate_history.append(current_mutation_rate)
            self.population_size_history.append(self.population.total_cells)

    def _apply_mutations(self, growth_rate: float, cell_count: int, current_step: int):
        """Apply both beneficial and deleterious mutations to a subpopulation"""
        current_beneficial_rate = float(self.mutation_scheduler(current_step))
        
        # Calculate number of mutations
        beneficial_mutations = np.random.binomial(cell_count, current_beneficial_rate)
        deleterious_mutations = np.random.binomial(cell_count, self.deleterious_mutation_rate)
        
        remaining_cells = cell_count - beneficial_mutations - deleterious_mutations
        
        # Process beneficial mutations
        if beneficial_mutations > 0:
            effect = np.random.normal(
                self.beneficial_effect_mean, 
                self.beneficial_effect_std, 
                beneficial_mutations
            )
            new_rates = growth_rate * (1 + effect)
            # Apply growth rate ceiling
            new_rates = np.minimum(new_rates, self.max_growth_rate)
            for rate, count in zip(*np.unique(new_rates, return_counts=True)):
                self.population.add_cells(float(rate), int(count), ('beneficial', float(rate - growth_rate)))
                
        # Process deleterious mutations
        if deleterious_mutations > 0:
            effect = np.random.normal(
                self.deleterious_effect_mean,
                self.deleterious_effect_std,
                deleterious_mutations
            )
            new_rates = growth_rate * (1 + effect)
            new_rates = np.maximum(0.01, new_rates)  # Prevent negative growth rates
            for rate, count in zip(*np.unique(new_rates, return_counts=True)):
                self.population.add_cells(float(rate), int(count), ('deleterious', float(rate - growth_rate)))
        
        # Keep unmutated cells
        if remaining_cells > 0:
            self.population.add_cells(growth_rate, remaining_cells)

    def run(self):
        """Run the complete simulation"""
        try:
            for step in tqdm(range(self.time_steps)):
                self.simulate_step(step)
        except ValueError as e:
            print(f"Simulation stopped at step {step}: {str(e)}")
            
        return (
            self.mean_fitness_history,
            self.max_fitness_history,
            self.mutation_rate_history
        )

def compare_schedulers(
    initial_growth_rate,
    max_growth_rate,
    population_size,
    initial_mutation_rate,
    time_steps
):
    """Compare different mutation rate scheduling strategies"""
    restarts = 3
    schedulers = {
        'Constant': ('constant', {'rate': initial_mutation_rate}),
        #'Cosine': ('cosine', {'alpha': 0.1}),
        'Cosine with Restarts': ('cosineRestarts', {'first_decay_steps': time_steps/restarts, 
                                                    't_mul': 1.0, 
                                                    'm_mul': 10.0, 
                                                    'alpha': 0.1}),
        #'Exponential': ('exponential', {'decay_rate': 0.75}),
        'Stepwise': ('stepwise', {
            'boundaries': [time_steps*1/3,
                           time_steps*2/3],
            'values': [initial_mutation_rate,
                       initial_mutation_rate*10,
                       initial_mutation_rate*100]
        })
    }
    
    results = {}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['b', 'g', 'r', 'm']  # Different color for each scheduler
    linestyles = ['-', '--', ':', '-.']  # Different line style for each scheduler
    
    for (name, (scheduler_type, kwargs)), color, linestyle in zip(schedulers.items(), colors, linestyles):
        print(f"\nRunning simulation with {name} scheduler...")
        simulator = ALESimulator(
            initial_growth_rate=initial_growth_rate,
            max_growth_rate=max_growth_rate,
            population_size=population_size,
            initial_mutation_rate=initial_mutation_rate,
            deleterious_mutation_rate=initial_mutation_rate*0.1,
            time_steps=time_steps,
            scheduler_type=scheduler_type,
            scheduler_kwargs=kwargs,
            name=name
        )
        mean_fitness, max_fitness, mutation_rates = simulator.run()
        
        # Plot fitness curves
        ax1.plot(max_fitness, color=color, linestyle='-', 
                 label=f'{name} Max', linewidth=2)
        ax1.plot(mean_fitness, color=color, linestyle='--',
                 label=f'{name} Mean', linewidth=1)
        
        # Plot mutation rates
        ax2.plot(mutation_rates, color=color, linestyle=linestyle,
                 label=name, linewidth=2)
        
        results[name] = {
            'mean_fitness': mean_fitness,
            'max_fitness': max_fitness,
            'mutation_rates': mutation_rates
        }
    
    # Customize fitness plot
    ax1.set_xlabel('Time Step (hours)')
    ax1.set_ylabel('Growth Rate (h⁻¹)')
    ax1.set_title('Population Fitness Evolution - All Strategies')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True)
    ax1.set_ylim(initial_growth_rate - 0.1, max_growth_rate + 0.1)
    
    # Customize mutation rate plot
    ax2.set_xlabel('Time Step (hours)')
    ax2.set_ylabel('Mutation Rate')
    ax2.set_title('Mutation Rate Schedules')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True)
    ax2.set_yscale('log')
    
    # Add time to convergence annotations
    for name, data in results.items():
        # Define convergence as reaching 95% of max_growth_rate
        convergence_threshold = 0.95 * max_growth_rate
        convergence_time = next((i for i, v in enumerate(data['max_fitness']) 
                               if v >= convergence_threshold), time_steps)
        ax1.axvline(x=convergence_time, color='gray', linestyle=':',
                    alpha=0.3)
        ax1.text(convergence_time, ax1.get_ylim()[0], 
                f'{name}\n{convergence_time}h',
                rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Get arguments from terminal
max_growth_rate = float(sys.argv[1])
population_size = int(sys.argv[2])
time_steps = int(sys.argv[3])

if __name__ == "__main__":
    results = compare_schedulers(
        initial_growth_rate=1.0,
        max_growth_rate=max_growth_rate,
        population_size=population_size, # ~ 10uL of cells at OD 0.1
        initial_mutation_rate=1e-5,
        time_steps=time_steps
    )
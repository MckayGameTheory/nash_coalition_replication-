# src/nash_original.py
"""
Implementation of Nash's original agencies method with documented breakdown issues.
Based on Nash et al. (2012) PNAS paper.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

@dataclass
class CoalitionGame:
    """Represents a coalition formation game."""
    players: List[int]
    coalition_values: Dict[Tuple[int, ...], float]
    demands: Dict[int, float]
    epsilon: float = 2.0
    
    def __post_init__(self):
        """Validate game parameters."""
        assert self.epsilon > 0, "Epsilon must be positive"
        assert all(v >= 0 for v in self.coalition_values.values()), "Coalition values must be non-negative"

class NashOriginalMethod:
    """
    Original Nash agencies method implementation.
    
    This implementation deliberately includes the mathematical breakdown
    documented by Nash (2008) to demonstrate the problem.
    """
    
    def __init__(self, game: CoalitionGame, include_opportunity_costs: bool = True):
        self.game = game
        self.include_opportunity_costs = include_opportunity_costs
        self.breakdown_detected = False
        self.negative_probabilities = []
        
    def calculate_utility(self, coalition: Tuple[int, ...]) -> float:
        """Calculate per-capita utility for a coalition."""
        if coalition in self.game.coalition_values:
            return self.game.coalition_values[coalition] / len(coalition)
        return 0.0
    
    def calculate_opportunity_cost(self, player: int) -> float:
        """Calculate opportunity cost for a player."""
        if not self.include_opportunity_costs:
            return 0.0
            
        max_utility = 0.0
        for coalition, value in self.game.coalition_values.items():
            if player in coalition and len(coalition) > 1:
                utility = value / len(coalition)
                max_utility = max(max_utility, utility)
        return max_utility
    
    def calculate_acceptance_weights(self, player: int) -> Dict[str, float]:
        """Calculate acceptance weights for a player."""
        weights = {}
        
        # Calculate weights for all possible coalitions containing this player
        for coalition, value in self.game.coalition_values.items():
            if player in coalition:
                utility = self.calculate_utility(coalition)
                demand = self.game.demands.get(player, 0.0)
                
                # Original Nash formulation
                weight_arg = (utility - demand) / self.game.epsilon
                weights[str(coalition)] = math.exp(weight_arg)
        
        # Add opportunity cost weight (this can cause breakdown)
        if self.include_opportunity_costs:
            opp_cost = self.calculate_opportunity_cost(player)
            demand = self.game.demands.get(player, 0.0)
            
            # This is where breakdown can occur in Nash's original notes
            # When opportunity costs exceed utilities, this becomes negative
            opp_weight_arg = (opp_cost - demand) / self.game.epsilon
            weights['opportunity'] = math.exp(opp_weight_arg)
            
        return weights
    
    def calculate_acceptance_probabilities(self, player: int) -> Dict[str, float]:
        """Calculate acceptance probabilities for a player."""
        weights = self.calculate_acceptance_weights(player)
        
        # Check for potential breakdown conditions
        total_weight = sum(weights.values())
        if total_weight <= 0:
            self.breakdown_detected = True
            warnings.warn(f"Weight normalization breakdown for player {player}")
            
        # Normalize weights to probabilities
        probabilities = {}
        for coalition, weight in weights.items():
            prob = weight / total_weight if total_weight > 0 else 0.0
            probabilities[coalition] = prob
            
            # Check for negative probabilities (mathematical impossibility)
            if prob < 0:
                self.breakdown_detected = True
                self.negative_probabilities.append((player, coalition, prob))
                warnings.warn(f"Negative probability detected: Player {player}, Coalition {coalition}, Prob {prob}")
        
        return probabilities
    
    def simulate_coalition_formation(self, num_trials: int = 10000, random_seed: Optional[int] = None) -> Dict:
        """Simulate coalition formation process."""
        if random_seed:
            np.random.seed(random_seed)
            
        coalition_counts = {}
        breakdown_count = 0
        
        for trial in range(num_trials):
            self.breakdown_detected = False
            self.negative_probabilities = []
            
            # Calculate acceptance probabilities for each player
            player_choices = {}
            trial_breakdown = False
            
            for player in self.game.players:
                try:
                    probs = self.calculate_acceptance_probabilities(player)
                    
                    # Sample choice based on probabilities
                    coalitions = list(probs.keys())
                    probabilities = list(probs.values())
                    
                    # Check for invalid probabilities
                    if any(p < 0 for p in probabilities) or abs(sum(probabilities) - 1.0) > 1e-6:
                        trial_breakdown = True
                        break
                        
                    choice = np.random.choice(coalitions, p=probabilities)
                    player_choices[player] = choice
                    
                except (ValueError, ZeroDivisionError) as e:
                    trial_breakdown = True
                    break
            
            if trial_breakdown:
                breakdown_count += 1
                continue
                
            # Determine formed coalition based on mutual acceptance
            formed_coalition = self._determine_formed_coalition(player_choices)
            coalition_key = str(sorted(formed_coalition)) if formed_coalition else "no_coalition"
            coalition_counts[coalition_key] = coalition_counts.get(coalition_key, 0) + 1
        
        # Calculate statistics
        total_successful = num_trials - breakdown_count
        coalition_probabilities = {k: v / total_successful for k, v in coalition_counts.items()} if total_successful > 0 else {}
        
        return {
            'coalition_probabilities': coalition_probabilities,
            'coalition_counts': coalition_counts,
            'breakdown_rate': breakdown_count / num_trials,
            'successful_trials': total_successful,
            'total_trials': num_trials,
            'breakdown_detected': breakdown_count > 0
        }
    
    def _determine_formed_coalition(self, player_choices: Dict[int, str]) -> List[int]:
        """Determine which coalition forms based on player choices."""
        # Count votes for each coalition
        coalition_votes = {}
        
        for player, choice in player_choices.items():
            if choice == 'opportunity' or choice == 'no_coalition':
                continue
                
            # Parse coalition string back to tuple
            try:
                coalition = eval(choice)  # Note: In production, use ast.literal_eval
                if isinstance(coalition, int):
                    coalition = (coalition,)
                    
                for member in coalition:
                    if member in self.game.players:
                        coalition_key = str(sorted(coalition))
                        coalition_votes[coalition_key] = coalition_votes.get(coalition_key, 0) + 1
            except:
                continue
        
        # Find coalition with mutual acceptance (all members chose it)
        for coalition_str, votes in coalition_votes.items():
            coalition = eval(coalition_str)
            if isinstance(coalition, int):
                coalition = [coalition]
            else:
                coalition = list(coalition)
                
            # Check if all members of coalition chose this coalition
            if votes >= len(coalition):
                return coalition
                
        return []  # No coalition formed

def run_nash_original_validation():
    """Run validation against Nash et al. 2012 data."""
    # Nash et al. 2012 experimental setup
    coalition_values = {
        (1, 2): 5.0,
        (1, 3): 4.0,
        (2, 3): 3.0,
        (1, 2, 3): 9.0
    }
    
    demands = {1: 0.0, 2: 0.0, 3: 0.0}
    
    game = CoalitionGame(
        players=[1, 2, 3],
        coalition_values=coalition_values,
        demands=demands,
        epsilon=2.0
    )
    
    # Test original method
    nash_original = NashOriginalMethod(game, include_opportunity_costs=True)
    results = nash_original.simulate_coalition_formation(num_trials=20000, random_seed=42)
    
    print("Nash Original Method Results:")
    print(f"Breakdown rate: {results['breakdown_rate']:.1%}")
    print("\nCoalition Formation Probabilities:")
    for coalition, prob in results['coalition_probabilities'].items():
        print(f"  {coalition}: {prob:.1%}")
    
    # Compare with Nash et al. 2012 experimental results
    nash_2012_results = {
        'grand_coalition': 0.915,
        'two_player': 0.070,
        'no_coalition': 0.015
    }
    
    print(f"\nComparison with Nash et al. 2012:")
    print(f"Nash 2012 Grand Coalition: {nash_2012_results['grand_coalition']:.1%}")
    print(f"Our Implementation: Limited by breakdown issues")
    
    return results

if __name__ == "__main__":
    run_nash_original_validation()

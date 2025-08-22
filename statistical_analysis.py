# src/statistical_analysis.py
"""
Comprehensive statistical analysis for Nash coalition formation results.
Includes all statistical tests mentioned in the paper.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency, kstest, bootstrap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

class StatisticalValidator:
    """Comprehensive statistical validation for Nash coalition results."""
    
    def __init__(self):
        self.results = {}
        
    def chi_square_goodness_of_fit(self, observed: List[float], expected: List[float], 
                                 labels: List[str] = None) -> Dict:
        """
        Perform chi-square goodness of fit test.
        
        Args:
            observed: Observed frequencies/probabilities
            expected: Expected frequencies/probabilities  
            labels: Labels for categories
            
        Returns:
            Dictionary with test results
        """
        # Convert probabilities to counts for chi-square test
        total_n = 20000  # Default sample size
        observed_counts = [o * total_n for o in observed]
        expected_counts = [e * total_n for e in expected]
        
        # Perform chi-square test
        chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
        
        # Calculate effect size (Cramer's V)
        n = sum(observed_counts)
        k = len(observed_counts)
        cramers_v = np.sqrt(chi2_stat / (n * (k - 1))) if k > 1 else 0
        
        degrees_of_freedom = len(observed_counts) - 1
        
        result = {
            'test_name': 'Chi-Square Goodness of Fit',
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': degrees_of_freedom,
            'cramers_v': cramers_v,
            'observed_counts': observed_counts,
            'expected_counts': expected_counts,
            'significant': p_value < 0.05,
            'interpretation': self._interpret_chi_square(p_value, cramers_v)
        }
        
        if labels:
            result['category_labels'] = labels
            
        return result
    
    def kolmogorov_smirnov_test(self, data1: np.ndarray, data2: np.ndarray = None, 
                              distribution: str = 'norm') -> Dict:
        """
        Perform Kolmogorov-Smirnov test for distribution comparison.
        
        Args:
            data1: First dataset
            data2: Second dataset (if None, test against theoretical distribution)
            distribution: Theoretical distribution to test against
            
        Returns:
            Dictionary with test results
        """
        if data2 is not None:
            # Two-sample KS test
            ks_stat, p_value = stats.ks_2samp(data1, data2)
            test_type = "Two-sample"
        else:
            # One-sample KS test against theoretical distribution
            if distribution == 'norm':
                ks_stat, p_value = kstest(data1, 'norm')
            elif distribution == 'uniform':
                ks_stat, p_value = kstest(data1, 'uniform')
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")
            test_type = f"One-sample vs {distribution}"
        
        result = {
            'test_name': 'Kolmogorov-Smirnov Test',
            'test_type': test_type,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': self._interpret_ks_test(p_value, ks_stat)
        }
        
        return result
    
    def bootstrap_confidence_intervals(self, data: np.ndarray, statistic: callable = np.mean,
                                     confidence_level: float = 0.95, n_bootstrap: int = 1000) -> Dict:
        """
        Calculate bootstrap confidence intervals.
        
        Args:
            data: Input data
            statistic: Function to compute statistic (default: mean)
            confidence_level: Confidence level (default: 0.95)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with confidence interval results
        """
        # Perform bootstrap
        bootstrap_result = bootstrap((data,), statistic, n_resamples=n_bootstrap, 
                                   confidence_level=confidence_level, random_state=42)
        
        result = {
            'test_name': 'Bootstrap Confidence Interval',
            'statistic_value': statistic(data),
            'confidence_level': confidence_level,
            'confidence_interval': (bootstrap_result.confidence_interval.low, 
                                  bootstrap_result.confidence_interval.high),
            'bootstrap_samples': n_bootstrap,
            'standard_error': np.std(bootstrap_result.bootstrap_distribution)
        }
        
        return result
    
    def cross_validation_analysis(self, X: np.ndarray, y: np.ndarray, 
                                model, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Sklearn-compatible model
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with CV results
        """
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        
        # Fit model for additional metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        
        result = {
            'test_name': 'Cross-Validation Analysis',
            'cv_folds': cv_folds,
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'in_sample_r2': r2_score(y, y_pred),
            'in_sample_rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'interpretation': self._interpret_cv_results(cv_scores)
        }
        
        return result
    
    def nash_2012_comparison(self, our_results: Dict, nash_2012_data: Dict) -> Dict:
        """
        Compare our results with Nash et al. 2012 experimental data.
        
        Args:
            our_results: Our simulation results
            nash_2012_data: Nash et al. 2012 experimental results
            
        Returns:
            Dictionary with comparison results
        """
        # Extract comparable metrics
        our_grand_coalition = our_results['coalition_probabilities'].get('[1, 2, 3]', 0.0)
        nash_grand_coalition = nash_2012_data.get('grand_coalition', 0.915)
        
        our_efficiency = our_results.get('efficiency', 0.0)
        nash_efficiency = nash_2012_data.get('efficiency', 0.90)
        
        # Calculate accuracy metrics
        grand_coalition_accuracy = 1 - abs(our_grand_coalition - nash_grand_coalition) / nash_grand_coalition
        efficiency_accuracy = 1 - abs(our_efficiency - nash_efficiency) / nash_efficiency
        
        # Overall accuracy
        overall_accuracy = (grand_coalition_accuracy + efficiency_accuracy) / 2
        
        # Statistical test for difference
        # Convert to counts for chi-square test
        n = 20000
        our_counts = [our_grand_coalition * n, (1 - our_grand_coalition) * n]
        nash_counts = [nash_grand_coalition * n, (1 - nash_grand_coalition) * n]
        
        chi2_stat, p_value = stats.chisquare(our_counts, nash_counts)
        
        result = {
            'test_name': 'Nash et al. 2012 Comparison',
            'our_grand_coalition': our_grand_coalition,
            'nash_grand_coalition': nash_grand_coalition,
            'grand_coalition_accuracy': grand_coalition_accuracy,
            'our_efficiency': our_efficiency,
            'nash_efficiency': nash_efficiency,
            'efficiency_accuracy': efficiency_accuracy,
            'overall_accuracy': overall_accuracy,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'statistically_different': p_value < 0.05,
            'interpretation': self._interpret_nash_comparison(overall_accuracy, p_value)
        }
        
        return result
    
    def robustness_analysis(self, results_list: List[Dict], parameter_values: List[float],
                          parameter_name: str) -> Dict:
        """
        Analyze robustness across parameter variations.
        
        Args:
            results_list: List of results from different parameter values
            parameter_values: List of parameter values tested
            parameter_name: Name of the parameter being varied
            
        Returns:
            Dictionary with robustness analysis
        """
        # Extract efficiency values
        efficiencies = [r.get('efficiency', 0.0) for r in results_list]
        
        # Calculate stability metrics
        efficiency_mean = np.mean(efficiencies)
        efficiency_std = np.std(efficiencies)
        coefficient_of_variation = efficiency_std / efficiency_mean if efficiency_mean > 0 else np.inf
        
        # Test for correlation with parameter
        correlation, correlation_p_value = stats.pearsonr(parameter_values, efficiencies)
        
        result = {
            'test_name': f'Robustness Analysis - {parameter_name}',
            'parameter_name': parameter_name,
            'parameter_values': parameter_values,
            'efficiencies': efficiencies,
            'efficiency_mean': efficiency_mean,
            'efficiency_std': efficiency_std,
            'coefficient_of_variation': coefficient_of_variation,
            'correlation_with_parameter': correlation,
            'correlation_p_value': correlation_p_value,
            'robust': coefficient_of_variation < 0.1,  # Threshold for robustness
            'interpretation': self._interpret_robustness(coefficient_of_variation, correlation)
        }
        
        return result
    
    def _interpret_chi_square(self, p_value: float, cramers_v: float) -> str:
        """Interpret chi-square test results."""
        if p_value >= 0.05:
            return f"No significant difference detected (p={p_value:.3f}). Model fits data well."
        else:
            effect_size = "small" if cramers_v < 0.1 else "medium" if cramers_v < 0.3 else "large"
            return f"Significant difference detected (p={p_value:.3f}, {effect_size} effect size)."
    
    def _interpret_ks_test(self, p_value: float, ks_stat: float) -> str:
        """Interpret Kolmogorov-Smirnov test results."""
        if p_value >= 0.05:
            return f"Distributions are not significantly different (p={p_value:.3f})."
        else:
            return f"Distributions are significantly different (p={p_value:.3f}, D={ks_stat:.3f})."
    
    def _interpret_cv_results(self, cv_scores: np.ndarray) -> str:
        """Interpret cross-validation results."""
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        if mean_score > 0.8:
            performance = "excellent"
        elif mean_score > 0.6:
            performance = "good"
        elif mean_score > 0.4:
            performance = "moderate"
        else:
            performance = "poor"
            
        stability = "stable" if std_score < 0.1 else "unstable"
        
        return f"Model shows {performance} performance (R²={mean_score:.3f}) with {stability} predictions (std={std_score:.3f})."
    
    def _interpret_nash_comparison(self, accuracy: float, p_value: float) -> str:
        """Interpret comparison with Nash et al. 2012."""
        accuracy_pct = accuracy * 100
        
        if accuracy > 0.9:
            match_quality = "excellent"
        elif accuracy > 0.8:
            match_quality = "good"
        elif accuracy > 0.7:
            match_quality = "moderate"
        else:
            match_quality = "poor"
            
        statistical_diff = "statistically different" if p_value < 0.05 else "statistically similar"
        
        return f"{match_quality.title()} match with Nash et al. 2012 ({accuracy_pct:.1f}% accuracy), {statistical_diff} (p={p_value:.3f})."
    
    def _interpret_robustness(self, cv: float, correlation: float) -> str:
        """Interpret robustness analysis."""
        if cv < 0.05:
            stability = "very stable"
        elif cv < 0.1:
            stability = "stable"
        elif cv < 0.2:
            stability = "moderately stable"
        else:
            stability = "unstable"
            
        if abs(correlation) < 0.3:
            sensitivity = "low sensitivity"
        elif abs(correlation) < 0.6:
            sensitivity = "moderate sensitivity"
        else:
            sensitivity = "high sensitivity"
            
        return f"Results are {stability} (CV={cv:.3f}) with {sensitivity} to parameter changes (r={correlation:.3f})."

def run_comprehensive_statistical_analysis():
    """Run all statistical analyses for the paper."""
    validator = StatisticalValidator()
    
    # Example usage with simulated data
    print("Running Comprehensive Statistical Analysis...")
    print("="*50)
    
    # Simulate some results for demonstration
    np.random.seed(42)
    
    # Example: Chi-square test
    observed = [0.837, 0.054, 0.109]  # Our results
    expected = [0.915, 0.070, 0.015]  # Nash 2012 results
    labels = ['Grand Coalition', 'Two-Player', 'No Coalition']
    
    chi2_result = validator.chi_square_goodness_of_fit(observed, expected, labels)
    print("Chi-Square Goodness of Fit Test:")
    print(f"  χ² = {chi2_result['chi2_statistic']:.3f}, p = {chi2_result['p_value']:.3f}")
    print(f"  {chi2_result['interpretation']}")
    print()
    
    # Example: Bootstrap confidence intervals
    efficiency_data = np.random.normal(0.858, 0.02, 1000)  # Simulated efficiency data
    bootstrap_result = validator.bootstrap_confidence_intervals(efficiency_data)
    print("Bootstrap Confidence Intervals:")
    print(f"  Mean efficiency: {bootstrap_result['statistic_value']:.3f}")
    print(f"  95% CI: [{bootstrap_result['confidence_interval'][0]:.3f}, {bootstrap_result['confidence_interval'][1]:.3f}]")
    print()
    
    return validator

if __name__ == "__main__":
    run_comprehensive_statistical_analysis()

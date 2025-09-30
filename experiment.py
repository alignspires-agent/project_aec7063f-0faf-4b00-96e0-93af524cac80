
import numpy as np
import sys
import logging
from typing import Tuple, List, Optional
from scipy.stats import norm
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LevyProkhorovRobustConformal:
    """
    Implementation of Lévy-Prokhorov robust conformal prediction for time series with distribution shifts.
    Based on the paper: "Conformal Prediction under Lévy–Prokhorov Distribution Shifts"
    """
    
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1, rho: float = 0.05):
        """
        Initialize the robust conformal predictor.
        
        Args:
            alpha: Significance level (1 - alpha is the target coverage)
            epsilon: Local perturbation parameter (Lévy-Prokhorov parameter)
            rho: Global perturbation parameter (Lévy-Prokhorov parameter)
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.rho = rho
        self.calibration_scores = None
        self.quantile_wc = None
        
    def compute_scores(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity scores using absolute error.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Array of nonconformity scores
        """
        try:
            return np.abs(predictions - targets)
        except Exception as e:
            logger.error(f"Error computing scores: {e}")
            sys.exit(1)
    
    def fit(self, calibration_predictions: np.ndarray, calibration_targets: np.ndarray):
        """
        Fit the conformal predictor using calibration data.
        
        Args:
            calibration_predictions: Predictions on calibration data
            calibration_targets: True values for calibration data
        """
        try:
            if len(calibration_predictions) != len(calibration_targets):
                raise ValueError("Predictions and targets must have same length")
                
            self.calibration_scores = self.compute_scores(calibration_predictions, calibration_targets)
            n = len(self.calibration_scores)
            
            # Compute worst-case quantile using Proposition 3.4
            beta_adjusted = 1 - self.alpha + self.rho
            quantile_level = min(beta_adjusted, 1.0)  # Ensure quantile level <= 1
            
            if quantile_level < 0:
                raise ValueError("Invalid quantile level after adjustment")
                
            # Empirical quantile with finite-sample correction
            empirical_quantile = np.quantile(self.calibration_scores, quantile_level, method='linear')
            self.quantile_wc = empirical_quantile + self.epsilon
            
            logger.info(f"Fitted robust conformal predictor: n={n}, alpha={self.alpha}, "
                       f"epsilon={self.epsilon}, rho={self.rho}, worst-case quantile={self.quantile_wc:.4f}")
                       
        except Exception as e:
            logger.error(f"Error in fit method: {e}")
            sys.exit(1)
    
    def predict(self, test_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals for test data.
        
        Args:
            test_predictions: Model predictions for test data
            
        Returns:
            Tuple of (lower_bounds, upper_bounds) for prediction intervals
        """
        try:
            if self.quantile_wc is None:
                raise ValueError("Model not fitted. Call fit() first.")
                
            lower_bounds = test_predictions - self.quantile_wc
            upper_bounds = test_predictions + self.quantile_wc
            
            return lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Error in predict method: {e}")
            sys.exit(1)
    
    def coverage(self, intervals: Tuple[np.ndarray, np.ndarray], test_targets: np.ndarray) -> float:
        """
        Compute empirical coverage of prediction intervals.
        
        Args:
            intervals: Tuple of (lower_bounds, upper_bounds)
            test_targets: True values for test data
            
        Returns:
            Empirical coverage probability
        """
        try:
            lower_bounds, upper_bounds = intervals
            coverage_mask = (test_targets >= lower_bounds) & (test_targets <= upper_bounds)
            return np.mean(coverage_mask)
        except Exception as e:
            logger.error(f"Error computing coverage: {e}")
            sys.exit(1)

class TimeSeriesSimulator:
    """
    Simulate time series data with distribution shifts for testing.
    """
    
    @staticmethod
    def generate_ar1_series(n: int, phi: float = 0.8, sigma: float = 1.0, 
                           shift_point: Optional[int] = None, 
                           shift_magnitude: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate AR(1) time series with optional distribution shift.
        
        Args:
            n: Number of time points
            phi: AR coefficient
            sigma: Noise standard deviation
            shift_point: Point at which distribution shifts (None for no shift)
            shift_magnitude: Magnitude of distribution shift
            
        Returns:
            Tuple of (time_series, time_indices)
        """
        try:
            time_series = np.zeros(n)
            time_series[0] = np.random.normal(0, sigma)
            
            current_sigma = sigma
            for t in range(1, n):
                if shift_point is not None and t >= shift_point:
                    current_sigma = sigma * (1 + shift_magnitude)
                
                time_series[t] = phi * time_series[t-1] + np.random.normal(0, current_sigma)
            
            time_indices = np.arange(n)
            return time_series, time_indices
            
        except Exception as e:
            logger.error(f"Error generating time series: {e}")
            sys.exit(1)
    
    @staticmethod
    def simple_forecast(series: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Simple forecasting method using moving average.
        
        Args:
            series: Time series data
            window: Moving average window
            
        Returns:
            Forecasted values
        """
        try:
            predictions = np.zeros_like(series)
            for i in range(len(series)):
                if i < window:
                    predictions[i] = np.mean(series[:i+1]) if i > 0 else series[0]
                else:
                    predictions[i] = np.mean(series[i-window:i])
            return predictions
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            sys.exit(1)

def run_experiment():
    """
    Main experiment to evaluate Lévy-Prokhorov robust conformal prediction on time series data.
    """
    logger.info("Starting Lévy-Prokhorov Robust Conformal Prediction Experiment")
    
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Experiment parameters
        n_train = 500
        n_calibration = 200
        n_test = 300
        alpha = 0.1
        
        # Generate time series data with distribution shift
        logger.info("Generating time series data with distribution shift...")
        simulator = TimeSeriesSimulator()
        
        # Training data (stable distribution)
        train_series, train_times = simulator.generate_ar1_series(n_train, phi=0.8, sigma=1.0)
        
        # Calibration data (stable distribution)
        cal_series, cal_times = simulator.generate_ar1_series(n_calibration, phi=0.8, sigma=1.0)
        
        # Test data with distribution shift (increased volatility)
        test_series, test_times = simulator.generate_ar1_series(
            n_test, phi=0.8, sigma=1.0, shift_point=n_test//3, shift_magnitude=0.5
        )
        
        # Generate predictions using simple forecasting
        logger.info("Generating forecasts...")
        cal_predictions = simulator.simple_forecast(cal_series)
        test_predictions = simulator.simple_forecast(test_series)
        
        # Test different Lévy-Prokhorov parameter combinations
        epsilon_values = [0.05, 0.1, 0.2]
        rho_values = [0.02, 0.05, 0.1]
        
        results = []
        
        for epsilon in epsilon_values:
            for rho in rho_values:
                logger.info(f"Testing epsilon={epsilon}, rho={rho}")
                
                # Initialize and fit robust conformal predictor
                lprc = LevyProkhorovRobustConformal(alpha=alpha, epsilon=epsilon, rho=rho)
                lprc.fit(cal_predictions, cal_series)
                
                # Generate prediction intervals
                lower_bounds, upper_bounds = lprc.predict(test_predictions)
                
                # Compute coverage and interval width
                coverage = lprc.coverage((lower_bounds, upper_bounds), test_series)
                avg_width = np.mean(upper_bounds - lower_bounds)
                
                results.append({
                    'epsilon': epsilon,
                    'rho': rho,
                    'coverage': coverage,
                    'avg_width': avg_width,
                    'quantile_wc': lprc.quantile_wc
                })
                
                logger.info(f"  Coverage: {coverage:.4f}, Avg Width: {avg_width:.4f}")
        
        # Compare with standard conformal prediction (epsilon=0, rho=0)
        logger.info("Testing standard conformal prediction (epsilon=0, rho=0)")
        standard_cp = LevyProkhorovRobustConformal(alpha=alpha, epsilon=0.0, rho=0.0)
        standard_cp.fit(cal_predictions, cal_series)
        lower_std, upper_std = standard_cp.predict(test_predictions)
        coverage_std = standard_cp.coverage((lower_std, upper_std), test_series)
        avg_width_std = np.mean(upper_std - lower_std)
        
        # Print final results
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS")
        logger.info("="*60)
        
        logger.info(f"Standard Conformal Prediction:")
        logger.info(f"  Coverage: {coverage_std:.4f} (Target: {1-alpha:.3f})")
        logger.info(f"  Average Interval Width: {avg_width_std:.4f}")
        
        logger.info("\nLévy-Prokhorov Robust Conformal Prediction:")
        for result in results:
            logger.info(f"  epsilon={result['epsilon']}, rho={result['rho']}:")
            logger.info(f"    Coverage: {result['coverage']:.4f}")
            logger.info(f"    Avg Width: {result['avg_width']:.4f}")
            logger.info(f"    Worst-case Quantile: {result['quantile_wc']:.4f}")
        
        # Find best parameter combination
        valid_results = [r for r in results if r['coverage'] >= (1 - alpha - 0.05)]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['avg_width'])
            logger.info(f"\nBest robust parameters: epsilon={best_result['epsilon']}, rho={best_result['rho']}")
            logger.info(f"Best coverage: {best_result['coverage']:.4f}")
            logger.info(f"Best average width: {best_result['avg_width']:.4f}")
        else:
            logger.warning("No parameter combination achieved target coverage")
            
        # Performance metrics
        improvement_coverage = max(r['coverage'] for r in results) - coverage_std
        improvement_robustness = any(r['coverage'] >= 0.85 for r in results)  # Check if robust method maintains reasonable coverage
        
        logger.info(f"\nPerformance Summary:")
        logger.info(f"Coverage improvement over standard CP: {improvement_coverage:.4f}")
        logger.info(f"Robustness achieved: {improvement_robustness}")
        
        return {
            'standard_coverage': coverage_std,
            'standard_width': avg_width_std,
            'robust_results': results,
            'best_robust_params': best_result if valid_results else None
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run the experiment
    final_results = run_experiment()
    
    logger.info("Experiment completed successfully!")

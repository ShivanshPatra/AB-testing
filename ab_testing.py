"""
Production-Grade A/B Testing Framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
import json
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Experiment status enumeration"""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    INVALID = "invalid"

class DecisionRule(Enum):
    """Statistical decision framework"""
    FIXED_HORIZON = "fixed_horizon"
    SEQUENTIAL = "sequential"
    ALWAYS_VALID = "always_valid"

@dataclass
class ExperimentConfig:
    """Experiment configuration with all parameters"""
    experiment_id: str
    experiment_name: str
    hypothesis: str
    variants: List[str]
    control_variant: str
    primary_metric: str
    guardrail_metrics: List[str]
    
    # Statistical parameters
    alpha: float = 0.05  # Significance level
    power: float = 0.80  # Statistical power
    mde: float = 0.02  # Minimum detectable effect (relative)
    baseline_rate: float = 0.08  # Expected baseline conversion rate
    
    # Sequential testing parameters
    decision_rule: DecisionRule = DecisionRule.SEQUENTIAL
    alpha_spending_function: str = "obrien_fleming"  # or "pocock"
    max_looks: int = 5  # Maximum number of interim analyses
    
    # Sample size and duration
    min_sample_size_per_variant: int = 1000
    max_duration_days: int = 30
    
    # Data quality thresholds
    max_srm_p_value: float = 0.01  # Sample Ratio Mismatch threshold
    min_data_quality_score: float = 0.95
    
    # Business parameters
    practical_significance_threshold: float = 0.01  # Minimum practical effect
    cost_per_user: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    owner: str = "data_science_team"
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate experiment configuration"""
        errors = []
        
        if self.control_variant not in self.variants:
            errors.append(f"Control variant '{self.control_variant}' not in variants list")
        
        if not 0 < self.alpha < 1:
            errors.append(f"Alpha must be between 0 and 1, got {self.alpha}")
        
        if not 0 < self.power < 1:
            errors.append(f"Power must be between 0 and 1, got {self.power}")
        
        if self.mde <= 0:
            errors.append(f"MDE must be positive, got {self.mde}")
        
        if self.baseline_rate <= 0 or self.baseline_rate >= 1:
            errors.append(f"Baseline rate must be between 0 and 1, got {self.baseline_rate}")
        
        if len(self.variants) < 2:
            errors.append("Need at least 2 variants (control + treatment)")
        
        if self.min_sample_size_per_variant < 100:
            errors.append("Minimum sample size per variant should be at least 100")
        
        return len(errors) == 0, errors


@dataclass
class MetricResult:
    """Results for a single metric"""
    metric_name: str
    control_value: float
    treatment_value: float
    control_count: int
    treatment_count: int
    absolute_diff: float
    relative_diff: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    is_guardrail: bool
    test_statistic: float


@dataclass
class ExperimentResults:
    """Complete experiment results"""
    experiment_id: str
    status: TestStatus
    metric_results: Dict[str, MetricResult]
    overall_decision: str
    sample_ratio_mismatch_detected: bool
    srm_p_value: float
    data_quality_score: float
    warnings: List[str]
    recommendations: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]


class DataValidator:
    """Validates experiment data quality"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, config: ExperimentConfig) -> Tuple[bool, List[str], float]:
        """
        Comprehensive data validation
        
        Returns:
            Tuple of (is_valid, error_messages, quality_score)
        """
        errors = []
        quality_checks = []
        
        # Check required columns
        required_cols = ['variant', config.primary_metric, 'user_id', 'timestamp']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
                quality_checks.append(0)
            else:
                quality_checks.append(1)
        
        if errors:
            return False, errors, 0.0
        
        # Check for nulls
        null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if null_pct > 0.01:
            errors.append(f"High null percentage: {null_pct:.2%}")
            quality_checks.append(0)
        else:
            quality_checks.append(1)
        
        # Check variant values
        invalid_variants = set(df['variant'].unique()) - set(config.variants)
        if invalid_variants:
            errors.append(f"Invalid variants found: {invalid_variants}")
            quality_checks.append(0)
        else:
            quality_checks.append(1)
        
        # Check metric bounds (conversion should be 0 or 1)
        if config.primary_metric in df.columns:
            unique_vals = df[config.primary_metric].unique()
            if not set(unique_vals).issubset({0, 1}):
                errors.append(f"Primary metric should be binary (0 or 1), found: {unique_vals}")
                quality_checks.append(0)
            else:
                quality_checks.append(1)
        
        # Check for duplicate users
        duplicate_pct = df['user_id'].duplicated().sum() / len(df)
        if duplicate_pct > 0.1:
            warnings.warn(f"High duplicate user percentage: {duplicate_pct:.2%}")
        quality_checks.append(1 - min(duplicate_pct, 1))
        
        # Check timestamp validity
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            quality_checks.append(1)
        except Exception as e:
            errors.append(f"Invalid timestamp format: {e}")
            quality_checks.append(0)
        
        # Check for extreme outliers in conversion rate by variant
        for variant in config.variants:
            variant_data = df[df['variant'] == variant]
            if len(variant_data) > 0:
                conv_rate = variant_data[config.primary_metric].mean()
                if conv_rate < 0.001 or conv_rate > 0.99:
                    errors.append(f"Suspicious conversion rate for {variant}: {conv_rate:.4f}")
                    quality_checks.append(0)
                else:
                    quality_checks.append(1)
        
        quality_score = np.mean(quality_checks)
        is_valid = len(errors) == 0 and quality_score >= config.min_data_quality_score
        
        return is_valid, errors, quality_score


class SampleSizeCalculator:
    """Calculate required sample sizes for experiments"""
    
    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.80,
        n_variants: int = 2
    ) -> Dict[str, Any]:
        """
        Calculate required sample size per variant
        
        Uses normal approximation for proportion tests
        """
        # Effect size
        treatment_rate = baseline_rate * (1 + mde)
        
        # Bonferroni correction for multiple variants
        alpha_corrected = alpha / (n_variants - 1) if n_variants > 2 else alpha
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha_corrected / 2)
        z_beta = stats.norm.ppf(power)
        
        # Pooled proportion under H0
        p_pooled = (baseline_rate + treatment_rate) / 2
        
        # Sample size formula for two proportions
        n_per_variant = (
            (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
             z_beta * np.sqrt(baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate))
            ) ** 2
        ) / (baseline_rate - treatment_rate) ** 2
        
        n_per_variant = int(np.ceil(n_per_variant))
        total_sample_size = n_per_variant * n_variants
        
        # Calculate expected duration (assuming 1000 users per day as default)
        estimated_days = np.ceil(total_sample_size / 1000)
        
        return {
            'n_per_variant': n_per_variant,
            'total_sample_size': total_sample_size,
            'estimated_days': estimated_days,
            'baseline_rate': baseline_rate,
            'treatment_rate': treatment_rate,
            'mde': mde,
            'alpha': alpha,
            'power': power,
            'alpha_corrected': alpha_corrected
        }


class SampleRatioMismatchDetector:
    """Detect sample ratio mismatch (SRM) - indicates randomization issues"""
    
    @staticmethod
    def check_srm(df: pd.DataFrame, config: ExperimentConfig) -> Tuple[bool, float, Dict[str, int]]:
        """
        Check for sample ratio mismatch using chi-square test
        
        Returns:
            Tuple of (srm_detected, p_value, observed_counts)
        """
        observed_counts = df['variant'].value_counts().to_dict()
        
        # Expected equal allocation
        total = len(df)
        expected = total / len(config.variants)
        
        # Chi-square test
        observed_array = np.array([observed_counts.get(v, 0) for v in config.variants])
        expected_array = np.array([expected] * len(config.variants))
        
        chi2_stat = np.sum((observed_array - expected_array) ** 2 / expected_array)
        df_chi = len(config.variants) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df_chi)
        
        # SRM detected if p-value is less than threshold (unusual imbalance)
        srm_detected = p_value < config.max_srm_p_value
        
        if srm_detected:
            logger.warning(f"Sample Ratio Mismatch detected! p-value: {p_value:.6f}")
            logger.warning(f"Observed counts: {observed_counts}")
        
        return srm_detected, p_value, observed_counts


class SequentialTestingEngine:
    """Sequential testing with alpha spending functions"""
    
    @staticmethod
    def obrien_fleming_boundary(k: int, K: int, alpha: float = 0.05) -> float:
        """
        O'Brien-Fleming alpha spending function
        
        Args:
            k: Current look number (1-indexed)
            K: Total number of planned looks
            alpha: Overall significance level
        """
        # O'Brien-Fleming spending function
        t = k / K
        spent_alpha = 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t)))
        return spent_alpha
    
    @staticmethod
    def pocock_boundary(k: int, K: int, alpha: float = 0.05) -> float:
        """
        Pocock alpha spending function (constant boundaries)
        
        Args:
            k: Current look number (1-indexed)
            K: Total number of planned looks
            alpha: Overall significance level
        """
        # Approximate Pocock boundary
        c = 2.413  # Pocock constant for alpha=0.05, adjust as needed
        return alpha * k / K
    
    @staticmethod
    def get_adjusted_alpha(
        look_number: int,
        total_looks: int,
        alpha: float,
        spending_function: str = "obrien_fleming",
        cumulative_alpha: float = 0.0
    ) -> Tuple[float, float]:
        """
        Get alpha to use at current look
        
        Returns:
            Tuple of (alpha_current_look, cumulative_alpha_spent)
        """
        if spending_function == "obrien_fleming":
            total_spent = SequentialTestingEngine.obrien_fleming_boundary(
                look_number, total_looks, alpha
            )
        elif spending_function == "pocock":
            total_spent = SequentialTestingEngine.pocock_boundary(
                look_number, total_looks, alpha
            )
        else:
            raise ValueError(f"Unknown spending function: {spending_function}")
        
        # Alpha for this specific look
        alpha_current = total_spent - cumulative_alpha
        
        return alpha_current, total_spent


class StatisticalTestEngine:
    """Core statistical testing engine"""
    
    @staticmethod
    def _proportions_ztest(count1, nobs1, count2, nobs2, alternative='two-sided'):
        """
        Two-sample proportions z-test (replacement for statsmodels)
        
        Returns: z_statistic, p_value
        """
        p1 = count1 / nobs1
        p2 = count2 / nobs2
        
        # Pooled proportion under null hypothesis
        p_pooled = (count1 + count2) / (nobs1 + nobs2)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/nobs1 + 1/nobs2))
        
        # Z statistic
        z = (p1 - p2) / se if se > 0 else 0
        
        # P-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        elif alternative == 'larger':
            p_value = 1 - stats.norm.cdf(z)
        elif alternative == 'smaller':
            p_value = stats.norm.cdf(z)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        return z, p_value
    
    @staticmethod
    def _wilson_score_interval(successes, trials, alpha=0.05):
        """
        Wilson score confidence interval for proportion
        More accurate than normal approximation, especially for small p or n
        """
        if trials == 0:
            return (0, 0)
        
        p = successes / trials
        z = stats.norm.ppf(1 - alpha/2)
        
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        offset = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
        
        return (centre - offset, centre + offset)
    
    @staticmethod
    def proportion_test(
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """
        Two-sample proportion z-test with confidence intervals
        
        Args:
            alternative: 'two-sided', 'larger' (treatment > control), 'smaller'
        """
        # Conversion rates
        p_control = control_conversions / control_total
        p_treatment = treatment_conversions / treatment_total
        
        # Z-test
        z_stat, p_value = StatisticalTestEngine._proportions_ztest(
            treatment_conversions, treatment_total,
            control_conversions, control_total,
            alternative=alternative
        )
        
        # Confidence intervals (Wilson score method)
        ci_control = StatisticalTestEngine._wilson_score_interval(
            control_conversions, control_total, alpha
        )
        ci_treatment = StatisticalTestEngine._wilson_score_interval(
            treatment_conversions, treatment_total, alpha
        )
        
        # Difference CI (using normal approximation)
        diff = p_treatment - p_control
        se_diff = np.sqrt(
            p_control * (1 - p_control) / control_total + 
            p_treatment * (1 - p_treatment) / treatment_total
        )
        z_critical = stats.norm.ppf(1 - alpha / 2)
        ci_diff = (diff - z_critical * se_diff, diff + z_critical * se_diff)
        
        # Relative lift
        relative_lift = (p_treatment - p_control) / p_control if p_control > 0 else np.inf
        
        return {
            'control_rate': p_control,
            'treatment_rate': p_treatment,
            'absolute_diff': diff,
            'relative_lift': relative_lift,
            'z_statistic': z_stat,
            'p_value': p_value,
            'ci_control': ci_control,
            'ci_treatment': ci_treatment,
            'ci_diff': ci_diff,
            'is_significant': p_value < alpha
        }


class ExperimentAnalyzer:
    """Main experiment analysis orchestrator"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.validator = DataValidator()
        self.srm_detector = SampleRatioMismatchDetector()
        self.stat_engine = StatisticalTestEngine()
        self.sequential_engine = SequentialTestingEngine()
        
        # Validate config
        is_valid, errors = config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {errors}")
        
        logger.info(f"Initialized experiment: {config.experiment_name} (ID: {config.experiment_id})")
    
    def run_power_analysis(self) -> Dict[str, Any]:
        """Run power analysis to determine required sample size"""
        logger.info("Running power analysis...")
        
        calculator = SampleSizeCalculator()
        sample_size_info = calculator.calculate_sample_size(
            baseline_rate=self.config.baseline_rate,
            mde=self.config.mde,
            alpha=self.config.alpha,
            power=self.config.power,
            n_variants=len(self.config.variants)
        )
        
        logger.info(f"Required sample size per variant: {sample_size_info['n_per_variant']:,}")
        logger.info(f"Estimated duration: {sample_size_info['estimated_days']} days")
        
        return sample_size_info
    
    def analyze(
        self,
        data: pd.DataFrame,
        look_number: int = 1,
        cumulative_alpha_spent: float = 0.0
    ) -> ExperimentResults:
        """
        Run complete experiment analysis
        
        Args:
            data: Experiment data
            look_number: Current analysis number (for sequential testing)
            cumulative_alpha_spent: Alpha already spent in previous looks
        """
        logger.info(f"Starting analysis (Look #{look_number})...")
        
        warnings_list = []
        recommendations = []
        
        # Step 1: Data Validation
        is_valid, validation_errors, quality_score = self.validator.validate_dataframe(
            data, self.config
        )
        
        if not is_valid:
            logger.error(f"Data validation failed: {validation_errors}")
            return ExperimentResults(
                experiment_id=self.config.experiment_id,
                status=TestStatus.INVALID,
                metric_results={},
                overall_decision="INVALID - Data quality issues",
                sample_ratio_mismatch_detected=False,
                srm_p_value=1.0,
                data_quality_score=quality_score,
                warnings=validation_errors,
                recommendations=["Fix data quality issues before analysis"],
                timestamp=datetime.now(),
                metadata={}
            )
        
        logger.info(f"Data quality score: {quality_score:.2%}")
        
        # Step 2: Sample Ratio Mismatch Check
        srm_detected, srm_p_value, variant_counts = self.srm_detector.check_srm(
            data, self.config
        )
        
        if srm_detected:
            warnings_list.append(
                f"Sample Ratio Mismatch detected (p={srm_p_value:.6f}). "
                f"Randomization may be compromised."
            )
            recommendations.append(
                "Investigate randomization logic and data collection pipeline"
            )
        
        # Step 3: Adjust alpha for sequential testing
        if self.config.decision_rule == DecisionRule.SEQUENTIAL:
            alpha_current, cumulative_alpha = self.sequential_engine.get_adjusted_alpha(
                look_number=look_number,
                total_looks=self.config.max_looks,
                alpha=self.config.alpha,
                spending_function=self.config.alpha_spending_function,
                cumulative_alpha=cumulative_alpha_spent
            )
            logger.info(f"Sequential testing: Using alpha={alpha_current:.6f} for look #{look_number}")
        else:
            alpha_current = self.config.alpha
            cumulative_alpha = self.config.alpha
        
        # Step 4: Sample Size Check
        actual_size = len(data) / len(self.config.variants)
        if actual_size < self.config.min_sample_size_per_variant:
            warnings_list.append(
                f"Sample size ({actual_size:.0f}) below minimum "
                f"({self.config.min_sample_size_per_variant})"
            )
            recommendations.append("Continue collecting data before making final decision")
        
        # Step 5: Analyze Primary Metric
        metric_results = {}
        
        # Get control and treatment data
        control_data = data[data['variant'] == self.config.control_variant]
        
        for variant in self.config.variants:
            if variant == self.config.control_variant:
                continue
            
            treatment_data = data[data['variant'] == variant]
            
            # Primary metric analysis
            result = self._analyze_metric(
                control_data=control_data,
                treatment_data=treatment_data,
                metric_name=self.config.primary_metric,
                alpha=alpha_current,
                is_guardrail=False
            )
            
            metric_results[f"{self.config.primary_metric}_{variant}_vs_{self.config.control_variant}"] = result
            
            # Guardrail metrics analysis
            for guardrail_metric in self.config.guardrail_metrics:
                if guardrail_metric in data.columns:
                    guardrail_result = self._analyze_metric(
                        control_data=control_data,
                        treatment_data=treatment_data,
                        metric_name=guardrail_metric,
                        alpha=alpha_current,
                        is_guardrail=True
                    )
                    metric_results[f"{guardrail_metric}_{variant}_vs_{self.config.control_variant}"] = guardrail_result
        
        # Step 6: Make Decision
        decision, decision_recommendations = self._make_decision(
            metric_results, 
            variant_counts,
            srm_detected
        )
        
        recommendations.extend(decision_recommendations)
        
        # Step 7: Compile Results
        results = ExperimentResults(
            experiment_id=self.config.experiment_id,
            status=TestStatus.RUNNING if look_number < self.config.max_looks else TestStatus.COMPLETED,
            metric_results=metric_results,
            overall_decision=decision,
            sample_ratio_mismatch_detected=srm_detected,
            srm_p_value=srm_p_value,
            data_quality_score=quality_score,
            warnings=warnings_list,
            recommendations=recommendations,
            timestamp=datetime.now(),
            metadata={
                'look_number': look_number,
                'max_looks': self.config.max_looks,
                'alpha_spent': cumulative_alpha,
                'variant_counts': variant_counts,
                'decision_rule': self.config.decision_rule.value,
                'config': self.config.__dict__
            }
        )
        
        return results
    
    def _analyze_metric(
        self,
        control_data: pd.DataFrame,
        treatment_data: pd.DataFrame,
        metric_name: str,
        alpha: float,
        is_guardrail: bool
    ) -> MetricResult:
        """Analyze a single metric"""
        
        control_conversions = control_data[metric_name].sum()
        control_total = len(control_data)
        treatment_conversions = treatment_data[metric_name].sum()
        treatment_total = len(treatment_data)
        
        # Run statistical test
        test_result = self.stat_engine.proportion_test(
            control_conversions=int(control_conversions),
            control_total=control_total,
            treatment_conversions=int(treatment_conversions),
            treatment_total=treatment_total,
            alpha=alpha,
            alternative='two-sided'
        )
        
        return MetricResult(
            metric_name=metric_name,
            control_value=test_result['control_rate'],
            treatment_value=test_result['treatment_rate'],
            control_count=control_total,
            treatment_count=treatment_total,
            absolute_diff=test_result['absolute_diff'],
            relative_diff=test_result['relative_lift'],
            p_value=test_result['p_value'],
            ci_lower=test_result['ci_diff'][0],
            ci_upper=test_result['ci_diff'][1],
            is_significant=test_result['is_significant'],
            is_guardrail=is_guardrail,
            test_statistic=test_result['z_statistic']
        )
    
    def _make_decision(
        self,
        metric_results: Dict[str, MetricResult],
        variant_counts: Dict[str, int],
        srm_detected: bool
    ) -> Tuple[str, List[str]]:
        """
        Make final decision based on all metrics
        
        Returns:
            Tuple of (decision_string, recommendations_list)
        """
        recommendations = []
        
        # Check if SRM invalidates results
        if srm_detected:
            return (
                "INCONCLUSIVE - Sample Ratio Mismatch detected",
                ["Do not ship. Investigate and fix randomization issues."]
            )
        
        # Get primary metric results
        primary_results = {
            k: v for k, v in metric_results.items() 
            if v.metric_name == self.config.primary_metric and not v.is_guardrail
        }
        
        # Get guardrail results
        guardrail_results = {k: v for k, v in metric_results.items() if v.is_guardrail}
        
        # Check guardrails first
        guardrail_violations = [
            k for k, v in guardrail_results.items() 
            if v.is_significant and v.absolute_diff < 0  # Negative impact on guardrail
        ]
        
        if guardrail_violations:
            return (
                f"DO NOT SHIP - Guardrail violations detected: {guardrail_violations}",
                ["Treatment negatively impacts guardrail metrics. Do not proceed."]
            )
        
        # Analyze primary metric
        decisions = []
        for variant_key, result in primary_results.items():
            # Statistical significance
            is_stat_sig = result.is_significant
            
            # Practical significance
            is_practical = abs(result.relative_diff) >= self.config.practical_significance_threshold
            
            # Direction
            is_positive = result.absolute_diff > 0
            
            if is_stat_sig and is_practical and is_positive:
                decisions.append("SHIP")
                recommendations.append(
                    f"{variant_key}: Statistically significant improvement "
                    f"({result.relative_diff:+.2%}). Recommend shipping."
                )
            elif is_stat_sig and is_positive:
                decisions.append("NEUTRAL")
                recommendations.append(
                    f"{variant_key}: Statistically significant but below practical "
                    f"threshold ({result.relative_diff:+.2%} vs {self.config.practical_significance_threshold:.2%}). "
                    f"Consider business factors."
                )
            elif is_stat_sig and not is_positive:
                decisions.append("DO_NOT_SHIP")
                recommendations.append(
                    f"{variant_key}: Statistically significant negative effect "
                    f"({result.relative_diff:+.2%}). Do not ship."
                )
            else:
                decisions.append("INCONCLUSIVE")
                recommendations.append(
                    f"{variant_key}: No significant effect detected "
                    f"(p={result.p_value:.4f}). May need more data."
                )
        
        # Final decision
        if all(d == "SHIP" for d in decisions):
            final_decision = "SHIP - All variants show positive results"
        elif any(d == "DO_NOT_SHIP" for d in decisions):
            final_decision = "DO NOT SHIP - Negative effects detected"
        elif all(d == "INCONCLUSIVE" for d in decisions):
            final_decision = "INCONCLUSIVE - Insufficient evidence"
        else:
            final_decision = "MIXED RESULTS - Review detailed metrics"
        
        return final_decision, recommendations
    
    def generate_report(self, results: ExperimentResults) -> str:
        """Generate human-readable experiment report"""
        
        report = []
        report.append("=" * 80)
        report.append(f"EXPERIMENT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Experiment ID: {results.experiment_id}")
        report.append(f"Status: {results.status.value.upper()}")
        report.append(f"Timestamp: {results.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Decision: {results.overall_decision}")
        report.append("=" * 80)
        
        # Data Quality
        report.append(f"\nDATA QUALITY")
        report.append("-" * 80)
        report.append(f"Quality Score: {results.data_quality_score:.2%}")
        report.append(f"Sample Ratio Mismatch: {'DETECTED' if results.sample_ratio_mismatch_detected else 'OK'}")
        report.append(f"SRM p-value: {results.srm_p_value:.6f}")
        
        if results.metadata.get('variant_counts'):
            report.append(f"\nSample Sizes:")
            for variant, count in results.metadata['variant_counts'].items():
                report.append(f"  {variant}: {count:,}")
        
        # Warnings
        if results.warnings:
            report.append(f"\nWARNINGS")
            report.append("-" * 80)
            for warning in results.warnings:
                report.append(f"  ⚠ {warning}")
        
        # Primary Metrics
        report.append(f"\nPRIMARY METRIC RESULTS")
        report.append("-" * 80)
        
        primary_metrics = {
            k: v for k, v in results.metric_results.items() 
            if not v.is_guardrail
        }
        
        for metric_key, result in primary_metrics.items():
            report.append(f"\n{metric_key}:")
            report.append(f"  Control Rate:    {result.control_value:.4f} ({result.control_value*100:.2f}%)")
            report.append(f"  Treatment Rate:  {result.treatment_value:.4f} ({result.treatment_value*100:.2f}%)")
            report.append(f"  Absolute Diff:   {result.absolute_diff:+.4f} ({result.absolute_diff*100:+.2f}pp)")
            report.append(f"  Relative Lift:   {result.relative_diff:+.2%}")
            report.append(f"  95% CI:          [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
            report.append(f"  p-value:         {result.p_value:.6f}")
            report.append(f"  Significant:     {'YES ✓' if result.is_significant else 'NO ✗'}")
        
        # Guardrail Metrics
        guardrail_metrics = {k: v for k, v in results.metric_results.items() if v.is_guardrail}
        
        if guardrail_metrics:
            report.append(f"\nGUARDRAIL METRICS")
            report.append("-" * 80)
            
            for metric_key, result in guardrail_metrics.items():
                status = "VIOLATED ✗" if (result.is_significant and result.absolute_diff < 0) else "OK ✓"
                report.append(f"\n{metric_key}: {status}")
                report.append(f"  Control:   {result.control_value:.4f}")
                report.append(f"  Treatment: {result.treatment_value:.4f}")
                report.append(f"  Diff:      {result.absolute_diff:+.4f} (p={result.p_value:.4f})")
        
        # Recommendations
        report.append(f"\nRECOMMENDATIONS")
        report.append("-" * 80)
        for i, rec in enumerate(results.recommendations, 1):
            report.append(f"{i}. {rec}")
        
        # Metadata
        report.append(f"\nMETADATA")
        report.append("-" * 80)
        report.append(f"Look Number: {results.metadata.get('look_number', 'N/A')} / {results.metadata.get('max_looks', 'N/A')}")
        alpha_spent = results.metadata.get('alpha_spent', 'N/A')
        if isinstance(alpha_spent, (int, float)):
            report.append(f"Alpha Spent: {alpha_spent:.6f}")
        else:
            report.append(f"Alpha Spent: {alpha_spent}")
        report.append(f"Decision Rule: {results.metadata.get('decision_rule', 'N/A')}")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Example usage of the production A/B testing framework"""
    
    # ========================================================================
    # STEP 1: Define Experiment Configuration
    # ========================================================================
    
    config = ExperimentConfig(
        experiment_id="EXP-2026-001",
        experiment_name="Checkout Flow Redesign",
        hypothesis="New checkout flow will increase conversion rate by at least 2%",
        variants=["control", "treatment"],
        control_variant="control",
        primary_metric="conversion",
        guardrail_metrics=["cart_abandonment", "page_load_time_acceptable"],
        
        # Statistical parameters
        alpha=0.05,
        power=0.80,
        mde=0.02,  # 2% relative improvement
        baseline_rate=0.08,
        
        # Sequential testing
        decision_rule=DecisionRule.SEQUENTIAL,
        alpha_spending_function="obrien_fleming",
        max_looks=5,
        
        # Sample size
        min_sample_size_per_variant=1000,
        max_duration_days=14,
        
        # Business parameters
        practical_significance_threshold=0.01,  # 1% minimum practical effect
        
        owner="growth_team"
    )
    
    # ========================================================================
    # STEP 2: Initialize Analyzer & Run Power Analysis
    # ========================================================================
    
    analyzer = ExperimentAnalyzer(config)
    
    # Run power analysis
    sample_size_info = analyzer.run_power_analysis()
    print(f"\n{'='*80}")
    print(f"POWER ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"Required sample size per variant: {sample_size_info['n_per_variant']:,}")
    print(f"Total required sample size: {sample_size_info['total_sample_size']:,}")
    print(f"Estimated duration: {sample_size_info['estimated_days']} days")
    print(f"Expected treatment rate: {sample_size_info['treatment_rate']:.4f}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # STEP 3: Simulate Experiment Data (In production, this comes from your system)
    # ========================================================================
    
    np.random.seed(42)
    n_users = 12000  # Above required minimum
    
    # Simulate realistic experiment data
    user_ids = [f"user_{i}" for i in range(n_users)]
    timestamps = [
        datetime.now() - timedelta(hours=np.random.randint(0, 336))  # Last 14 days
        for _ in range(n_users)
    ]
    
    # Random assignment with slight imbalance to test SRM detection
    variant_assignments = np.random.choice(
        ["control", "treatment"],
        size=n_users,
        p=[0.51, 0.49]  # Slight imbalance
    )
    
    # Generate conversions with realistic rates
    conversions = []
    for variant in variant_assignments:
        if variant == "control":
            conv = np.random.binomial(1, 0.08)  # 8% baseline
        else:
            conv = np.random.binomial(1, 0.10)  # 10% treatment (25% relative lift)
        conversions.append(conv)
    
    # Guardrail metrics
    cart_abandonments = []
    page_load_acceptable = []
    
    for variant, conv in zip(variant_assignments, conversions):
        # Cart abandonment (inverse of conversion, should be neutral)
        cart_abandon = 1 - conv if np.random.random() > 0.1 else np.random.binomial(1, 0.3)
        cart_abandonments.append(cart_abandon)
        
        # Page load time acceptable (should remain stable)
        page_load_ok = np.random.binomial(1, 0.95)
        page_load_acceptable.append(page_load_ok)
    
    # Create DataFrame
    experiment_data = pd.DataFrame({
        'user_id': user_ids,
        'variant': variant_assignments,
        'conversion': conversions,
        'cart_abandonment': cart_abandonments,
        'page_load_time_acceptable': page_load_acceptable,
        'timestamp': timestamps
    })
    
    # ========================================================================
    # STEP 4: Run Analysis (Sequential - Multiple Looks)
    # ========================================================================
    
    print("\n" + "="*80)
    print("RUNNING SEQUENTIAL ANALYSIS")
    print("="*80 + "\n")
    
    # Simulate 3 looks at the data
    look_sizes = [0.4, 0.7, 1.0]  # 40%, 70%, 100% of data
    cumulative_alpha = 0.0
    
    for look_num, fraction in enumerate(look_sizes, 1):
        print(f"\n{'='*80}")
        print(f"LOOK #{look_num} - Analyzing {fraction*100:.0f}% of data")
        print(f"{'='*80}\n")
        
        # Get subset of data
        n_samples = int(len(experiment_data) * fraction)
        current_data = experiment_data.iloc[:n_samples].copy()
        
        # Run analysis
        results = analyzer.analyze(
            data=current_data,
            look_number=look_num,
            cumulative_alpha_spent=cumulative_alpha
        )
        
        # Update cumulative alpha
        cumulative_alpha = results.metadata.get('alpha_spent', cumulative_alpha)
        
        # Generate and print report
        report = analyzer.generate_report(results)
        print(report)
        
        # Check if we should stop early
        if "SHIP" in results.overall_decision:
            print("\n" + "🎉" * 40)
            print("EARLY STOPPING: Significant positive result detected!")
            print("🎉" * 40)
            break
        elif "DO NOT SHIP" in results.overall_decision:
            print("\n" + "🛑" * 40)
            print("EARLY STOPPING: Negative result detected!")
            print("🛑" * 40)
            break
    
    # ========================================================================
    # STEP 5: Export Results (Production would save to database)
    # ========================================================================
    
    # Save results to JSON
    results_dict = {
        'experiment_id': results.experiment_id,
        'timestamp': results.timestamp.isoformat(),
        'status': results.status.value,
        'decision': results.overall_decision,
        'data_quality_score': results.data_quality_score,
        'srm_detected': results.sample_ratio_mismatch_detected,
        'srm_p_value': results.srm_p_value,
        'warnings': results.warnings,
        'recommendations': results.recommendations,
        'metric_results': {
            k: {
                'metric_name': v.metric_name,
                'control_value': v.control_value,
                'treatment_value': v.treatment_value,
                'absolute_diff': v.absolute_diff,
                'relative_diff': v.relative_diff,
                'p_value': v.p_value,
                'ci_lower': v.ci_lower,
                'ci_upper': v.ci_upper,
                'is_significant': v.is_significant
            }
            for k, v in results.metric_results.items()
        },
        'metadata': results.metadata
    }
    
    with open('/home/claude/experiment_results.json', 'w') as f:
        # Convert non-serializable objects
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        json.dump(results_dict, f, indent=2, default=json_serializer)
    
    print("\n" + "="*80)
    print("Results exported to: experiment_results.json")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()

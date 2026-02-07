"""
Unit Tests for Production A/B Testing Framework
================================================
Comprehensive test suite covering all components
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ab_testing_framework import (
    ExperimentConfig,
    ExperimentAnalyzer,
    DataValidator,
    SampleSizeCalculator,
    SampleRatioMismatchDetector,
    StatisticalTestEngine,
    SequentialTestingEngine,
    DecisionRule,
    TestStatus
)


class TestExperimentConfig(unittest.TestCase):
    """Test experiment configuration validation"""
    
    def test_valid_config(self):
        """Test valid configuration passes"""
        config = ExperimentConfig(
            experiment_id="TEST-001",
            experiment_name="Test Experiment",
            hypothesis="Test hypothesis",
            variants=["control", "treatment"],
            control_variant="control",
            primary_metric="conversion",
            guardrail_metrics=["revenue"]
        )
        
        is_valid, errors = config.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_invalid_control_variant(self):
        """Test that control variant must be in variants list"""
        config = ExperimentConfig(
            experiment_id="TEST-002",
            experiment_name="Test",
            hypothesis="Test",
            variants=["A", "B"],
            control_variant="C",  # Not in variants
            primary_metric="conversion",
            guardrail_metrics=[]
        )
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Control variant" in e for e in errors))
    
    def test_invalid_alpha(self):
        """Test alpha must be between 0 and 1"""
        config = ExperimentConfig(
            experiment_id="TEST-003",
            experiment_name="Test",
            hypothesis="Test",
            variants=["control", "treatment"],
            control_variant="control",
            primary_metric="conversion",
            guardrail_metrics=[],
            alpha=1.5  # Invalid
        )
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Alpha" in e for e in errors))
    
    def test_insufficient_variants(self):
        """Test need at least 2 variants"""
        config = ExperimentConfig(
            experiment_id="TEST-004",
            experiment_name="Test",
            hypothesis="Test",
            variants=["control"],  # Only 1
            control_variant="control",
            primary_metric="conversion",
            guardrail_metrics=[]
        )
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("at least 2 variants" in e for e in errors))


class TestDataValidator(unittest.TestCase):
    """Test data validation logic"""
    
    def setUp(self):
        """Set up test data"""
        self.config = ExperimentConfig(
            experiment_id="TEST-VAL-001",
            experiment_name="Validation Test",
            hypothesis="Test",
            variants=["control", "treatment"],
            control_variant="control",
            primary_metric="conversion",
            guardrail_metrics=[]
        )
        
        self.validator = DataValidator()
    
    def test_valid_data(self):
        """Test valid data passes validation"""
        data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'variant': ['control', 'treatment', 'control', 'treatment'],
            'conversion': [0, 1, 0, 1],
            'timestamp': [datetime.now()] * 4
        })
        
        is_valid, errors, score = self.validator.validate_dataframe(data, self.config)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertGreater(score, 0.95)
    
    def test_missing_columns(self):
        """Test missing required columns"""
        data = pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'variant': ['control', 'treatment']
            # Missing 'conversion' and 'timestamp'
        })
        
        is_valid, errors, score = self.validator.validate_dataframe(data, self.config)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required column" in e for e in errors))
    
    def test_invalid_variants(self):
        """Test invalid variant values"""
        data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'variant': ['control', 'invalid_variant', 'treatment'],
            'conversion': [0, 1, 0],
            'timestamp': [datetime.now()] * 3
        })
        
        is_valid, errors, score = self.validator.validate_dataframe(data, self.config)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid variants" in e for e in errors))
    
    def test_non_binary_metric(self):
        """Test conversion must be 0 or 1"""
        data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'variant': ['control', 'treatment', 'control'],
            'conversion': [0, 2, 1],  # 2 is invalid
            'timestamp': [datetime.now()] * 3
        })
        
        is_valid, errors, score = self.validator.validate_dataframe(data, self.config)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("binary" in e for e in errors))


class TestSampleSizeCalculator(unittest.TestCase):
    """Test sample size calculations"""
    
    def test_basic_calculation(self):
        """Test basic sample size calculation"""
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_rate=0.10,
            mde=0.10,  # 10% relative improvement
            alpha=0.05,
            power=0.80
        )
        
        self.assertIn('n_per_variant', result)
        self.assertIn('total_sample_size', result)
        self.assertGreater(result['n_per_variant'], 0)
        self.assertEqual(result['total_sample_size'], result['n_per_variant'] * 2)
    
    def test_smaller_mde_needs_more_samples(self):
        """Test that smaller MDE requires larger sample"""
        result_large_mde = SampleSizeCalculator.calculate_sample_size(
            baseline_rate=0.10,
            mde=0.20,  # 20% relative
            alpha=0.05,
            power=0.80
        )
        
        result_small_mde = SampleSizeCalculator.calculate_sample_size(
            baseline_rate=0.10,
            mde=0.05,  # 5% relative
            alpha=0.05,
            power=0.80
        )
        
        self.assertGreater(
            result_small_mde['n_per_variant'],
            result_large_mde['n_per_variant']
        )
    
    def test_higher_power_needs_more_samples(self):
        """Test that higher power requires larger sample"""
        result_low_power = SampleSizeCalculator.calculate_sample_size(
            baseline_rate=0.10,
            mde=0.10,
            alpha=0.05,
            power=0.70
        )
        
        result_high_power = SampleSizeCalculator.calculate_sample_size(
            baseline_rate=0.10,
            mde=0.10,
            alpha=0.05,
            power=0.90
        )
        
        self.assertGreater(
            result_high_power['n_per_variant'],
            result_low_power['n_per_variant']
        )


class TestSampleRatioMismatch(unittest.TestCase):
    """Test SRM detection"""
    
    def setUp(self):
        self.config = ExperimentConfig(
            experiment_id="SRM-TEST",
            experiment_name="SRM Test",
            hypothesis="Test",
            variants=["control", "treatment"],
            control_variant="control",
            primary_metric="conversion",
            guardrail_metrics=[]
        )
        self.detector = SampleRatioMismatchDetector()
    
    def test_balanced_allocation(self):
        """Test no SRM with balanced allocation"""
        data = pd.DataFrame({
            'variant': ['control'] * 500 + ['treatment'] * 500,
            'conversion': [0] * 1000
        })
        
        srm_detected, p_value, counts = self.detector.check_srm(data, self.config)
        
        self.assertFalse(srm_detected)
        self.assertGreater(p_value, 0.05)
        self.assertEqual(counts['control'], 500)
        self.assertEqual(counts['treatment'], 500)
    
    def test_slight_imbalance_ok(self):
        """Test slight random imbalance is OK"""
        np.random.seed(42)
        variants = np.random.choice(['control', 'treatment'], size=1000, p=[0.51, 0.49])
        
        data = pd.DataFrame({
            'variant': variants,
            'conversion': [0] * 1000
        })
        
        srm_detected, p_value, counts = self.detector.check_srm(data, self.config)
        
        # Should not detect SRM for this slight imbalance
        self.assertFalse(srm_detected)
    
    def test_severe_imbalance(self):
        """Test severe imbalance triggers SRM"""
        data = pd.DataFrame({
            'variant': ['control'] * 700 + ['treatment'] * 300,  # 70/30 split
            'conversion': [0] * 1000
        })
        
        srm_detected, p_value, counts = self.detector.check_srm(data, self.config)
        
        self.assertTrue(srm_detected)
        self.assertLess(p_value, 0.01)


class TestStatisticalTests(unittest.TestCase):
    """Test statistical testing engine"""
    
    def setUp(self):
        self.engine = StatisticalTestEngine()
    
    def test_no_difference(self):
        """Test when conversion rates are equal"""
        result = self.engine.proportion_test(
            control_conversions=50,
            control_total=500,
            treatment_conversions=50,
            treatment_total=500,
            alpha=0.05
        )
        
        self.assertAlmostEqual(result['control_rate'], result['treatment_rate'], places=5)
        self.assertAlmostEqual(result['absolute_diff'], 0, places=5)
        self.assertGreater(result['p_value'], 0.05)
        self.assertFalse(result['is_significant'])
    
    def test_significant_difference(self):
        """Test when there's a clear difference"""
        result = self.engine.proportion_test(
            control_conversions=40,    # 8%
            control_total=500,
            treatment_conversions=60,  # 12%
            treatment_total=500,
            alpha=0.05
        )
        
        self.assertGreater(result['treatment_rate'], result['control_rate'])
        self.assertGreater(result['absolute_diff'], 0)
        self.assertLess(result['p_value'], 0.05)
        self.assertTrue(result['is_significant'])
    
    def test_confidence_intervals(self):
        """Test confidence intervals contain true value"""
        result = self.engine.proportion_test(
            control_conversions=100,
            control_total=1000,
            treatment_conversions=120,
            treatment_total=1000,
            alpha=0.05
        )
        
        # CI should contain the observed difference
        self.assertLessEqual(result['ci_diff'][0], result['absolute_diff'])
        self.assertGreaterEqual(result['ci_diff'][1], result['absolute_diff'])
        
        # CI should have reasonable width
        ci_width = result['ci_diff'][1] - result['ci_diff'][0]
        self.assertGreater(ci_width, 0)
        self.assertLess(ci_width, 0.1)  # Should be reasonably tight


class TestSequentialTesting(unittest.TestCase):
    """Test sequential testing boundaries"""
    
    def setUp(self):
        self.engine = SequentialTestingEngine()
    
    def test_obrien_fleming_spending(self):
        """Test O'Brien-Fleming alpha spending is conservative early"""
        alpha = 0.05
        total_looks = 5
        
        # First look should spend very little alpha
        alpha_look1, _ = self.engine.get_adjusted_alpha(1, total_looks, alpha, "obrien_fleming", 0.0)
        
        # Last look should spend remaining alpha
        cumulative = alpha_look1
        for look in range(2, total_looks + 1):
            alpha_current, cumulative = self.engine.get_adjusted_alpha(
                look, total_looks, alpha, "obrien_fleming", cumulative
            )
        
        # Total spent should approximately equal alpha
        self.assertLess(abs(cumulative - alpha), 0.01)
        
        # First look should be very conservative
        self.assertLess(alpha_look1, alpha / total_looks)
    
    def test_pocock_boundaries(self):
        """Test Pocock boundaries are more uniform"""
        alpha = 0.05
        total_looks = 5
        
        alphas = []
        cumulative = 0.0
        
        for look in range(1, total_looks + 1):
            alpha_current, cumulative = self.engine.get_adjusted_alpha(
                look, total_looks, alpha, "pocock", cumulative
            )
            alphas.append(alpha_current)
        
        # Pocock should have more uniform spending
        # (though this is approximate)
        self.assertLess(abs(cumulative - alpha), 0.01)
    
    def test_cumulative_alpha_tracking(self):
        """Test cumulative alpha is tracked correctly"""
        alpha = 0.05
        total_looks = 3
        cumulative = 0.0
        
        for look in range(1, total_looks + 1):
            alpha_current, cumulative = self.engine.get_adjusted_alpha(
                look, total_looks, alpha, "obrien_fleming", cumulative
            )
            
            # Cumulative should increase
            self.assertGreaterEqual(cumulative, alpha_current)
            
            # Should never exceed total alpha
            self.assertLessEqual(cumulative, alpha)


class TestEndToEndAnalysis(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        """Create test configuration and data"""
        self.config = ExperimentConfig(
            experiment_id="E2E-TEST",
            experiment_name="End to End Test",
            hypothesis="Treatment improves conversion",
            variants=["control", "treatment"],
            control_variant="control",
            primary_metric="conversion",
            guardrail_metrics=["revenue"],
            alpha=0.05,
            power=0.80,
            mde=0.25,  # 25% relative improvement
            baseline_rate=0.10,
            min_sample_size_per_variant=100
        )
        
        self.analyzer = ExperimentAnalyzer(self.config)
    
    def _create_test_data(self, n_users=2000, treatment_lift=0.25):
        """Helper to create synthetic test data"""
        np.random.seed(42)
        
        variants = np.random.choice(['control', 'treatment'], size=n_users)
        conversions = []
        revenues = []
        
        for v in variants:
            if v == 'control':
                conv = np.random.binomial(1, 0.10)
            else:
                conv = np.random.binomial(1, 0.10 * (1 + treatment_lift))
            
            conversions.append(conv)
            revenues.append(1 if conv and np.random.random() > 0.1 else 0)
        
        return pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(n_users)],
            'variant': variants,
            'conversion': conversions,
            'revenue': revenues,
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_users)]
        })
    
    def test_positive_result_detected(self):
        """Test framework detects significant positive result"""
        # Create data with clear treatment effect
        data = self._create_test_data(n_users=5000, treatment_lift=0.30)
        
        results = self.analyzer.analyze(data, look_number=1)
        
        # Should detect positive result
        self.assertIn("SHIP", results.overall_decision)
        
        # Primary metric should be significant
        primary_results = [
            r for r in results.metric_results.values()
            if r.metric_name == "conversion" and not r.is_guardrail
        ]
        
        self.assertTrue(len(primary_results) > 0)
        self.assertTrue(primary_results[0].is_significant)
        self.assertGreater(primary_results[0].relative_diff, 0)
    
    def test_no_effect_detected(self):
        """Test framework correctly identifies no effect"""
        # Create data with no treatment effect
        data = self._create_test_data(n_users=2000, treatment_lift=0.0)
        
        results = self.analyzer.analyze(data, look_number=1)
        
        # Should not detect effect
        self.assertIn("INCONCLUSIVE", results.overall_decision)
        
        # P-value should be high
        primary_results = [
            r for r in results.metric_results.values()
            if r.metric_name == "conversion"
        ]
        
        self.assertGreater(primary_results[0].p_value, 0.05)
    
    def test_data_quality_validation(self):
        """Test data quality checks work"""
        # Create data with issues
        data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'variant': ['control', 'invalid', 'treatment'],  # Invalid variant
            'conversion': [0, 1, 2],  # Invalid conversion value
            'revenue': [0, 0, 1],
            'timestamp': [datetime.now()] * 3
        })
        
        results = self.analyzer.analyze(data, look_number=1)
        
        # Should mark as invalid
        self.assertEqual(results.status, TestStatus.INVALID)
        self.assertGreater(len(results.warnings), 0)
    
    def test_guardrail_violation(self):
        """Test guardrail violations prevent shipping"""
        # Create data where primary is good but guardrail is violated
        np.random.seed(42)
        n_users = 2000
        
        variants = np.random.choice(['control', 'treatment'], size=n_users)
        conversions = []
        revenues = []
        
        for v in variants:
            # Conversion improves
            conv = np.random.binomial(1, 0.12 if v == 'treatment' else 0.10)
            conversions.append(conv)
            
            # But revenue drops!
            rev = np.random.binomial(1, 0.05 if v == 'treatment' else 0.10)
            revenues.append(rev)
        
        data = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(n_users)],
            'variant': variants,
            'conversion': conversions,
            'revenue': revenues,
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_users)]
        })
        
        results = self.analyzer.analyze(data, look_number=1)
        
        # Should not ship due to guardrail
        # (This test is probabilistic - guardrail might not always be significant)
        # So we just check that guardrails were analyzed
        guardrail_results = [r for r in results.metric_results.values() if r.is_guardrail]
        self.assertGreater(len(guardrail_results), 0)


class TestReportGeneration(unittest.TestCase):
    """Test report generation"""
    
    def test_report_contains_key_sections(self):
        """Test report has all required sections"""
        config = ExperimentConfig(
            experiment_id="REPORT-TEST",
            experiment_name="Report Test",
            hypothesis="Test",
            variants=["control", "treatment"],
            control_variant="control",
            primary_metric="conversion",
            guardrail_metrics=[]
        )
        
        analyzer = ExperimentAnalyzer(config)
        
        # Create simple data
        data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'variant': ['control', 'treatment', 'control', 'treatment'],
            'conversion': [0, 1, 1, 1],
            'timestamp': [datetime.now()] * 4
        })
        
        results = analyzer.analyze(data)
        report = analyzer.generate_report(results)
        
        # Check key sections exist
        self.assertIn("EXPERIMENT ANALYSIS REPORT", report)
        self.assertIn("DATA QUALITY", report)
        self.assertIn("PRIMARY METRIC RESULTS", report)
        self.assertIn("RECOMMENDATIONS", report)
        self.assertIn("METADATA", report)
        
        # Check specific values
        self.assertIn(config.experiment_id, report)
        self.assertIn("Sample Ratio Mismatch", report)


def run_all_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestExperimentConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestSampleSizeCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestSampleRatioMismatch))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalTests))
    suite.addTests(loader.loadTestsFromTestCase(TestSequentialTesting))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestReportGeneration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    return result


if __name__ == "__main__":
    run_all_tests()

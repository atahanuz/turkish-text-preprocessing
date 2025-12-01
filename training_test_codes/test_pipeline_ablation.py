#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study Test Script for Turkish Normalization Pipeline
Tests pipeline accuracy by removing one module at a time (6/7 configuration)

This script evaluates the impact of each module by testing the pipeline
with that module disabled. Results show which modules contribute most
to overall accuracy.

Usage:
    python test_pipeline_ablation.py [csv_path] [num_processes]

Arguments:
    csv_path: Path to test CSV file (default: test_normalization_200.csv)
    num_processes: Number of parallel processes (default: CPU count)

Examples:
    python test_pipeline_ablation.py
    python test_pipeline_ablation.py test_normalization_200.csv
    python test_pipeline_ablation.py test_normalization_200.csv 4
"""

import sys
import os
import csv
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from normalization.abbreviation_extension import expand_abbreviation
from normalization.letter_case_transformation import letter_case_transformation, turkish_capitalize
from normalization.diacritic_restoration import correct_word as restore_diacritics, load_vocabulary as load_diacritic_vocab
from normalization.vowel_restoration import restore_vowels
from normalization.spelling_correction import spelling_correction
from normalization.accent_normalization import normalize_accent
from normalization.morho_parser import analyze_text


def get_absolute_path(relative_path):
    """Convert relative path to absolute path based on project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, relative_path)


class AblationPipeline:
    """
    Modified pipeline that can disable specific modules for ablation testing.
    """

    # Module names for reference
    MODULES = [
        'abbreviation_extension',
        'letter_case_transformation',
        'diacritic_restoration',
        'vowel_restoration',
        'spelling_correction',
        'accent_normalization',
        'morpho_parser'
    ]

    def __init__(self, lexicon_path='data/Turkish_Corpus_3M.txt',
                 use_morpho_parser=False, disabled_module=None):
        """
        Initialize the normalization pipeline with optional module disabling.

        Args:
            lexicon_path: Path to Turkish lexicon file
            use_morpho_parser: Whether to use morphological parser
            disabled_module: Name of module to disable (None for full pipeline)
        """
        # Convert to absolute path if relative
        if not os.path.isabs(lexicon_path):
            self.lexicon_path = get_absolute_path(lexicon_path)
        else:
            self.lexicon_path = lexicon_path

        # Pre-load vocabulary for diacritic restoration
        self.diacritic_vocab = load_diacritic_vocab(self.lexicon_path)

        # Configuration
        self.use_morpho_parser = use_morpho_parser
        self.disabled_module = disabled_module

    def normalize_token(self, token):
        """
        Process a single token through the pipeline, skipping disabled module.

        Args:
            token: Input token to normalize

        Returns:
            Normalized token
        """
        # Skip empty tokens
        if not token:
            return token

        # Skip tokens with no alphabetic characters
        if not any(c.isalpha() for c in token):
            return token

        result = token

        # Step 0: Abbreviation Extension
        if self.disabled_module != 'abbreviation_extension':
            result = expand_abbreviation(result)

        # Step 1: Letter Case Transformation
        if self.disabled_module != 'letter_case_transformation':
            result = letter_case_transformation(result)

        # Step 2: Diacritic Restoration
        if self.disabled_module != 'diacritic_restoration':
            result = restore_diacritics(result, self.diacritic_vocab)

        # Step 3: Vowel Restoration
        if self.disabled_module != 'vowel_restoration':
            result = restore_vowels(result, self.lexicon_path)

        # Step 4: Spelling Correction
        if self.disabled_module != 'spelling_correction':
            result = spelling_correction(result, self.lexicon_path, max_edit_distance=2)

        # Step 5: Accent Normalization
        if self.disabled_module != 'accent_normalization':
            result = normalize_accent(result)

        return result

    def _is_proper_noun(self, word, morpho_data):
        """Check if a word is a proper noun based on morphological analysis."""
        if word not in morpho_data:
            return False

        analyses = morpho_data[word]
        for analysis in analyses:
            if '[Prop]' in analysis:
                return True
        return False

    def normalize_sequence(self, tokens):
        """
        Process a sequence of tokens through the normalization pipeline.

        Args:
            tokens: List of tokens or space-separated string

        Returns:
            List of normalized tokens
        """
        # Handle both list and string input
        if isinstance(tokens, str):
            tokens = tokens.split()

        # Get morphological analysis if enabled and not disabled
        morpho_data = {}
        if (self.use_morpho_parser and
            self.disabled_module != 'morpho_parser' and
            tokens):
            try:
                text = ' '.join(tokens)
                morpho_data = analyze_text(text)
            except Exception as e:
                pass  # Silently continue without morpho parser

        normalized_tokens = []
        for token in tokens:
            normalized = self.normalize_token(token)

            # Check if original token is a proper noun and capitalize it
            if (self.use_morpho_parser and
                self.disabled_module != 'morpho_parser' and
                self._is_proper_noun(token, morpho_data)):
                normalized = turkish_capitalize(normalized)

            normalized_tokens.append(normalized)

        return normalized_tokens


def load_test_data(csv_path):
    """Load test data from CSV file"""
    test_cases = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append({
                'input': row['erroneous'],
                'expected': row['correct']
            })

    return test_cases


def process_test_case(test_data, lexicon_path, disabled_module):
    """
    Worker function to process a single test case.

    Args:
        test_data: Tuple of (test_number, test_case)
        lexicon_path: Path to the lexicon file
        disabled_module: Name of module to disable

    Returns:
        Dictionary with test results
    """
    test_number, test_case = test_data
    input_word = test_case['input']
    expected = test_case['expected']

    # Each worker creates its own pipeline instance
    pipeline = AblationPipeline(
        lexicon_path=lexicon_path,
        use_morpho_parser=True,
        disabled_module=disabled_module
    )

    try:
        # Normalize the word
        output = pipeline.normalize_sequence([input_word])[0]

        # Check if correct
        is_correct = output.strip() == expected.strip()

        return {
            'test_number': test_number,
            'input': input_word,
            'expected': expected,
            'output': output,
            'is_correct': is_correct,
            'status': 'PASS' if is_correct else 'FAIL'
        }

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        return {
            'test_number': test_number,
            'input': input_word,
            'expected': expected,
            'output': error_msg,
            'is_correct': False,
            'status': 'ERROR'
        }


def test_single_configuration(csv_path, lexicon_path, disabled_module,
                              test_cases, num_processes):
    """
    Test pipeline with a single module disabled.

    Args:
        csv_path: Path to test CSV
        lexicon_path: Path to lexicon
        disabled_module: Name of module to disable (None for full pipeline)
        test_cases: List of test cases
        num_processes: Number of parallel processes

    Returns:
        Dictionary with test results
    """
    config_name = f"WITHOUT {disabled_module}" if disabled_module else "FULL PIPELINE (7/7)"
    total_tests = len(test_cases)

    print("=" * 80)
    print(f"Testing: {config_name}")
    print("=" * 80)

    # Prepare test data with test numbers
    test_data = [(i, test_case) for i, test_case in enumerate(test_cases, 1)]

    # Create partial function with parameters bound
    worker_func = partial(
        process_test_case,
        lexicon_path=lexicon_path,
        disabled_module=disabled_module
    )

    # Process test cases in parallel
    all_results = []
    completed = 0

    with Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(worker_func, test_data, chunksize=10):
            all_results.append(result)
            completed += 1

            # Print progress every 20 tests
            if completed % 20 == 0 or completed == 1:
                progress = (completed / total_tests) * 100
                print(f"  Progress: {completed}/{total_tests} ({progress:.1f}%)")

    # Sort results by test_number
    all_results.sort(key=lambda x: x['test_number'])

    # Calculate statistics
    correct = sum(1 for r in all_results if r['is_correct'])
    incorrect = total_tests - correct
    accuracy = (correct / total_tests) * 100

    print(f"  Results: {correct}/{total_tests} correct ({accuracy:.2f}%)")
    print()

    return {
        'configuration': config_name,
        'disabled_module': disabled_module,
        'total': total_tests,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'results': all_results
    }


def run_ablation_study(csv_path='test_normalization_200.csv',
                      lexicon_path='data/Turkish_Corpus_3M.txt',
                      num_processes=None):
    """
    Run complete ablation study: test pipeline with each module disabled.

    Args:
        csv_path: Path to test CSV file
        lexicon_path: Path to lexicon file
        num_processes: Number of parallel processes

    Returns:
        Dictionary with all results
    """
    print("=" * 80)
    print("TURKISH NORMALIZATION PIPELINE - ABLATION STUDY")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load test data
    print(f"Loading test data from: {csv_path}")
    test_cases = load_test_data(csv_path)
    total_tests = len(test_cases)
    print(f"Total test cases: {total_tests}")
    print()

    # Determine number of processes
    if num_processes is None:
        num_processes = cpu_count()
    print(f"Using {num_processes} parallel processes")
    print()

    # Test configurations: full pipeline + each module disabled
    configurations = [None] + AblationPipeline.MODULES

    all_configs_results = []

    # Test each configuration
    for config in configurations:
        result = test_single_configuration(
            csv_path=csv_path,
            lexicon_path=lexicon_path,
            disabled_module=config,
            test_cases=test_cases,
            num_processes=num_processes
        )
        all_configs_results.append(result)

    # Summary results
    print("=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print()

    # Sort by accuracy descending
    sorted_results = sorted(all_configs_results, key=lambda x: x['accuracy'], reverse=True)

    print(f"{'Configuration':<45} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 80)

    baseline_accuracy = None
    for result in sorted_results:
        config_name = result['configuration']
        accuracy = result['accuracy']
        correct = result['correct']

        # Track baseline (full pipeline)
        if result['disabled_module'] is None:
            baseline_accuracy = accuracy
            print(f"{config_name:<45} {accuracy:>9.2f}% {correct:>10d} ★ BASELINE")
        else:
            # Calculate impact (negative means module helps)
            impact = accuracy - baseline_accuracy if baseline_accuracy else 0
            impact_str = f"({impact:+.2f}%)"
            print(f"{config_name:<45} {accuracy:>9.2f}% {correct:>10d} {impact_str}")

    print()

    # Module importance analysis
    print("=" * 80)
    print("MODULE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print()
    print("Impact shows accuracy change when module is REMOVED (negative = helpful)")
    print()

    # Calculate impact for each module
    module_impacts = []
    for result in all_configs_results:
        if result['disabled_module'] is not None:
            impact = result['accuracy'] - baseline_accuracy
            module_impacts.append({
                'module': result['disabled_module'],
                'impact': impact,
                'accuracy_without': result['accuracy']
            })

    # Sort by impact (most negative = most important)
    module_impacts.sort(key=lambda x: x['impact'])

    print(f"{'Module':<35} {'Impact':>12} {'Accuracy w/o':>15}")
    print("-" * 80)
    for item in module_impacts:
        impact_str = f"{item['impact']:+.2f}%"
        importance = "CRITICAL" if item['impact'] < -5 else "HIGH" if item['impact'] < -2 else "MEDIUM" if item['impact'] < -0.5 else "LOW"
        print(f"{item['module']:<35} {impact_str:>12} {item['accuracy_without']:>14.2f}% [{importance}]")

    print()
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Save detailed results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = f'ablation_study_results_{timestamp}.csv'

    print(f"Saving detailed results to: {output_csv}")

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['configuration', 'disabled_module', 'total', 'correct',
                     'incorrect', 'accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in sorted_results:
            writer.writerow({
                'configuration': result['configuration'],
                'disabled_module': result['disabled_module'] or 'NONE',
                'total': result['total'],
                'correct': result['correct'],
                'incorrect': result['incorrect'],
                'accuracy': f"{result['accuracy']:.2f}"
            })

    print(f"✓ Summary saved to {output_csv}")

    # Save per-test detailed results
    detailed_csv = f'ablation_study_detailed_{timestamp}.csv'
    print(f"Saving per-test results to: {detailed_csv}")

    with open(detailed_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['configuration', 'test_number', 'input', 'expected',
                     'output', 'is_correct', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for config_result in all_configs_results:
            for test_result in config_result['results']:
                writer.writerow({
                    'configuration': config_result['configuration'],
                    'test_number': test_result['test_number'],
                    'input': test_result['input'],
                    'expected': test_result['expected'],
                    'output': test_result['output'],
                    'is_correct': test_result['is_correct'],
                    'status': test_result['status']
                })

    print(f"✓ Detailed results saved to {detailed_csv}")
    print()

    return {
        'all_results': all_configs_results,
        'module_impacts': module_impacts,
        'baseline_accuracy': baseline_accuracy,
        'summary_file': output_csv,
        'detailed_file': detailed_csv
    }


if __name__ == '__main__':
    # Parse command-line arguments
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'test_normalization_200.csv'
    num_processes = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Run ablation study
    results = run_ablation_study(csv_path, num_processes=num_processes)

    # Exit with success
    sys.exit(0)

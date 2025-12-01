#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Turkish normalization pipeline
Tests pipeline accuracy with test_normalization_200.csv

This script uses parallel processing to speed up test execution.
Usage:
    python test_pipeline.py [csv_path] [num_processes]

Arguments:
    csv_path: Path to test CSV file (default: test_normalization_200.csv)
    num_processes: Number of parallel processes (default: CPU count)

Examples:
    python test_pipeline.py
    python test_pipeline.py test_normalization_200.csv
    python test_pipeline.py test_normalization_200.csv 4
"""

import sys
import os
import csv
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from normalization.normalization_pipeline import NormalizationPipeline


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


def process_test_case(test_data, lexicon_path):
    """
    Worker function to process a single test case.
    This will be called in parallel by multiple processes.

    Args:
        test_data: Tuple of (test_number, test_case)
        lexicon_path: Path to the lexicon file

    Returns:
        Dictionary with test results
    """
    test_number, test_case = test_data
    input_word = test_case['input']
    expected = test_case['expected']

    # Each worker creates its own pipeline instance
    pipeline = NormalizationPipeline(lexicon_path=lexicon_path, use_morpho_parser=True)

    try:
        # Normalize the word using normalize_sequence to enable proper noun detection
        output = pipeline.normalize_sequence([input_word])[0]

        # Check if correct (ignore whitespace differences)
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


def test_pipeline(csv_path='test_normalization_200.csv', lexicon_path='data/Turkish_Corpus_3M.txt', num_processes=None):
    """
    Test the normalization pipeline with CSV data using parallel processing.

    Args:
        csv_path: Path to the test data CSV file
        lexicon_path: Path to the lexicon file
        num_processes: Number of parallel processes (defaults to CPU count)
    """

    print("=" * 80)
    print("TURKISH NORMALIZATION PIPELINE TEST (PARALLEL)")
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

    print("=" * 80)
    print("RUNNING TESTS IN PARALLEL")
    print("=" * 80)
    print()

    # Prepare test data with test numbers
    test_data = [(i, test_case) for i, test_case in enumerate(test_cases, 1)]

    # Create partial function with lexicon_path bound
    worker_func = partial(process_test_case, lexicon_path=lexicon_path)

    # Process test cases in parallel
    all_results = []
    completed = 0

    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for better performance (results may come back out of order)
        for result in pool.imap_unordered(worker_func, test_data, chunksize=10):
            all_results.append(result)
            completed += 1

            # Print progress every 10 tests
            if completed % 10 == 0 or completed == 1:
                progress = (completed / total_tests) * 100
                print(f"Progress: {completed}/{total_tests} ({progress:.1f}%) - Last: '{result['input']}' → '{result['output']}'")

    # Sort results by test_number to maintain original order
    all_results.sort(key=lambda x: x['test_number'])

    # Calculate statistics
    correct = 0
    incorrect = 0
    errors = []

    for result in all_results:
        if result['is_correct']:
            correct += 1
        else:
            incorrect += 1
            if result['status'] != 'ERROR':
                errors.append({
                    'input': result['input'],
                    'expected': result['expected'],
                    'output': result['output']
                })
            else:
                errors.append({
                    'input': result['input'],
                    'expected': result['expected'],
                    'output': result['output']
                })

    # Final results
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total tests:      {total_tests}")
    print(f"Correct:          {correct} ({(correct/total_tests)*100:.2f}%)")
    print(f"Incorrect:        {incorrect} ({(incorrect/total_tests)*100:.2f}%)")
    print()

    # Show errors if any
    if errors:
        print("=" * 80)
        print(f"ERRORS ({len(errors)} total)")
        print("=" * 80)
        print()

        # Show first 20 errors
        for i, error in enumerate(errors[:20], 1):
            print(f"{i:2d}. Input: '{error['input']}'")
            print(f"    Expected: '{error['expected']}'")
            print(f"    Got:      '{error['output']}'")
            print()

        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors")
            print()

    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Save all results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = f'pipeline_test_results_{timestamp}.csv'

    print(f"Saving detailed results to: {output_csv}")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['test_number', 'input', 'expected', 'output', 'is_correct', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"✓ Results saved successfully to {output_csv}")
    print()

    return {
        'total': total_tests,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': (correct/total_tests)*100,
        'errors': errors,
        'results_file': output_csv
    }


if __name__ == '__main__':
    # Parse command-line arguments
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'test_normalization_200.csv'
    num_processes = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Run tests with parallel processing
    results = test_pipeline(csv_path, num_processes=num_processes)

    # Exit with appropriate code
    sys.exit(0 if results['accuracy'] > 80 else 1)

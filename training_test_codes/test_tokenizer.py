"""
Test tokenizer modules against TR-BOUN-UD test data.
"""
import ast
import sys
from typing import List, Tuple
from tokenizer.tokenizer_rule_based import TurkishTokenizer


def load_test_data(file_path: str) -> List[Tuple[str, List[str]]]:
    """
    Load test data from TR-BOUN-UD format.
    Format: sentence, ['token1', 'token2', ...]

    Args:
        file_path: Path to test file

    Returns:
        List of (sentence, expected_tokens) tuples
    """
    test_cases = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split by first comma to separate sentence from tokens
            parts = line.split(', [', 1)
            if len(parts) != 2:
                continue

            sentence = parts[0]
            tokens_str = '[' + parts[1]

            try:
                expected_tokens = ast.literal_eval(tokens_str)
                test_cases.append((sentence, expected_tokens))
            except Exception as e:
                print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
                continue

    return test_cases


def calculate_metrics(predicted: List[str], expected: List[str]) -> dict:
    """
    Calculate precision, recall, and F1 score for tokenization.

    Args:
        predicted: List of predicted tokens
        expected: List of expected tokens

    Returns:
        Dictionary with precision, recall, f1, and accuracy metrics
    """
    # Exact match
    exact_match = predicted == expected

    # Token-level precision and recall
    predicted_set = set(predicted)
    expected_set = set(expected)

    if len(predicted) == 0:
        precision = 0.0
    else:
        true_positives = len(predicted_set & expected_set)
        precision = true_positives / len(predicted) if len(predicted) > 0 else 0.0

    if len(expected) == 0:
        recall = 0.0
    else:
        true_positives = len(predicted_set & expected_set)
        recall = true_positives / len(expected) if len(expected) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        'exact_match': exact_match,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predicted_count': len(predicted),
        'expected_count': len(expected)
    }


def test_rule_based_tokenizer(test_data: List[Tuple[str, List[str]]],
                              mwe_file: str = None) -> dict:
    """
    Test the rule-based tokenizer.

    Args:
        test_data: List of (sentence, expected_tokens) tuples
        mwe_file: Path to multi-word expression file (optional)

    Returns:
        Dictionary with overall metrics
    """
    print("\n" + "=" * 70)
    print("Testing Rule-Based Tokenizer")
    print("=" * 70)

    tokenizer = TurkishTokenizer(mwe_file) if mwe_file else TurkishTokenizer()

    total_exact_matches = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    errors = []

    for i, (sentence, expected_tokens) in enumerate(test_data, 1):
        predicted_tokens = tokenizer.tokenize(sentence)
        metrics = calculate_metrics(predicted_tokens, expected_tokens)

        total_exact_matches += 1 if metrics['exact_match'] else 0
        total_precision += metrics['precision']
        total_recall += metrics['recall']
        total_f1 += metrics['f1']

        # Store errors for analysis
        if not metrics['exact_match']:
            errors.append({
                'sentence': sentence,
                'expected': expected_tokens,
                'predicted': predicted_tokens,
                'metrics': metrics
            })

    # Calculate overall metrics
    num_samples = len(test_data)
    results = {
        'total_samples': num_samples,
        'exact_match_count': total_exact_matches,
        'accuracy': total_exact_matches / num_samples if num_samples > 0 else 0.0,
        'avg_precision': total_precision / num_samples if num_samples > 0 else 0.0,
        'avg_recall': total_recall / num_samples if num_samples > 0 else 0.0,
        'avg_f1': total_f1 / num_samples if num_samples > 0 else 0.0,
        'errors': errors
    }

    return results


def test_ml_tokenizer(test_data: List[Tuple[str, List[str]]],
                     model_path: str = 'tokenizer_model.pkl',
                     vectorizer_path: str = 'tokenizer_vectorizer.pkl') -> dict:
    """
    Test the ML-based tokenizer.

    Args:
        test_data: List of (sentence, expected_tokens) tuples
        model_path: Path to trained model
        vectorizer_path: Path to trained vectorizer

    Returns:
        Dictionary with overall metrics
    """
    print("\n" + "=" * 70)
    print("Testing ML-Based Tokenizer")
    print("=" * 70)

    try:
        from tokenizer.tokenizer_inference import load_trained_tokenizer, tokenize

        model, vectorizer = load_trained_tokenizer(model_path, vectorizer_path)

        total_exact_matches = 0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        errors = []

        for i, (sentence, expected_tokens) in enumerate(test_data, 1):
            predicted_tokens = tokenize(sentence, model, vectorizer)
            metrics = calculate_metrics(predicted_tokens, expected_tokens)

            total_exact_matches += 1 if metrics['exact_match'] else 0
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']

            # Store errors for analysis
            if not metrics['exact_match']:
                errors.append({
                    'sentence': sentence,
                    'expected': expected_tokens,
                    'predicted': predicted_tokens,
                    'metrics': metrics
                })

            if (i % 100) == 0:
                print(f"Processed {i}/{len(test_data)} samples...")

        # Calculate overall metrics
        num_samples = len(test_data)
        results = {
            'total_samples': num_samples,
            'exact_match_count': total_exact_matches,
            'accuracy': total_exact_matches / num_samples if num_samples > 0 else 0.0,
            'avg_precision': total_precision / num_samples if num_samples > 0 else 0.0,
            'avg_recall': total_recall / num_samples if num_samples > 0 else 0.0,
            'avg_f1': total_f1 / num_samples if num_samples > 0 else 0.0,
            'errors': errors
        }

        return results

    except FileNotFoundError as e:
        print(f"\nError: Could not load ML model files: {e}")
        print("Skipping ML tokenizer test.")
        return None
    except Exception as e:
        print(f"\nError testing ML tokenizer: {e}")
        return None


def print_results(results: dict, tokenizer_name: str):
    """Print results in a formatted way."""
    if results is None:
        return

    print(f"\n{tokenizer_name} Results:")
    print("-" * 70)
    print(f"Total Samples:        {results['total_samples']}")
    print(f"Exact Matches:        {results['exact_match_count']}")
    print(f"Accuracy:             {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Average Precision:    {results['avg_precision']:.4f}")
    print(f"Average Recall:       {results['avg_recall']:.4f}")
    print(f"Average F1 Score:     {results['avg_f1']:.4f}")
    print(f"Error Count:          {len(results['errors'])}")


def show_error_examples(results: dict, num_examples: int = 5):
    """Show some error examples for analysis."""
    if results is None or not results['errors']:
        return

    print(f"\n\nExample Errors (showing first {num_examples}):")
    print("=" * 70)

    for i, error in enumerate(results['errors'][:num_examples], 1):
        print(f"\nExample {i}:")
        print(f"Sentence:  {error['sentence']}")
        print(f"Expected:  {error['expected']}")
        print(f"Predicted: {error['predicted']}")
        print(f"Precision: {error['metrics']['precision']:.4f}, "
              f"Recall: {error['metrics']['recall']:.4f}, "
              f"F1: {error['metrics']['f1']:.4f}")


def main():
    """Main test function."""
    print("=" * 70)
    print("Turkish Tokenizer Test Suite")
    print("=" * 70)

    # Load test data
    test_file = 'tr_boun-ud-test_tokens.txt'
    print(f"\nLoading test data from: {test_file}")

    try:
        test_data = load_test_data(test_file)
        print(f"Loaded {len(test_data)} test cases")
    except FileNotFoundError:
        print(f"Error: Test file '{test_file}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)

    # Test rule-based tokenizer
    rule_based_results = test_rule_based_tokenizer(test_data)
    print_results(rule_based_results, "Rule-Based Tokenizer")

    # Test ML tokenizer (if available)
    ml_results = test_ml_tokenizer(
        test_data,
        model_path='tokenizer/tokenizer_model.pkl',
        vectorizer_path='tokenizer/tokenizer_vectorizer.pkl'
    )
    if ml_results:
        print_results(ml_results, "ML-Based Tokenizer")

    # Compare results
    if ml_results:
        print("\n" + "=" * 70)
        print("Comparison Summary")
        print("=" * 70)
        print(f"{'Metric':<20} {'Rule-Based':<15} {'ML-Based':<15} {'Winner'}")
        print("-" * 70)

        metrics = ['accuracy', 'avg_precision', 'avg_recall', 'avg_f1']
        for metric in metrics:
            rb_val = rule_based_results[metric]
            ml_val = ml_results[metric]
            winner = 'Rule-Based' if rb_val > ml_val else 'ML-Based' if ml_val > rb_val else 'Tie'
            print(f"{metric:<20} {rb_val:<15.4f} {ml_val:<15.4f} {winner}")

    # Show some error examples
    print("\n" + "=" * 70)
    print("Error Analysis - Rule-Based Tokenizer")
    show_error_examples(rule_based_results, num_examples=10)

    if ml_results:
        print("\n" + "=" * 70)
        print("Error Analysis - ML-Based Tokenizer")
        show_error_examples(ml_results, num_examples=10)


if __name__ == "__main__":
    main()

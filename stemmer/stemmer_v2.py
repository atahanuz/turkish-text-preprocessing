import re
import csv
from typing import List, Set, Tuple, Dict, Optional
from collections import deque
from enum import IntEnum


class SuffixCategory(IntEnum):
    """Categories of Turkish suffixes"""
    DERIVATIONAL = 1  # Changes word class or meaning significantly
    INFLECTIONAL = 2  # Grammar markers (tense, person, number, etc.)
    FUNCTIONAL = 3  # Functional words (question particles, emphasis, etc.)


class ImprovedTurkishStemmer:
    def __init__(self,
                 suffix_categories_file: str = None,
                 suffix_file: str = None,
                 words_file: str = None,
                 min_stem_length: int = 2):
        """
        Initialize the improved Turkish stemmer with phonological rules and advanced features.

        Args:
            suffix_categories_file: Path to CSV file containing suffix categories
            suffix_file: Path to file containing suffixes (one per line)
            words_file: Path to file containing valid Turkish words (one per line)
            min_stem_length: Minimum length of the stem (to avoid over-stemming)
        """
        self.suffixes = []
        self.valid_words = set()
        self.min_stem_length = min_stem_length
        self.suffix_categories = {}
        self.categorized_suffixes = {cat: [] for cat in SuffixCategory}

        # Load suffix categories from CSV if provided
        if suffix_categories_file:
            self.load_suffix_categories(suffix_categories_file)

        # Load additional suffixes if provided (backward compatibility)
        if suffix_file:
            self.load_suffixes(suffix_file)

        if words_file:
            self.load_words(words_file)

    def load_suffix_categories(self, csv_file: str):
        """
        Load suffix categories from a CSV file.

        Args:
            csv_file: Path to CSV file with columns 'word' (suffix) and 'category'
        """
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                suffixes_loaded = []

                for row in reader:
                    suffix = row['word'].strip()
                    category_str = row['category'].strip()

                    # Remove leading dash if present
                    if suffix.startswith('-'):
                        suffix = suffix[1:]
                    # Remove trailing dash if present (for intermediate suffixes)
                    if suffix.endswith('-'):
                        suffix = suffix[:-1]

                    if not suffix:
                        continue

                    # Map category string to enum
                    if category_str.lower() == 'derivational':
                        category = SuffixCategory.DERIVATIONAL
                    elif category_str.lower() == 'inflectional':
                        category = SuffixCategory.INFLECTIONAL
                    elif category_str.lower() == 'functional':
                        category = SuffixCategory.FUNCTIONAL
                    else:
                        # Default to inflectional if unknown
                        category = SuffixCategory.INFLECTIONAL

                    # Store the suffix and its category
                    self.suffix_categories[suffix] = category
                    self.categorized_suffixes[category].append(suffix)
                    suffixes_loaded.append(suffix)

                # Update the main suffix list (sorted by length, longest first)
                self.suffixes = sorted(set(suffixes_loaded), key=len, reverse=True)

                # Also sort categorized suffixes
                for category in self.categorized_suffixes:
                    self.categorized_suffixes[category] = sorted(
                        set(self.categorized_suffixes[category]),
                        key=len,
                        reverse=True
                    )

                print(f"Loaded {len(self.suffixes)} suffixes from CSV with categories:")
                print(f"  - Derivational: {len(self.categorized_suffixes[SuffixCategory.DERIVATIONAL])}")
                print(f"  - Inflectional: {len(self.categorized_suffixes[SuffixCategory.INFLECTIONAL])}")
                print(f"  - Functional: {len(self.categorized_suffixes[SuffixCategory.FUNCTIONAL])}")

        except FileNotFoundError:
            print(f"Warning: Suffix categories file '{csv_file}' not found.")
        except Exception as e:
            print(f"Error loading suffix categories: {e}")

    def load_suffixes(self, suffix_file: str):
        """
        Load additional suffixes from file (for backward compatibility).
        These will be categorized as INFLECTIONAL by default if not already in categories.
        """
        try:
            with open(suffix_file, 'r', encoding='utf-8') as f:
                new_suffixes = []
                for line in f:
                    suffix = line.strip()
                    if suffix:
                        # Remove leading dash if present
                        if suffix.startswith('-'):
                            suffix = suffix[1:]
                        # Remove trailing dash if present
                        if suffix.endswith('-'):
                            suffix = suffix[:-1]
                        if suffix and suffix not in self.suffix_categories:
                            new_suffixes.append(suffix)
                            # Default category for unknown suffixes
                            self.suffix_categories[suffix] = SuffixCategory.INFLECTIONAL
                            self.categorized_suffixes[SuffixCategory.INFLECTIONAL].append(suffix)

                if new_suffixes:
                    # Merge and resort
                    all_suffixes = set(self.suffixes) | set(new_suffixes)
                    self.suffixes = sorted(all_suffixes, key=len, reverse=True)

                    # Resort categorized suffixes
                    for category in self.categorized_suffixes:
                        self.categorized_suffixes[category] = sorted(
                            set(self.categorized_suffixes[category]),
                            key=len,
                            reverse=True
                        )

                    print(f"Loaded {len(new_suffixes)} additional suffixes from file.")

        except FileNotFoundError:
            print(f"Warning: Suffix file '{suffix_file}' not found.")

    def load_words(self, words_file: str):
        """Load valid Turkish words from file."""
        try:
            with open(words_file, 'r', encoding='utf-8') as f:
                self.valid_words = set(line.strip().lower() for line in f if line.strip())
            print(f"Loaded {len(self.valid_words)} valid Turkish words.")
        except FileNotFoundError:
            print(f"Warning: Words file '{words_file}' not found.")
            self.valid_words = set()

    def apply_phonological_rules(self, stem: str, removed_suffix: str) -> List[str]:
        """
        Generate possible stem variants considering Turkish phonological changes.

        Args:
            stem: The stem after suffix removal
            removed_suffix: The suffix that was removed

        Returns:
            List of possible stem variants
        """
        if not stem:
            return [stem]

        variants = [stem]

        # 1. Consonant softening (reverse the softening that happens with suffixation)
        # When certain consonants appear at the end of stem before suffix, they might be softened versions
        # We need to check the hardened versions
        hardening_map = {
            'b': 'p',  # kitab → kitap
            'c': 'ç',  # ağac → ağaç
            'd': 't',  # yurd → yurt
            'ğ': 'k',  # ayağ → ayak
            'g': 'k'  # renk variation
        }

        if stem and stem[-1] in hardening_map:
            hardened = stem[:-1] + hardening_map[stem[-1]]
            variants.append(hardened)

        # 2. Vowel dropping reversal
        # Some words drop their last vowel when adding suffixes
        # Common pattern: CVC[ıiuü]C + suffix → CVCC + suffix
        if len(stem) >= 2:
            last_char = stem[-1]
            second_last = stem[-2] if len(stem) > 1 else ''

            # Check if we have consonant cluster at the end
            if self._is_consonant(last_char) and self._is_consonant(second_last):
                # Try inserting narrow vowels
                for vowel in ['ı', 'i', 'u', 'ü']:
                    # Insert vowel between last two consonants
                    variant = stem[:-1] + vowel + stem[-1]
                    variants.append(variant)

        # 3. Consonant doubling reversal
        # Some consonants are doubled before vowel-initial suffixes
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] in 'kpçtfsşh':
            # Remove the doubling
            variants.append(stem[:-1])

        # 4. 'n' buffer consonant removal
        # Turkish uses 'n' as a buffer consonant in some cases
        if (len(stem) > 2 and stem[-1] == 'n' and
                removed_suffix and removed_suffix[0] in 'aeıioöuü'):
            variants.append(stem[:-1])

        # 5. Y-buffer removal (for vowel-ending stems)
        if len(stem) > 2 and stem[-1] == 'y' and stem[-2] in 'aeıioöuü':
            variants.append(stem[:-1])

        # 6. Possessive special case
        # Words ending in 'su/sü' (water compounds) have special behavior
        if stem.endswith('su') or stem.endswith('sü'):
            # Try with 'y' buffer
            variants.append(stem[:-1] + 'yu')
            variants.append(stem[:-1] + 'yü')

        return list(set(variants))  # Remove duplicates

    def _is_consonant(self, char: str) -> bool:
        """Check if a character is a consonant"""
        return char and char.lower() not in 'aeıioöuü'

    def _is_vowel(self, char: str) -> bool:
        """Check if a character is a vowel"""
        return char and char.lower() in 'aeıioöuü'

    def check_vowel_harmony(self, stem: str, suffix: str) -> bool:
        """
        Check if suffix follows Turkish vowel harmony rules with the stem.

        Args:
            stem: The word stem
            suffix: The suffix to check

        Returns:
            True if vowel harmony is satisfied, False otherwise
        """
        if not stem or not suffix:
            return True

        # Find the last vowel in the stem
        stem_vowels = [c for c in stem.lower() if c in 'aeıioöuü']
        if not stem_vowels:
            return True

        last_stem_vowel = stem_vowels[-1]

        # Find the first vowel in the suffix
        suffix_vowels = [c for c in suffix.lower() if c in 'aeıioöuü']
        if not suffix_vowels:
            return True

        first_suffix_vowel = suffix_vowels[0]

        # Turkish vowel harmony rules
        # Front vowels: e, i, ö, ü
        # Back vowels: a, ı, o, u
        front_vowels = set('eiöü')
        back_vowels = set('aıou')

        # Rounded vowels: o, ö, u, ü
        # Unrounded vowels: a, e, ı, i
        rounded_vowels = set('oöuü')
        unrounded_vowels = set('aeıi')

        # 1. Front-Back Harmony (Major/Palatal Harmony)
        # Front vowels follow front vowels, back vowels follow back vowels
        if last_stem_vowel in front_vowels and first_suffix_vowel in back_vowels:
            return False
        if last_stem_vowel in back_vowels and first_suffix_vowel in front_vowels:
            return False

        # 2. Rounding Harmony (Minor/Labial Harmony) - for high vowels
        # This is more complex and has exceptions, but basic rule:
        # After rounded vowels, high vowels should be rounded (u, ü)
        # After unrounded vowels, high vowels should be unrounded (ı, i)
        high_vowels = set('ıiuü')

        if first_suffix_vowel in high_vowels:
            # Check rounding harmony for high vowels
            if last_stem_vowel in rounded_vowels:
                # Expect rounded high vowels (u, ü)
                if first_suffix_vowel not in set('uü'):
                    # This is often acceptable in Turkish, so we'll be lenient
                    pass  # Don't reject, as Turkish has many exceptions
            elif last_stem_vowel in unrounded_vowels:
                # Expect unrounded high vowels (ı, i)
                if first_suffix_vowel not in set('ıi'):
                    # This is often acceptable in Turkish, so we'll be lenient
                    pass  # Don't reject, as Turkish has many exceptions

        # 3. Special case for 'e' and 'a' in suffixes
        # These typically follow major harmony but have some flexibility
        if first_suffix_vowel in 'ea':
            # 'e' follows front vowels, 'a' follows back vowels
            if first_suffix_vowel == 'e' and last_stem_vowel in back_vowels:
                return False
            if first_suffix_vowel == 'a' and last_stem_vowel in front_vowels:
                return False

        return True

    def get_suffix_category(self, suffix: str) -> SuffixCategory:
        """Get the category of a suffix from the loaded categories"""
        return self.suffix_categories.get(suffix, SuffixCategory.INFLECTIONAL)

    def is_valid_suffix_order(self, removed_suffixes: List[str]) -> bool:
        """
        Check if the suffix removal order is morphologically valid.
        Suffixes should be removed in reverse order of attachment.

        Turkish morphological order (typical):
        ROOT → DERIVATIONAL → INFLECTIONAL → FUNCTIONAL

        When removing, we go: FUNCTIONAL → INFLECTIONAL → DERIVATIONAL
        """
        if len(removed_suffixes) <= 1:
            return True

        # Get categories of removed suffixes
        categories = [self.get_suffix_category(s) for s in removed_suffixes]

        # Generally, functional elements come after inflectional elements,
        # and inflectional elements come after derivational elements
        # So when removing, we should see them in reverse order
        # However, Turkish morphology is complex and there can be valid exceptions

        # Basic check: derivational suffixes shouldn't be removed before inflectional ones
        for i in range(1, len(categories)):
            # If we removed a derivational suffix before an inflectional/functional one,
            # it might be invalid (with some exceptions)
            if (categories[i - 1] == SuffixCategory.DERIVATIONAL and
                    categories[i] in [SuffixCategory.INFLECTIONAL, SuffixCategory.FUNCTIONAL]):
                # This could be invalid, but Turkish has exceptions
                # We'll be somewhat lenient here
                pass

        return True  # Be permissive due to Turkish morphological complexity

    def stem_with_backtracking(self, word: str, max_depth: int = 10) -> str:
        """
        Try multiple stemming paths using backtracking and return the best valid root.

        Args:
            word: Input word to stem
            max_depth: Maximum depth of suffix removal

        Returns:
            The most likely stem
        """
        if not word:
            return word

        word_lower = word.lower()

        # Use BFS to explore different stemming paths
        # Queue contains tuples of (current_form, removed_suffixes, depth)
        queue = deque([(word_lower, [], 0)])
        visited = set([word_lower])
        valid_stems = []

        while queue:
            current, removed_suffixes, depth = queue.popleft()

            # Check if we've reached maximum depth
            if depth >= max_depth:
                continue

            # Check if current form is a valid root
            if current in self.valid_words and len(current) >= self.min_stem_length:
                # Calculate a score for this stem
                score = self._score_stem(current, removed_suffixes)
                valid_stems.append((current, removed_suffixes, score))

            # Try removing each possible suffix
            for suffix in self.suffixes:
                if current.endswith(suffix):
                    stem_base = current[:-len(suffix)]

                    # Check minimum stem length
                    if len(stem_base) < self.min_stem_length:
                        continue

                    # Check vowel harmony
                    if not self.check_vowel_harmony(stem_base, suffix):
                        continue

                    # Generate phonological variants
                    variants = self.apply_phonological_rules(stem_base, suffix)

                    for variant in variants:
                        if variant not in visited and len(variant) >= self.min_stem_length:
                            # Check if the suffix order is valid
                            new_removed = removed_suffixes + [suffix]
                            if self.is_valid_suffix_order(new_removed):
                                visited.add(variant)
                                queue.append((variant, new_removed, depth + 1))

        # Return the best stem based on scoring
        if valid_stems:
            # Sort by score (higher is better)
            valid_stems.sort(key=lambda x: x[2], reverse=True)
            return valid_stems[0][0]

        # If no valid stem found, return the shortest form we could achieve
        if visited:
            shortest = min(visited, key=len)
            if len(shortest) < len(word_lower):
                return shortest

        return word_lower

    def _score_stem(self, stem: str, removed_suffixes: List[str]) -> float:
        """
        Score a potential stem based on various factors.

        Args:
            stem: The potential stem
            removed_suffixes: List of suffixes that were removed

        Returns:
            A score (higher is better)
        """
        score = 0.0

        # Prefer stems that are valid words
        if stem in self.valid_words:
            score += 10.0

        # Prefer longer stems (but not too long)
        optimal_length = 5
        length_diff = abs(len(stem) - optimal_length)
        score -= length_diff * 0.5

        # Prefer fewer suffix removals
        score -= len(removed_suffixes) * 0.2

        # Bonus for morphologically valid suffix ordering
        if self.is_valid_suffix_order(removed_suffixes):
            score += 2.0

        # Give different weights based on suffix categories
        for suffix in removed_suffixes:
            category = self.get_suffix_category(suffix)
            if category == SuffixCategory.FUNCTIONAL:
                score += 0.5  # Functional suffixes are easily removable
            elif category == SuffixCategory.INFLECTIONAL:
                score += 0.3  # Inflectional suffixes are common
            elif category == SuffixCategory.DERIVATIONAL:
                score += 0.1  # Derivational suffixes change meaning more

        # Penalty for very short stems
        if len(stem) < 3:
            score -= 5.0

        return score

    def stem(self, word: str) -> str:
        """
        Stem a Turkish word using the improved algorithm with backtracking.

        Args:
            word: Input word to stem

        Returns:
            Stemmed word (root)
        """
        return self.stem_with_backtracking(word)

    def stem_with_details(self, word: str) -> Dict:
        """
        Stem a word and return detailed analysis.

        Args:
            word: Input word to stem

        Returns:
            Dictionary containing stem and analysis details
        """
        if not word:
            return {
                'original': word,
                'stem': word,
                'removed_suffixes': [],
                'suffix_categories': [],
                'phonological_variants': [],
                'valid': False
            }

        word_lower = word.lower()

        # Use the backtracking algorithm to get all valid paths
        queue = deque([(word_lower, [], 0)])
        visited = set([word_lower])
        valid_stems = []
        all_paths = []

        while queue:
            current, removed_suffixes, depth = queue.popleft()

            if depth >= 10:  # Max depth
                continue

            # Record this path
            all_paths.append((current, removed_suffixes))

            # Check if valid
            if current in self.valid_words and len(current) >= self.min_stem_length:
                score = self._score_stem(current, removed_suffixes)
                suffix_cats = [self.get_suffix_category(s).name for s in removed_suffixes]
                valid_stems.append({
                    'stem': current,
                    'removed_suffixes': removed_suffixes,
                    'suffix_categories': suffix_cats,
                    'score': score,
                    'valid': True
                })

            # Try removing suffixes
            for suffix in self.suffixes:
                if current.endswith(suffix):
                    stem_base = current[:-len(suffix)]

                    if len(stem_base) < self.min_stem_length:
                        continue

                    if not self.check_vowel_harmony(stem_base, suffix):
                        continue

                    variants = self.apply_phonological_rules(stem_base, suffix)

                    for variant in variants:
                        if variant not in visited and len(variant) >= self.min_stem_length:
                            new_removed = removed_suffixes + [suffix]
                            if self.is_valid_suffix_order(new_removed):
                                visited.add(variant)
                                queue.append((variant, new_removed, depth + 1))

        # Get the best stem
        if valid_stems:
            valid_stems.sort(key=lambda x: x['score'], reverse=True)
            best = valid_stems[0]

            return {
                'original': word,
                'stem': best['stem'],
                'removed_suffixes': best['removed_suffixes'],
                'suffix_categories': best['suffix_categories'],
                'score': best['score'],
                'valid': True,
                'alternative_stems': valid_stems[1:3] if len(valid_stems) > 1 else [],
                'total_paths_explored': len(all_paths)
            }
        else:
            # Return the shortest form found
            shortest = min(visited, key=len) if visited else word_lower
            return {
                'original': word,
                'stem': shortest,
                'removed_suffixes': [],
                'suffix_categories': [],
                'score': 0.0,
                'valid': shortest in self.valid_words,
                'alternative_stems': [],
                'total_paths_explored': len(all_paths)
            }

    def get_category_statistics(self) -> Dict:
        """
        Get statistics about loaded suffix categories.

        Returns:
            Dictionary with category statistics
        """
        stats = {
            'total_suffixes': len(self.suffixes),
            'categories': {}
        }

        for category in SuffixCategory:
            count = len(self.categorized_suffixes[category])
            stats['categories'][category.name] = {
                'count': count,
                'percentage': (count / len(self.suffixes) * 100) if self.suffixes else 0,
                'examples': self.categorized_suffixes[category][:5]  # First 5 examples
            }

        return stats


# Example usage and testing
if __name__ == "__main__":
    import os

    # Check if the CSV file exists
    csv_file = '../data/suffix_categories.csv'
    words_file = '../data/Turkish_Corpus_3M.txt'  # You can provide this if available

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        csv_file = None

    # Initialize the stemmer with the CSV file
    stemmer = ImprovedTurkishStemmer(
        suffix_categories_file=csv_file,
        words_file=words_file if os.path.exists(words_file) else None,
        min_stem_length=2
    )

    # Show category statistics
    print("\n" + "=" * 80)
    print("SUFFIX CATEGORY STATISTICS")
    print("=" * 80)
    stats = stemmer.get_category_statistics()
    print(f"Total suffixes loaded: {stats['total_suffixes']}")
    for cat_name, cat_stats in stats['categories'].items():
        print(f"\n{cat_name}:")
        print(f"  Count: {cat_stats['count']} ({cat_stats['percentage']:.1f}%)")
        if cat_stats['examples']:
            print(f"  Examples: {', '.join(cat_stats['examples'])}")

    # Test with some Turkish words
    test_words = [
        "kitaplarımızdan",  # book-PLUR-POSS.1PL-ABL
        "evlerimizde",  # house-PLUR-POSS.1PL-LOC
        "çocuklarının",  # child-PLUR-POSS.3PL-GEN
        "öğrencilerden",  # student-PLUR-ABL
        "gelebileceksiniz",  # to be able to come-FUT-2PL
        "yapmaktayız",  # we are doing
        "güzelleştirmek",  # to beautify
        "arkadaşlarımla",  # with my friends
        "kitabın"
    ]

    # Create a mock word list for testing
    stemmer.valid_words = {
        'kitap', 'ev', 'çocuk', 'öğrenci', 'gel', 'yap',
        'güzel', 'arkadaş', 'masa', 'araba', 'okul',"kitap"
    }

    print("\n" + "=" * 80)
    print("STEMMING TEST WITH CSV CATEGORIES")
    print("=" * 80)

    for word in test_words:
        print(f"\nWord: {word}")
        print("-" * 40)

        # Get detailed analysis
        result = stemmer.stem_with_details(word)

        print(f"Stem: {result['stem']}")
        print(f"Valid: {'✓' if result['valid'] else '✗'}")

        if result['removed_suffixes']:
            print(f"Removed suffixes:")
            for suffix, category in zip(result['removed_suffixes'], result['suffix_categories']):
                print(f"  - {suffix} ({category})")
        else:
            print(f"Removed suffixes: None")

        print(f"Score: {result['score']:.2f}")
        print(f"Paths explored: {result['total_paths_explored']}")

        if result.get('alternative_stems'):
            print(f"Alternatives:")
            for alt in result['alternative_stems']:
                print(f"  - {alt['stem']} (score: {alt['score']:.2f})")
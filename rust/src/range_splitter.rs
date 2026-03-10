use std::collections::{HashMap, HashSet};

#[derive(Debug, PartialEq, Clone)]
pub struct MinimalIntRange {
    pub start_int: num_bigint::BigInt,
    pub end_int: num_bigint::BigInt,
    pub min_len: usize,
}

pub struct RangeSplitter {
    alphabet_map: HashMap<char, usize>,
    sorted_alphabet: Vec<char>,
    alphabet_set: HashSet<char>,
}

impl RangeSplitter {
    pub fn new(alphabet: &str) -> Self {
        let mut sorted_alphabet: Vec<char> = alphabet.chars().collect();
        sorted_alphabet.sort_unstable();
        sorted_alphabet.dedup(); // Ensure uniqueness
        
        let alphabet_set: HashSet<char> = sorted_alphabet.iter().cloned().collect();
        let alphabet_map: HashMap<char, usize> = sorted_alphabet
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        Self {
            alphabet_map,
            sorted_alphabet,
            alphabet_set,
        }
    }

    pub fn split_range(&mut self, start_range: &str, end_range: &str, num_splits: usize) -> Vec<String> {
        if num_splits < 1 {
            return vec![];
        }
        
        // If end_range is provided and start >= end, invalid range.
        if !end_range.is_empty() && start_range >= end_range {
            return vec![];
        }

        if self.is_range_equal_with_padding(start_range, end_range) {
            return vec![];
        }

        let combined = format!("{}{}", start_range, end_range);
        self.add_characters_to_alphabet(&combined);

        let min_int_range = self.string_to_minimal_int_range(start_range, end_range, num_splits);

        self.generate_splits(&min_int_range, num_splits, start_range, end_range)
    }

    fn generate_splits(
        &self,
        min_int_range: &MinimalIntRange,
        num_splits: usize,
        start_range: &str,
        end_range: &str,
    ) -> Vec<String> {
        let mut split_points = Vec::new();
        let range_diff = &min_int_range.end_int - &min_int_range.start_int;
        let range_interval = num_splits + 1;
        
        use num_bigint::BigInt;
        
        let interval_bg = BigInt::from(range_interval);
        let range_diff = &min_int_range.end_int - &min_int_range.start_int;

        for i in 1..=num_splits {
            // Equivalent to start_int + int(Fraction(range_diff, interval) * i)
            // We multiply first, then divide, so we don't lose precision until the end.
            let split_point = &min_int_range.start_int + (&range_diff * BigInt::from(i)) / &interval_bg;

            let split_string = self.int_to_string(&split_point, min_int_range.min_len);

            let is_greater_than_start = !split_string.is_empty() && split_string.as_str() > start_range;
            let is_less_than_end = end_range.is_empty() || (!split_string.is_empty() && split_string.as_str() < end_range);

            if is_greater_than_start && is_less_than_end {
                split_points.push(split_string);
            }
        }

        split_points
    }

    fn int_to_string(&self, split_point: &num_bigint::BigInt, string_len: usize) -> String {
        use num_traits::cast::ToPrimitive;
        let alphabet_len = num_bigint::BigInt::from(self.sorted_alphabet.len());
        let mut current_point = split_point.clone();
        let mut split_string = String::with_capacity(string_len);

        for _ in 0..string_len {
            let remainder = (&current_point % &alphabet_len).to_usize().unwrap_or(0);
            current_point /= &alphabet_len;
            split_string.push(self.sorted_alphabet[remainder]);
        }

        // Python assembles backwards: `split_string[::-1]`
        split_string.chars().rev().collect()
    }

    fn string_to_minimal_int_range(&self, start_range: &str, end_range: &str, num_splits: usize) -> MinimalIntRange {
        use num_bigint::BigInt;
        let mut start_int = BigInt::from(0);
        let mut end_int = BigInt::from(0);

        let alphabet_len = BigInt::from(self.sorted_alphabet.len());
        let start_char = self.sorted_alphabet[0];
        let end_char = *self.sorted_alphabet.last().unwrap();

        let end_default_char = if end_range.is_empty() { end_char } else { start_char };

        for i in 0.. {
            let start_c = get_char_or_default(start_range, i, start_char);
            let start_pos = *self.alphabet_map.get(&start_c).unwrap_or(&0);
            start_int *= &alphabet_len;
            start_int += BigInt::from(start_pos);

            let end_c = get_char_or_default(end_range, i, end_default_char);
            let end_pos = *self.alphabet_map.get(&end_c).unwrap_or(&0);
            end_int *= &alphabet_len;
            end_int += BigInt::from(end_pos);

            let difference = &end_int - &start_int;
            if difference > BigInt::from(num_splits) {
                return MinimalIntRange {
                    start_int,
                    end_int,
                    min_len: i + 1,
                };
            }
        }
        unreachable!()
    }

    fn is_range_equal_with_padding(&self, start_range: &str, end_range: &str) -> bool {
        if end_range.is_empty() {
            return false;
        }

        let longest = std::cmp::max(start_range.len(), end_range.len());
        let smallest_char = self.sorted_alphabet[0];

        for i in 0..longest {
            let char_start = get_char_or_default(start_range, i, smallest_char);
            let char_end = get_char_or_default(end_range, i, smallest_char);

            if char_start != char_end {
                return false;
            }
        }

        true
    }

    fn add_characters_to_alphabet(&mut self, characters: &str) {
        let mut new_chars = false;
        for c in characters.chars() {
            if self.alphabet_set.insert(c) {
                new_chars = true;
            }
        }

        if new_chars {
            self.sorted_alphabet = self.alphabet_set.iter().cloned().collect();
            self.sorted_alphabet.sort_unstable();
            self.alphabet_map = self.sorted_alphabet
                .iter()
                .enumerate()
                .map(|(i, &c)| (c, i))
                .collect();
        }
    }
}

fn get_char_or_default(characters: &str, index: usize, default_char: char) -> char {
    characters.chars().nth(index).unwrap_or(default_char)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_range_simple() {
        let mut splitter = RangeSplitter::new("ab");
        let splits = splitter.split_range("a", "b", 1);
        assert_eq!(splits, vec!["ab"]);
    }

    #[test]
    fn test_split_range_alphabet_expansion() {
        let mut splitter = RangeSplitter::new("ab");
        // 'c' and 'd' are added to the alphabet dynamically
        let splits = splitter.split_range("a", "d", 1);
        assert_eq!(splits, vec!["b"]);
    }

    #[test]
    fn test_split_range_unbounded_end() {
        let mut splitter = RangeSplitter::new("ab");
        // No end range means it goes to infinity essentially
        let splits = splitter.split_range("a", "", 1);
        assert_eq!(splits, vec!["ab"]);
    }
}

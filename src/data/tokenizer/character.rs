use super::Tokenizer;
use std::collections::HashMap;

pub struct CharTokenizer {
    char_to_id: HashMap<char, usize>,
    id_to_char: HashMap<usize, char>,
    vocab_size: usize,
}

impl Default for CharTokenizer {
    fn default() -> Self {
        // Create basic vocabulary from ASCII printable characters
        let mut chars: Vec<char> = (32..127).map(|i| i as u8 as char).collect();
        // Add newline
        chars.push('\n');

        // Create mappings
        let char_to_id: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        let id_to_char: HashMap<usize, char> =
            chars.iter().enumerate().map(|(i, &c)| (i, c)).collect();

        // Add special tokens at the end
        let vocab_size = chars.len() + 3; // +3 for [START], [END], [PAD]

        Self {
            char_to_id,
            id_to_char,
            vocab_size,
        }
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize> {
        let mut tokens = Vec::new();

        if special_tokens {
            tokens.push(self.start_token());
        }

        // Convert each character to its token ID
        tokens.extend(
            value
                .chars()
                .map(|c| *self.char_to_id.get(&c).unwrap_or(&self.pad_token())),
        );

        if special_tokens {
            tokens.push(self.end_token());
        }

        tokens
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|&id| {
                // Skip special tokens in output
                if id >= self.vocab_size - 3 {
                    None
                } else {
                    self.id_to_char.get(&id).copied()
                }
            })
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn pad_token(&self) -> usize {
        self.vocab_size - 1
    }

    fn start_token(&self) -> usize {
        self.vocab_size - 3
    }

    fn end_token(&self) -> usize {
        self.vocab_size - 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_tokenizer() {
        let tokenizer = CharTokenizer::default();

        // Test basic encoding/decoding
        let text = "Hello, World!";
        let tokens = tokenizer.encode(text, false);
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(text, decoded);

        // Test with special tokens
        let tokens = tokenizer.encode(text, true);
        assert_eq!(tokens[0], tokenizer.start_token());
        assert_eq!(tokens[tokens.len() - 1], tokenizer.end_token());
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(text, decoded);

        // Test vocab size
        assert!(tokenizer.vocab_size() > 0);
    }
}

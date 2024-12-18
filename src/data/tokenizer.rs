// data/tokenizer.rs
use tokenizers::Tokenizer as HfTokenizer;
mod character;
pub use character::CharTokenizer;

#[allow(dead_code)]
pub trait Tokenizer: Send + Sync {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
    fn pad_token(&self) -> usize;
    fn start_token(&self) -> usize;
    fn end_token(&self) -> usize;
}

pub struct GPTTokenizer {
    tokenizer: HfTokenizer,
}

impl Default for GPTTokenizer {
    fn default() -> Self {
        let mut tokenizer = HfTokenizer::from_pretrained("gpt2", None).unwrap();
        tokenizer.add_special_tokens(&[
            tokenizers::AddedToken::from("[START]", true),
            tokenizers::AddedToken::from("[END]", true),
            tokenizers::AddedToken::from("[PAD]", true),
        ]);

        Self { tokenizer }
    }
}

impl Tokenizer for GPTTokenizer {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize> {
        let text = match special_tokens {
            true => "[START]".to_owned() + value + "[END]",
            false => value.to_string(),
        };
        let tokens = self.tokenizer.encode(text, true).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>();
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn pad_token(&self) -> usize {
        self.tokenizer.token_to_id("[PAD]").unwrap() as usize
    }

    fn start_token(&self) -> usize {
        self.tokenizer.token_to_id("[START]").unwrap() as usize
    }

    fn end_token(&self) -> usize {
        self.tokenizer.token_to_id("[END]").unwrap() as usize
    }
}

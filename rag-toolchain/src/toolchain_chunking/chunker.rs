use tiktoken_rs::cl100k_base;
use tiktoken_rs::CoreBPE;

fn generate_chunks(raw_text: &str, window_size: usize, chunk_size: usize) -> Vec<String> {

  let bpe = cl100k_base().unwrap();
  get_amount_of_tokens(raw_text, &bpe);

  return Vec::new();
}

fn get_amount_of_tokens(text: &str, bpe: &CoreBPE) -> usize {
  let tokens = bpe.split_by_token_iter(text, true);
  println!("{:?}", tokens.map(|x| x.unwrap()).collect::<Vec<_>>());
  return 0;
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test() {
    let raw_text = "This is a test, on how to split text";
    let window_size = 5;
    let chunk_size = 3;
    let _chunks = generate_chunks(raw_text, window_size, chunk_size);
  }
}
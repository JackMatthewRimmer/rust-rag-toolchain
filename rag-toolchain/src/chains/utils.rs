use crate::{clients::PromptMessage, common::Chunks};

/// There are a number of utility functions that are used in the chains module.
/// as the number of chains grows we will see specifc patterns emerge and these
/// will be refactored into this module. Any function that is not taking a reference
/// to &self should be placed here.

/// # [`build_prompt`]
///
/// function to builder the user prompt from the original user prompt and the retrieved
/// supporting chunks.
///
/// # Arguments
/// * `base_message` - the original user prompt
/// * `chunks` - the supporting chunks retrieved from the retriever
///
/// # Returns
/// [`PromptMessage`] - the new user prompt
pub fn build_prompt(base_message: &PromptMessage, chunks: Chunks) -> PromptMessage {
    let mut builder: String = String::new();
    builder.push_str(base_message.content());
    builder.push_str("\nHere is some supporting information:\n");
    for chunk in chunks {
        builder.push_str(&format!("{}\n", chunk.content()))
    }
    PromptMessage::HumanMessage(builder)
}

#[cfg(test)]
mod chains_utils_tests {

    use super::*;
    use crate::common::Chunk;

    #[test]
    fn build_prompt_gives_correct_output() {
        const USER_MESSAGE: &str = "can you explain the data to me";
        let user_prompt: PromptMessage = PromptMessage::HumanMessage(USER_MESSAGE.into());
        let chunks = vec![Chunk::new("data point 1"), Chunk::new("data point 2")];
        let response = build_prompt(&user_prompt, chunks);
        let expected_response: &str = "can you explain the data to me\nHere is some supporting information:\ndata point 1\ndata point 2\n";
        println!("{}", expected_response);
        matches!(response, PromptMessage::HumanMessage(_));
        assert_eq!(expected_response, response.content());
    }
}

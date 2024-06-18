/// # [`PromptMessage`]
/// This enum is used to represent the different types of messages that can be sent to the LLM.
/// we will map the PromptMessage within the client into the compatible format.
/// * [`PromptMessage::SystemMessage`] - This is a message that typically we asign the model a role.
/// * [`PromptMessage::HumanMessage`] - This is a message that is from a human i.e you.
/// * [`PromptMessage::AIMessage`] - This is a message that we get back from the LLM.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum PromptMessage {
    SystemMessage(String),
    HumanMessage(String),
    AIMessage(String),
}

impl PromptMessage {
    /// # [`PromptMessage::content`]
    ///
    /// Given that the clients will return a message that we only care for the message
    /// this function will return the message as a string to avoid pattern matching.
    ///
    /// # Returns
    /// * &[`str`] - the message content
    pub fn content(&self) -> &str {
        match self {
            PromptMessage::SystemMessage(message) => message,
            PromptMessage::HumanMessage(message) => message,
            PromptMessage::AIMessage(message) => message,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_message_content() {
        let test_string = String::from("Test String");
        assert_eq!(
            &test_string,
            PromptMessage::HumanMessage(test_string.clone()).content()
        );
        assert_eq!(
            &test_string,
            PromptMessage::AIMessage(test_string.clone()).content()
        );
        assert_eq!(
            &test_string,
            PromptMessage::SystemMessage(test_string.clone()).content()
        );
    }
}

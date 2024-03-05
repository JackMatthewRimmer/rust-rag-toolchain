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
    /// * [`String`] - the message content
    pub fn content(&self) -> String {
        match self {
            PromptMessage::SystemMessage(message) => message.clone(),
            PromptMessage::HumanMessage(message) => message.clone(),
            PromptMessage::AIMessage(message) => message.clone(),
        }
    }
}

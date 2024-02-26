/// # [`PromptMessage`] 
/// This enum is used to represent the different types of messages that can be sent to the LLM.
/// we will map the PromptMessage within the client into the compatible format.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum PromptMessage {
    SystemMessage(String),
    HumanMessage(String),
    AIMessage(String),
}

impl PromptMessage {
    /// # content
    ///
    /// Given that the clients will return a message that we only care for the message
    /// this function will return the message as a string to avoid pattern matching.
    pub fn content(&self) -> String {
        match self {
            PromptMessage::SystemMessage(message) => message.clone(),
            PromptMessage::HumanMessage(message) => message.clone(),
            PromptMessage::AIMessage(message) => message.clone(),
        }
    }
}

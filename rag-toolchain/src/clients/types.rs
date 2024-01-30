/// # PromptMessage
/// This enum is used to represent the different types of messages that can be sent to the LLM.
/// we will map the PromptMessage within the client into the compatible format.
pub enum PromptMessage {
    SystemMessage(String),
    HumanMessage(String),
    AIMessage(String),
}

impl PromptMessage {
    pub fn content(&self) -> String {
        match self {
            PromptMessage::SystemMessage(message) => message.clone(),
            PromptMessage::HumanMessage(message) => message.clone(),
            PromptMessage::AIMessage(message) => message.clone(),
        }
    }
}

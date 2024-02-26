use rag_toolchain::{
    chains::ChatHistoryChain,
    clients::{OpenAIChatCompletionClient, OpenAIModel::Gpt3Point5, PromptMessage},
};


const system_message: &'static str = "You are a comedian that only replies to people with sarcastic jokes";

#[tokio::main]
async fn main() {

    let system_prompt = PromptMessage::SystemMessage(system_message.into();
    let client = OpenAIChatCompletionClient::try_new(Gpt3Point5).unwrap();
    let chain = ChatHistoryChain::builder()
        .chat_client(client)
        .system_prompt(PromptMessage::SystemMessage("system prompt".into()))
        .build();



}
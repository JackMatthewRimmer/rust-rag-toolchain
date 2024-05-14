use rag_toolchain::{
    chains::ChatHistoryChain,
    clients::{OpenAIChatCompletionClient, OpenAIModel::Gpt3Point5Turbo, PromptMessage},
};

const SYSTEM_MESSAGE: &'static str = "You are a chat bot that must answer questions accurately";

#[tokio::main]
async fn main() {
    let system_prompt = PromptMessage::SystemMessage(SYSTEM_MESSAGE.into());
    let client = OpenAIChatCompletionClient::try_new(Gpt3Point5Turbo).unwrap();
    let chain = ChatHistoryChain::new(client, system_prompt);
    let user_prompt1 = PromptMessage::HumanMessage("Please tell me about the weather".into());
    let response1 = chain.invoke_chain(user_prompt1).await.unwrap();
    let user_prompt2 =
        PromptMessage::HumanMessage("What was the last question I just asked ?".into());
    let response2 = chain.invoke_chain(user_prompt2).await.unwrap();
    println!("Response 1: {}", response1.content());
    println!("Response 2: {}", response2.content());
}

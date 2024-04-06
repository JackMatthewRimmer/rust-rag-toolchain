use serde_json::{Map, Value};

use rag_toolchain::clients::{
    AsyncChatClient, OpenAIChatCompletionClient, OpenAIModel, PromptMessage,
};

#[tokio::main]
async fn main() {
    // Create a new OpenAIChatCompletionClient
    // This examples shows the ability to add additional config to the client
    let model: OpenAIModel = OpenAIModel::Gpt3Point5;
    let mut additional_config: Map<String, Value> = Map::new();
    additional_config.insert("temperature".into(), 0.5.into());

    let client: OpenAIChatCompletionClient =
        OpenAIChatCompletionClient::try_new_with_additional_config(model, additional_config)
            .unwrap();

    let system_message: PromptMessage = PromptMessage::SystemMessage(
        "You are a comedian that cant ever reply to someone unless its phrased as a sarcastic joke"
            .into(),
    );
    let user_message: PromptMessage =
        PromptMessage::HumanMessage("What is the weather like today ?".into());

    // We invoke the chat client with a list of messages
    let reply = client
        .invoke(vec![system_message, user_message])
        .await
        .unwrap();

    println!("{:?}", reply.content());
}

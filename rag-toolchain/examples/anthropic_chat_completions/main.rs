use serde_json::{Map, Value};

use rag_toolchain::clients::{
    AnthropicChatCompletionClient, AnthropicModel, AsyncChatClient, PromptMessage,
};

#[tokio::main]
async fn main() {
    let model: AnthropicModel = AnthropicModel::Claude3Sonnet;
    let mut additional_config: Map<String, Value> = Map::new();
    additional_config.insert("temperature".into(), 0.5.into());

    let client: AnthropicChatCompletionClient =
        AnthropicChatCompletionClient::try_new_with_additional_config(
            model,
            4096,
            additional_config,
        )
        .unwrap();

    let system_message: PromptMessage =
        PromptMessage::SystemMessage("You only reply in a bullet point list".into());
    let user_message: PromptMessage = PromptMessage::HumanMessage("How does the water flow".into());

    // We invoke the chat client with a list of messages
    let reply = client
        .invoke(vec![system_message.clone(), user_message.clone()])
        .await
        .unwrap();

    println!("{:?}", reply.content());
}

use serde_json::{Map, Value};

use rag_toolchain::clients::{
    AsyncChatClient, OpenAIChatCompletionClient, OpenAIModel, PromptMessage,
};

#[tokio::main]
async fn main() {
    let model: OpenAIModel = OpenAIModel::Gpt3Point5;
    let mut client: OpenAIChatCompletionClient =
        OpenAIChatCompletionClient::try_new(model).unwrap();
    let mut additional_config: Map<String, Value> = Map::new();
    additional_config.insert("temperature".into(), 0.5.into());
    client.with_additional_config(additional_config);

    let system_message: PromptMessage = PromptMessage::SystemMessage(
        "You are a comedian that cant ever reply to someone unless its phrased as a sarcastic joke"
            .into(),
    );
    let user_message: PromptMessage =
        PromptMessage::HumanMessage("What is the weather like today ?".into());

    let reply = client
        .invoke(vec![system_message, user_message])
        .await
        .unwrap();

    println!("{:?}", reply.content());
}

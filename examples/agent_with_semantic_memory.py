"""
Semantic long-term memory example.

Facts are stored with embeddings; relevant ones are retrieved by meaning.
Requires: pip install 'kognios[vector]' and OPENAI_API_KEY.
"""

from kognios import Agent, AnthropicModel, LongTermMemory
import openai


def make_embed_fn():
    client = openai.OpenAI()

    def embed(texts):
        r = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [item.embedding for item in r.data]

    return embed


def main() -> None:
    long = LongTermMemory(db_path="semantic_demo.db", embed_fn=make_embed_fn())

    # Store facts
    long.store("project", "Building a from-scratch Python agent framework called Kogni-OS")
    long.store("language", "Python 3.11+")
    long.store("goal", "Keep the framework lean, under 1000 lines of core code")
    long.store("license", "MIT")

    # Semantic recall
    relevant = long.semantic_recall("What is the user working on?", top_k=2)
    print("Relevant facts:", relevant)


if __name__ == "__main__":
    main()

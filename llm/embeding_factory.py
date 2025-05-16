from langchain_community.embeddings import FakeEmbeddings


def get_embedding_model():
    return FakeEmbeddings(size=1352)
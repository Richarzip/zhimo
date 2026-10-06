from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from .data import CALLIGRAPHER_KNOWLEDGE
from ..config import get_settings


# 本地 embedding 模型（完全离线，不需要 API Key）
# 如果你用的是 bge-m3，改成 model="bge-m3"
def _embeddings():
    return OllamaEmbeddings(model="bge-m3")

def build_vectorstore():
    """构建/加载向量库"""
    return Chroma(
        collection_name="calligrapher_knowledge",
        embedding_function=_embeddings(),
        persist_directory=str(get_settings().chroma_dir),
    )


def init_knowledge(vectorstore):
    """把知识写入向量库（避免重复写入）"""
    existing = vectorstore.get()
    if existing and len(existing["ids"]) > 0:
        print(f"向量库已存在 {len(existing['ids'])} 条记录，跳过写入。")
        return

    texts = [item["text"] for item in CALLIGRAPHER_KNOWLEDGE]
    metadatas = [{"name": item["name"]} for item in CALLIGRAPHER_KNOWLEDGE]

    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    print(f"已写入 {len(texts)} 位书法家的知识。")


if __name__ == "__main__":
    vs = build_vectorstore()
    init_knowledge(vs)

    # 测试检索
    print("\n=== 测试检索：'褚遂良的风格特点' ===")
    results = vs.similarity_search("褚遂良的风格特点", k=2)
    for i, doc in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(f"书法家：{doc.metadata['name']}")
        print(f"内容：{doc.page_content[:100]}...")

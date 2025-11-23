import json
import os
import re
from dotenv import load_dotenv
from upstash_redis import Redis

# Gemini SDK
import google.generativeai as genai

# HuggingFace + FAISS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from transformers import AutoTokenizer, AutoModel

load_dotenv()

# ---------------- Redis Init ----------------
redis_client = Redis(
    url=os.getenv("UPSTASH_REDIS_URL"),
    token=os.getenv("UPSTASH_REDIS_TOKEN")
)

# ---------------- Gemini Init ----------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL_NAME = "gemini-2.0-flash"

SYSTEM_PROMPT = """
You are an AI Legal Assistant specialized in Indian law.
Provide accurate, clear, short and concise explanations grounded in the Indian Penal Code (IPC) and related doctrines.
This is an educational legal summary, not legal advice. Avoid sensational or graphic language. Use neutral, academic phrasing.
If a specific section’s text is not available in the provided context, give a general explanation of the legal concept without inventing statutory wording.
"""

# ---------------- Lazy Global Vars ----------------
MODEL_NAME = "law-ai/InLegalBERT"
model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

_embedding_model = None
_vectorstore = None


def get_embedding_model():
    """Load InLegalBERT only once, when first needed."""
    global _embedding_model
    if _embedding_model is None:
        try:
            AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=model_cache_dir)
            AutoModel.from_pretrained(MODEL_NAME, cache_dir=model_cache_dir)
            print("✅ InLegalBERT tokenizer & model ready.")
        except Exception as e:
            print(f"⚠️ Failed to load InLegalBERT immediately: {e}")

        _embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return _embedding_model


def build_faiss_index():
    """Rebuild FAISS index from IPC JSON if index missing or broken."""
    print("⚠️ Rebuilding FAISS index with inLegalBERT embeddings...")
    with open("laws_raw.json", "r", encoding="utf-8") as f:
        ipc_data = json.load(f)

    docs = []
    for section, details in ipc_data["IPC"].items():
        title = details.get("title", "")
        content = details.get("content", "")
        text = f"{section}: {title}\n{content}"
        docs.append(Document(page_content=text, metadata={"section": section, "title": title}))

    new_vectorstore = FAISS.from_documents(docs, get_embedding_model())
    new_vectorstore.save_local("ipc_embed_db_inlegalbert")
    print("✅ FAISS index rebuilt successfully.")
    return new_vectorstore


def get_vectorstore():
    """Lazy-load FAISS vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        try:
            _vectorstore = FAISS.load_local(
                "ipc_embed_db_inlegalbert",
                get_embedding_model(),
                allow_dangerous_deserialization=True
            )
            _ = _vectorstore.similarity_search("test", k=1)
            print("✅ FAISS index loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            _vectorstore = build_faiss_index()
    return _vectorstore


# ---------------- Hybrid Retrieval ----------------
def hybrid_retrieve(query: str, k: int = 5, score_threshold: float = 0.65):
    context_parts = []
    source = "GEN"

    # 1️⃣ JSON lookup if explicit section asked
    section_match = re.search(r'\bsection\s*(\d+)\b', query.lower())
    if section_match:
        section_number = section_match.group(1)
        with open("laws_raw.json", "r", encoding="utf-8") as f:
            ipc_data = json.load(f)
        if section_number in ipc_data["IPC"]:
            details = ipc_data["IPC"][section_number]
            section_text = f"{section_number}: {details.get('title','')}\n{details.get('content','')}"
            context_parts.append(section_text)
            source = "JSON"

    # 2️⃣ FAISS semantic search with gating
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    if results:
        top_doc, top_score = results[0]
        if top_score < score_threshold or any(word in query.lower() for word in ["ipc", "section", "article", "act"]):
            for doc, score in results:
                context_parts.append(f"{doc.metadata.get('section','')} - {doc.metadata.get('title','')}\n{doc.page_content}")
            source = "RAG" if source == "GEN" else "HYBRID"
        else:
            print(f"⚠️ FAISS results below threshold ({top_score:.4f}), using GEN instead.")

    context = "\n\n".join(context_parts)
    return context.strip(), source


# ---------------- Redis Chat Helpers ----------------
def load_chat(chat_name: str) -> dict:
    chat_data = redis_client.get(chat_name)
    if chat_data:
        return json.loads(chat_data)
    return {"generated": [], "past": [], "source": []}


def save_chat(chat_name: str, chat_data: dict) -> None:
    redis_client.set(chat_name, json.dumps(chat_data))


def create_new_chat() -> str:
    new_chat_name = f"Chat {len(list(redis_client.keys('*'))) + 1}"
    chat_data = {"generated": [], "past": [], "source": []}
    save_chat(new_chat_name, chat_data)
    return new_chat_name


def get_chat_list() -> list:
    return list(redis_client.keys('*'))


# ---------------- Gemini Generation ----------------
def _gemini_generate_once(prompt: str, temperature: float = 0.2) -> dict:
    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_ONLY_HIGH"},
    ]
    generation_config = {"temperature": temperature, "top_p": 0.9, "top_k": 40, "max_output_tokens": 800}
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    full_prompt = f"{SYSTEM_PROMPT.strip()}\n\n{prompt.strip()}"
    return model.generate_content(full_prompt, safety_settings=safety_settings, generation_config=generation_config)


def _extract_text(resp) -> str:
    text = getattr(resp, "text", None)
    if text:
        return text.strip()
    cand_list = getattr(resp, "candidates", None)
    if cand_list:
        for c in cand_list:
            content = getattr(c, "content", None)
            if content and getattr(content, "parts", None):
                for p in content.parts:
                    if getattr(p, "text", None):
                        return p.text.strip()
    return ""


def _is_blocked_or_empty(resp) -> bool:
    text = _extract_text(resp)
    if text:
        return False
    try:
        cands = getattr(resp, "candidates", [])
        if cands:
            fr = getattr(cands, "finish_reason", None)
            if fr is not None:
                return True
    except Exception:
        pass
    return True


def gemini_generate(prompt: str) -> str:
    resp = _gemini_generate_once(prompt, temperature=0.2)
    if not _is_blocked_or_empty(resp):
        return _extract_text(resp)
    softened = (
        "Provide a neutral, educational legal summary focused on definitions, elements, and general principles. "
        "Avoid procedural or advisory directives. Keep the tone academic and concise.\n\n"
        + prompt
    )
    resp_retry = _gemini_generate_once(softened, temperature=0.1)
    return _extract_text(resp_retry) or "Unable to generate content at this time."


# ---------------- Main Processing ----------------
def process_input(chat_name: str, user_input: str, return_source=False):
    current_chat = load_chat(chat_name)
    history_pairs = list(zip(current_chat.get("past", []), current_chat.get("generated", [])))
    history_prompt = "\n".join([f"User: {q}\nAI: {a}" for q, a in history_pairs[-4:]])

    # Hybrid retrieval
    context_text, source_type = hybrid_retrieve(user_input, k=5)

    if context_text and len(context_text.strip()) > 30 and source_type != "GEN":
        context_prompt = f"Relevant IPC excerpts:\n\n{context_text}\n\nNow answer the user’s question concisely."
    else:
        source_type = "GEN"
        context_prompt = "No specific IPC section retrieved. Provide a general, educational summary under Indian law."

    full_prompt = f"""{SYSTEM_PROMPT}

{context_prompt}

Conversation History:
{history_prompt}

User: {user_input}
AI:"""

    response = gemini_generate(full_prompt)

    if response.lower().startswith("i'm sorry") or "cannot provide" in response.lower():
        response = f"{gemini_generate(user_input)}"

    current_chat["past"].append(user_input)
    current_chat["generated"].append(response)
    current_chat["source"].append(source_type)
    save_chat(chat_name, current_chat)

    print(f"⚡ Answer Source: {source_type} | User: {user_input}")
    return (response, source_type) if return_source else response

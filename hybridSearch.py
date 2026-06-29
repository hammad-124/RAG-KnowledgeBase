"""Hybrid Search Demo — Vector + BM25 for RAG"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import HybridSearchRetriever
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()  # Loads OPENAI_API_KEY from .env

# ─── Sample Dataset ───────────────────────────────────────────────────────────
# Each Document has content + metadata for filtering/tracking.
documents = [
    # --- Products (SKUs are meaningless to embeddings) ---
    Document(
        page_content="Product SKU-7742X is our flagship dual-band router. It supports gigabit speeds, advanced QoS, and up to 120 connected devices.",
        metadata={"type": "product", "category": "router", "sku": "SKU-7742X"}
    ),
    Document(
        page_content="Product SKU-1188M is a mesh Wi-Fi extender designed for large homes and offices. It improves wireless coverage in dead zones.",
        metadata={"type": "product", "category": "mesh", "sku": "SKU-1188M"}
    ),
    Document(
        page_content="Product SKU-5521F is a fiber modem compatible with most ISP networks. It supports high-speed fiber internet connections.",
        metadata={"type": "product", "category": "modem", "sku": "SKU-5521F"}
    ),

    # --- Troubleshooting guides ---
    Document(
        page_content="For network connectivity issues, first check the ethernet cable, router power light, WAN indicator, and internet status LED.",
        metadata={"type": "troubleshooting", "category": "network"}
    ),
    Document(
        page_content="If the Wi-Fi signal is weak, place the router in an open central location and avoid walls, metal objects, and microwave interference.",
        metadata={"type": "troubleshooting", "category": "wifi"}
    ),
    Document(
        page_content="If the router keeps restarting, check the power adapter, overheating issues, firmware version, and factory reset settings.",
        metadata={"type": "troubleshooting", "category": "router"}
    ),

    # --- Error codes (meaningless to embeddings) ---
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall rules, port settings, and backend service status.",
        metadata={"type": "error", "code": "E_CONN_REFUSED"}
    ),
    Document(
        page_content="Error code AUTH_401 means the request is unauthorized. The user must provide valid login credentials or a valid access token.",
        metadata={"type": "error", "code": "AUTH_401"}
    ),
    Document(
        page_content="Error code DNS_404 means the domain name could not be resolved. Verify DNS records, nameserver settings, and domain configuration.",
        metadata={"type": "error", "code": "DNS_404"}
    ),

    # --- Auth & Security ---
    Document(
        page_content="The authentication process requires valid credentials. Use OAuth2 for secure API access and refresh tokens for long sessions.",
        metadata={"type": "auth", "method": "OAuth2"}
    ),
    Document(
        page_content="API keys should never be exposed in frontend code. Store secrets in environment variables and rotate them regularly.",
        metadata={"type": "security", "category": "api_keys"}
    ),
    Document(
        page_content="JWT tokens are used to verify user identity. They should include expiration time and must be signed using a secure secret.",
        metadata={"type": "auth", "method": "JWT"}
    ),

    # --- Configuration guides ---
    Document(
        page_content="Router configuration guide: Access the admin panel at 192.168.1.1, enter admin credentials, and update wireless settings.",
        metadata={"type": "config", "category": "router"}
    ),
    Document(
        page_content="To change the Wi-Fi password, open the router dashboard, go to Wireless Settings, enter a new password, and save changes.",
        metadata={"type": "config", "category": "wifi"}
    ),

    # --- Compliance (acronyms like WCAG are missed by vector search) ---
    Document(
        page_content="WCAG 2.1 compliance requires all images to have alt text, sufficient color contrast, keyboard navigation, and readable labels.",
        metadata={"type": "compliance", "standard": "WCAG 2.1"}
    ),
    Document(
        page_content="GDPR compliance requires user consent before collecting personal data. Users must be able to request data deletion.",
        metadata={"type": "compliance", "standard": "GDPR"}
    ),
    Document(
        page_content="PCI DSS compliance is required when handling card payments. Do not store raw card numbers or CVV values.",
        metadata={"type": "compliance", "standard": "PCI DSS"}
    ),

    # --- Pricing & Support ---
    Document(
        page_content="The monthly subscription plan includes cloud backup, device monitoring, priority support, and automatic firmware updates.",
        metadata={"type": "pricing", "plan": "monthly"}
    ),
    Document(
        page_content="Customer support is available Monday to Friday from 9 AM to 6 PM. Emergency network outage support is available 24/7.",
        metadata={"type": "support", "category": "availability"}
    ),
]

print(f"Total documents created: {len(documents)}")

# ─── 1. Vector Retriever (semantic search) ────────────────────────────────────
# Embeds documents using OpenAI, then finds the closest matches by meaning.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="hybrid_search_collection",
)

vector_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Return top 3 matches
)
# Query: "wireless networking hardware" → router/mesh docs
# Query: "E_CONN_REFUSED" → likely misses the error doc (no semantic meaning)

# ─── 2. BM25 Retriever (keyword / exact-match search) ─────────────────────────
# Uses term frequency matching — great for SKUs, error codes, exact names.
bm25_retriever = BM25Retriever.from_documents(documents, k=3)
# Query: "SKU-7742X" → pinpoints the exact product doc
# Query: "WCAG"      → finds the WCAG compliance doc by keyword

# ─── 3. Ensemble Retriever (hybrid) ──────────────────────────────────────────
# Merges vector + BM25 results using weighted reciprocal rank fusion.
# One catches what the other misses.
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],  # Tune: favor meaning or keywords
)
# Result: reliable retrieval even for semantically empty queries.

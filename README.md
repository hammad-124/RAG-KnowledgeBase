<!-- CORE DOCUMENT LOADER -->

PyPdfLoader       ---------------        to load pdf
TextLader       -----------------
DirectoryLoader ------------------
WebBaseLoader ---------------------
UnStructuredLoader ----------------

<!-- POUR CHUNKING VARIBALES THAT EFFECT QUALITY -->
1-Chunk Size
too Small ----> loses context
too large ---->  Dilute Meaning

2-Overlap
3-Split Boundaries
4-Content type


<!-- CHUNKING STARTEGIES -->
1-Fixed
2-Recursive
3-Semantic
4-Late

<!-- seprators for chunking -->
In LangChain, the separator (used primarily within TextSplitters like CharacterTextSplitter or RecursiveCharacterTextSplitter) is absolutely critical.

Separators dictate where those breaks happen. Their importance boils down to two main things:
1. Preserving Semantic Context
2. Retrieval Accuracy (Garbage In, Garbage Out)

<!-- 
How Separators Work: Naive vs. Smart -->

1-The Naive Way:
CharacterTextSplitter
This splitter uses a single fixed separator (the default is a single space " ").

The Risk: It will count characters until it hits your chunk_size, and then look for the next space to split. This often results in chunks breaking right in the middle of a crucial sentence or paragraph, completely degrading the context.

2-The Smart Way:
 RecursiveCharacterTextSplitterThis is the recommended default splitter in LangChain because it uses a hierarchy of separators. Instead of looking for just one character, it tries to split by the most logical boundary first, and only moves down the list if the chunk is still too big.The default separator list for the recursive splitter is

:$$\text{Separators} = [ \text{"\textbackslash n\textbackslash n"}, \text{"\textbackslash n"}, \text{" "}, \text{""} ]$$


<!-- Data types and seperators for splitting for chunks  -->

| Data Type | Best Splitter | Core Separators Used | Why It Matters |
| :--- | :--- | :--- | :--- |
| **Markdown** | `MarkdownHeaderTextSplitter` | `#`, `##`, `###`, `####` | Keeps sections and sub-sections grouped logically under their respective headers. |
| **Code** | `RecursiveCharacterTextSplitter.from_language` | `class`, `def`, `if`, `\n` | Ensures functions and classes aren't cut in half, preserving syntax validity. |
| **JSON** | `RecursiveJsonSplitter` | `{`, `}`, `[`, `]` | Keeps nested key-value pairs together without breaking the JSON structure. |



<!-- When vector search isnt enough -->

All though the correct document is into your database but your Rag search completely failed.

1-Product Code
like SKU-7742X 
user query: what are the specs of product  SKU-7742X
their is no semantic meaning of product code so rag returns some specs but not specifically of that with product code.

2-Acronyms
WCAG compliance
Note:  Model doesnot know observations
it might retun docuent about compliance or requirement but i gonna miss WCAG  

3-Exact Names
If their is a document of family tree and user search john smith accounting vector search might find document of acounting and smith but might skip john because it will semantic search not exact name

4-Error Codes
example "E_CONN_REFUSED"
it doesnt hane semantic meaning so model has no idea what is meant

<!-- 
For this problems BM25 comes into picture -->




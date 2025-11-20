import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    DOCS_DIR = "docs"
    MARKDOWN_DIR = "markdown" 
    PARENT_STORE_PATH = "parent_store"
    CHILD_COLLECTION = "document_child_chunks"

    os.makedirs(DOCS_DIR, exist_ok=True) 
    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    os.makedirs(PARENT_STORE_PATH, exist_ok=True)
    return CHILD_COLLECTION, MARKDOWN_DIR, PARENT_STORE_PATH, os


@app.cell
def _():
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="granite4:tiny-h", temperature=0)
    return (llm,)


@app.cell
def _():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant.fastembed_sparse import FastEmbedSparse

    dense_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    sparse_embeddings = FastEmbedSparse(
        model_name="Qdrant/bm25"
    )
    return dense_embeddings, sparse_embeddings


@app.cell
def _(dense_embeddings):
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    from langchain_qdrant import QdrantVectorStore
    from langchain_qdrant.qdrant import RetrievalMode

    # Initialize Qdrant client (local file-based storage)
    client = QdrantClient(path="qdrant_db")

    # Get embedding dimension
    embedding_dimension = len(dense_embeddings.embed_query("test"))

    def ensure_collection(collection_name):
        """Create Qdrant collection if it doesn't exist"""
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=embedding_dimension,
                    distance=qmodels.Distance.COSINE
                )
            )
    return QdrantVectorStore, RetrievalMode, client, ensure_collection


@app.cell
def _(MARKDOWN_DIR, doc_name, os):
    # import os
    import pymupdf.layout
    import pymupdf4llm
    from pathlib import Path
    import glob

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def pdf_to_markdown(pdf_path, output_dir):
        doc = pymupdf.open(pdf_path)
        md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False, image_path=None)
        md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
        output_path = Path(output_dir) / Path(doc_name).stem
        Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))

    def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
        output_dir = Path(MARKDOWN_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        for pdf_path in map(Path, glob.glob(path_pattern)):
            md_path = (output_dir / pdf_path.stem).with_suffix('.md')
            if overwrite or not md_path.exists():
                pdf_to_markdown(pdf_path, output_dir)

    pdfs_to_markdowns('./docs/*.pdf')
    return Path, glob


@app.cell
def _(
    CHILD_COLLECTION,
    MARKDOWN_DIR,
    Path,
    QdrantVectorStore,
    RetrievalMode,
    client,
    dense_embeddings,
    ensure_collection,
    glob,
    os,
    sparse_embeddings,
):
    # import os
    # import glob
    import json
    # from pathlib import Path
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    from langchain_text_splitters.base import Document


    if client.collection_exists(CHILD_COLLECTION):
        print(f'Removing existing Qdrant collection: {CHILD_COLLECTION}')
        client.delete_collection(CHILD_COLLECTION)
        ensure_collection(CHILD_COLLECTION)
    else:
        ensure_collection(CHILD_COLLECTION)

    child_vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name='sparse',
    )

    def index_documents():
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        min_parent_size = 2000
        max_parent_size = 10000

        all_parent_pairs, all_child_chunks = [], []
        md_files = sorted(glob.glob(os.path.join(MARKDOWN_DIR, "*.md")))

        if not md_files:
            print(f"⚠️  No .md files found in {MARKDOWN_DIR}/")
            return

        for doc_path_str in md_files:
            doc_path = Path(doc_path_str)
            print(f"📄 Processing: {doc_path.name}")

            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    md_text = f.read()
            except Exception as e:
                print(f"❌ Error reading {doc_path.name}: {e}")
                continue

            parent_chunks = parent_splitter.split_text(md_text)
            merged_parents = merge_small_parents(parent_chunks, min_parent_size)
            split_parents = split_large_parents(merged_parents, max_parent_size, child_splitter)
            cleaned_parents = clean_small_chunks(split_parents, min_parent_size)

            for i, p_chunk in enumerate(cleaned_parents):
                parent_id = f"{doc_path.stem}_parent_{i}"
                p_chunk.metadata.update({"source": doc_path.stem + ".pdf", "parent_id": parent_id})
                all_parent_pairs.append((parent_id, p_chunk))
                children = child_splitter.split_documents([p_chunk])
                all_child_chunks.extend(children)

        if not all_child_chunks:
            print("⚠️ No child chunks to index")
            return


    def merge_small_parents(chunks, min_size):
        if not chunks:
            return []

        merged, current = [], None

        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v

            if len(current.page_content) >= min_size:
                merged.append(current)
                current = None

        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                merged.append(current)

        return merged

    def split_large_parents(chunks, max_size, splitter):
        split_chunks = []

        for chunk in chunks:
            if len(chunk.page_content) <= max_size:
                split_chunks.append(chunk)
            else:
                large_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_size,
                    chunk_overlap=splitter._chunk_overlap
                )
                sub_chunks = large_splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)

        return split_chunks

    def clean_small_chunks(chunks, min_size):
        cleaned = []

        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < min_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)

        return cleaned

    index_documents()
    return child_vector_store, json


@app.cell
def _(PARENT_STORE_PATH, child_vector_store, json, llm, os):
    # import json
    from langchain_core.tools import tool

    @tool
    def search_child_chunks(query: str, k: int = 5) -> list[dict]:
        """Search for the top K most relevant child chunks.

        Args:
           query: Search query string
           k: Number of results to return
        """
        try:
           results = child_vector_store.similarity_search(query, k=k, score_threshold=0.7)
           return [
               {
                   "content": doc.page_content,
                   "parent_id": doc.metadata.get("parent_id", ""),
                   "source": doc.metadata.get("source", "")
               }
               for doc in results
           ]
        except Exception as e:
           print(f"Error searching child chunks: {e}")
           return []

    @tool
    def retrieve_parent_chunks(parent_ids: list[str]) -> list[dict]:
        """Retrieve full parent chunks by their IDs.

        Args:
           parent_ids: List of parent chunk IDs to retrieve
        """
        unique_ids = sorted(list(set(parent_ids)))
        results = []

        for parent_id in unique_ids:
           file_path = os.path.join(PARENT_STORE_PATH, parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json")
           if os.path.exists(file_path):
               try:
                   with open(file_path, "r", encoding="utf-8") as f:
                       doc_dict = json.load(f)
                       results.append({
                           "content": doc_dict["page_content"],
                           "parent_id": parent_id,
                           "metadata": doc_dict["metadata"]
                       })
               except Exception as e:
                   print(f"Error loading parent chunk {parent_id}: {e}")

        return results

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools([search_child_chunks, retrieve_parent_chunks])
    return llm_with_tools, retrieve_parent_chunks, search_child_chunks


@app.cell
def _():
    from langchain_core.messages import SystemMessage

    AGENT_SYSTEM_PROMPT = """
    You are an intelligent assistant that MUST use the available tools to answer questions.

    **MANDATORY WORKFLOW — Follow these steps for EVERY question:**

    1. **Call `search_child_chunks`** with the user's query (K = 3–7).

    2. **Review the retrieved chunks** and identify the relevant ones.

    3. **For each relevant chunk, call `retrieve_parent_chunks`** using its parent_id to get full context.

    4. **If the retrieved context is still incomplete, retrieve additional parent chunks** as needed.

    5. **If metadata helps clarify or support the answer, USE IT**

    6. **Answer using ONLY the retrieved information**
       - Cite source files from metadata.

    7. **If no relevant information is found,** rewrite the query into an **answer-focused declarative statement** and search again **only once**.
    """

    agent_system_message = SystemMessage(content=AGENT_SYSTEM_PROMPT)
    return SystemMessage, agent_system_message


@app.cell
def _():
    from langgraph.graph import MessagesState
    from pydantic import BaseModel, Field

    class State(MessagesState):
        questionIsClear: bool
        conversation_summary: str = ""

    class QueryAnalysis(BaseModel):
        is_clear: bool = Field(
            description="Indicates if the user's question is clear and answerable."
        )
        questions: list[str] = Field(
            description="List of rewritten, self-contained questions."
        )
        clarification_needed: str = Field(
            description="Explanation if the question is unclear."
        )
    return QueryAnalysis, State


@app.cell
def _(
    QueryAnalysis,
    State,
    SystemMessage,
    agent_system_message,
    llm,
    llm_with_tools,
):
    from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
    from typing import Literal


    def analyze_chat_and_summarize(state: State):
        """
        Analyzes chat history and summarizes key points for context.
        """
        if len(state["messages"]) < 4:  # Need some history to summarize
            return {"conversation_summary": ""}

        # Extract relevant messages (excluding current query and system messages)
        relevant_msgs = [
            msg
            for msg in state["messages"][:-1]  # Exclude current query
            if isinstance(msg, (HumanMessage, AIMessage))
            and not getattr(msg, "tool_calls", None)
        ]
        if not relevant_msgs:
            return {"conversation_summary": ""}

        summary_prompt = """**Summarize the key topics and context from this conversation concisely (1-2 sentences max).**
        Discard irrelevant information, such as misunderstandings or off-topic queries/responses.
        If there are no key topics, return an empty string.

        """
        for msg in relevant_msgs[-6:]:  # Last 6 messages for context
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            summary_prompt += f"{role}: {msg.content}\n"

        summary_prompt += "\nBrief Summary:"
        summary_response = llm.with_config(temperature=0.3).invoke(
            [SystemMessage(content=summary_prompt)]
        )
        return {"conversation_summary": summary_response.content}


    def analyze_and_rewrite_query(state: State):
        """
        Analyzes user query and rewrites it for clarity, optionally using conversation context.
        """
        last_message = state["messages"][-1]
        conversation_summary = state.get("conversation_summary", "")

        context_section = (
            f"**Conversation Context:**\n{conversation_summary}"
            if conversation_summary.strip()
            else "**Conversation Context:**\n[First query in conversation]"
        )

        # Create analysis prompt
        prompt = f"""
        **Rewrite the user's query** to be clear, self-contained, and optimized for information retrieval.

        **User Query:**
        "{last_message.content}"

        {context_section}

        **Instructions:**

        1. **Resolve references for follow-ups:**
        - If the query uses pronouns or refers to previous topics, use the context to make it self-contained.

        2. **Ensure clarity for new queries:**
        - Make the query specific, concise, and unambiguous.

        3. **Correct errors and interpret intent:**
        - If the query is grammatically incorrect, contains typos, or has abbreviations, correct it and infer the intended meaning.

        4. **Split only when necessary:**
        - If multiple distinct questions exist, split into **up to 3 focused sub-queries** to avoid over-segmentation.
        - Each sub-query must still be meaningful on its own.

        5. **Optimize for search:**
        - Use **keywords, proper nouns, numbers, dates, and technical terms**.
        - Remove conversational filler, vague words, and redundancies.
        - Make the query concise and focused for information retrieval.

        6. **Mark as unclear if intent is missing:**
        - This includes nonsense, gibberish, insults, or statements without an apparent question.
        """

        llm_with_structure = llm.with_config(
            temperature=0.3
        ).with_structured_output(QueryAnalysis)
        response = llm_with_structure.invoke([SystemMessage(content=prompt)])

        if response.is_clear:
            # Remove all non-system messages
            delete_all = [
                RemoveMessage(id=m.id)
                for m in state["messages"]
                if not isinstance(m, SystemMessage)
            ]

            # Format rewritten query
            rewritten = (
                "\n".join(
                    [f"{i + 1}. {q}" for i, q in enumerate(response.questions)]
                )
                if len(response.questions) > 1
                else response.questions[0]
            )
            return {
                "questionIsClear": True,
                "messages": delete_all + [HumanMessage(content=rewritten)],
            }
        else:
            clarification = (
                response.clarification_needed
                or "I need more information to understand your question."
            )
            return {
                "questionIsClear": False,
                "messages": [AIMessage(content=clarification)],
            }


    def human_input_node(state: State):
        """Placeholder node for human-in-the-loop interruption"""
        return {}


    def route_after_rewrite(state: State) -> Literal["agent", "human_input"]:
        """Route to agent if question is clear, otherwise wait for human input"""
        return "agent" if state.get("questionIsClear", False) else "human_input"


    def agent_node(state: State):
        """Main agent node that processes queries using tools"""
        messages = [SystemMessage(content=agent_system_message.content)] + state[
            "messages"
        ]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    return (
        HumanMessage,
        agent_node,
        analyze_and_rewrite_query,
        analyze_chat_and_summarize,
        human_input_node,
        route_after_rewrite,
    )


@app.cell
def _(
    State,
    agent_node,
    analyze_and_rewrite_query,
    analyze_chat_and_summarize,
    human_input_node,
    retrieve_parent_chunks,
    route_after_rewrite,
    search_child_chunks,
):
    from langgraph.graph import START, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()

    graph_builder = StateGraph(State)

    graph_builder.add_node("summarize", analyze_chat_and_summarize)
    graph_builder.add_node("analyze_rewrite", analyze_and_rewrite_query)
    graph_builder.add_node("human_input", human_input_node)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode([search_child_chunks, retrieve_parent_chunks]))

    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", "analyze_rewrite")
    graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)
    graph_builder.add_edge("human_input", "analyze_rewrite")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")

    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_input"]
    )

    print("✓ Agent graph compiled successfully.")
    return (agent_graph,)


@app.cell
def _(HumanMessage, agent_graph):
    import gradio as gr
    import uuid

    def create_thread_id():
        """Generate a unique thread ID for each conversation"""
        return {"configurable": {"thread_id": str(uuid.uuid4())}}

    def clear_session():
        """Clear thread for new conversation and clear up checkpointer state"""
        global config
        agent_graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
        config = create_thread_id()

    def chat_with_agent(message, history):
        """
        Handle chat with human-in-the-loop support.
        Returns: response text
        """
        current_state = agent_graph.get_state(config)
        if current_state.next:
            agent_graph.update_state(config,{"messages": [HumanMessage(content=message.strip())]})
            result = agent_graph.invoke(None, config)
        else:
            result = agent_graph.invoke({"messages": [HumanMessage(content=message.strip())]}, config)
        return result['messages'][-1].content

    config = create_thread_id()

    with gr.Blocks(theme=gr.themes.Citrus()) as demo:
        chatbot = gr.Chatbot(height=600, placeholder="<strong>DIRMACS CHATBOT - Ask me anything!</strong><br><em>I'll search, reason, and act to give you the best answer :)</em>")
        chatbot.clear(clear_session)
        gr.ChatInterface(fn=chat_with_agent, type="messages", chatbot=chatbot)

    print("\nLaunching application...")
    demo.launch()
    return


if __name__ == "__main__":
    app.run()

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
    return


@app.cell
def _():
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="granite4:tiny-h", temperature=0)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

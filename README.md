# LangGraph ReAct Agent Demo

## Overview

This repository contains a **Jupyter Notebook** that demonstrates how to build, extend, and run a **ReAct‑style agent** using the **LangGraph** framework, **LangChain**, and the **Groq** LLM (GPT‑OSS‑120B).  
The notebook walks through:

- Creating a LLM‑only chatbot.
- Adding external tools (search via **Tavily**, a custom Python function).
- Enabling tool‑calling with LangChain’s `bind_tools`.
- Implementing a ReAct loop where the agent can **act → observe → reason** repeatedly.
- Adding **memory** (checkpointing) so the agent retains context across turns.

The result is a lightweight, modular, and reusable graph‑based agent that can answer general queries, perform web searches, and execute arbitrary Python functions while remembering prior interactions.

---

## Problem Statement

Large language models excel at natural‑language understanding but often lack the ability to:

1. **Interact with external tools** (search engines, calculators, APIs).
2. **Maintain conversational state** across multiple turns.
3. **Iteratively reason** (act → observe → reason) when a single LLM call is insufficient.

The goal of this project is to **bridge these gaps** by constructing a graph‑driven ReAct agent that can seamlessly call tools, observe their outputs, and continue reasoning until a final answer is produced, all while persisting conversation history.

---

## Approach

| Step                   | Description                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **1️⃣ Define State**    | Use a `TypedDict` with a `messages` field annotated by `add_messages` to hold the chat history.                                      |
| **2️⃣ Simple LLM Node** | Build a `StateGraph` with a single node (`llmchatbot`) that forwards the user message to the Groq LLM.                               |
| **3️⃣ Add Tools**       | - **TavilySearch** – web‑search tool (max 2 results). <br> - **multiply** – custom Python function.                                  |
| **4️⃣ Bind Tools**      | `llm.bind_tools(tools)` creates an LLM capable of emitting tool calls.                                                               |
| **5️⃣ Tool Node**       | `ToolNode(tools)` executes the requested tool and returns its result as a new message.                                               |
| **6️⃣ ReAct Loop**      | Conditional edges (`tools_condition`) route the flow back to the LLM after a tool call, enabling repeated act‑observe‑reason cycles. |
| **7️⃣ Memory**          | `MemorySaver` checkpoint stores the full state per `thread_id`, allowing the agent to recall prior messages.                         |
| **8️⃣ Visualization**   | Mermaid diagrams (`graph.get_graph().draw_mermaid_png()`) illustrate the graph topology at each stage.                               |

The final graph looks like:

```
START → LLM (tool‑calling) → [ToolNode] ↺ (loop) → END
```

When a tool is not needed, the flow proceeds directly to `END`.

---

## Tech Stack

| Component     | Library / Service                                 | Purpose                                  |
| ------------- | ------------------------------------------------- | ---------------------------------------- |
| **LangGraph** | `langgraph`                                       | Graph‑based orchestration of LLM + tools |
| **LangChain** | `langchain`, `langchain_groq`, `langchain_tavily` | LLM wrappers, tool integration           |
| **Groq**      | `groq:openai/gpt-oss-120b`                        | Fast, open‑source LLM inference          |
| **Tavily**    | `langchain_tavily.TavilySearch`                   | Lightweight web‑search tool              |
| **Python**    | `3.10+`                                           | Core language                            |
| **Jupyter**   | `ipykernel`                                       | Interactive notebook                     |
| **dotenv**    | `python-dotenv`                                   | Load API keys from `.env`                |
| **Mermaid**   | `graph.draw_mermaid_png()`                        | Visual graph rendering                   |
| **Memory**    | `langgraph.checkpoint.memory.MemorySaver`         | Persistent conversation state            |

---

## Project Structure

```
├── notebook.ipynb          # Full demonstration (the source of this README)
├── .env.example            # Template for required environment variables
├── requirements.txt        # Python dependencies
└── README.md               # ← You are here
```

_The notebook is self‑contained; no additional Python modules are required._

---

## Results

| Query                                                                                  | Agent Output                                                                                              | Observations                                                         |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **“Hi”**                                                                               | Friendly greeting from Groq LLM.                                                                          | Simple LLM node works.                                               |
| **“What is langgraph?”**                                                               | Search results from Tavily (2 snippets).                                                                  | Tool call executed correctly.                                        |
| **“What is 2 multiply by 3?”**                                                         | Direct answer `6` (via `multiply` function).                                                              | Tool calling works for custom Python functions.                      |
| **Combined query**<br>`“What is the recent news with Sam Altman? and multiply 2 by 4”` | ReAct loop: first performs web search, then calls `multiply`, finally returns both pieces of information. | Demonstrates iterative reasoning.                                    |
| **Memory test**<br>1️⃣ “Hi, my name is Jag” <br>2️⃣ “What is my name?”                   | Agent remembers the name “Jag”.                                                                           | MemorySaver persists state across turns (identified by `thread_id`). |

The agent reliably switches between LLM reasoning and tool execution, and retains context when configured with a checkpoint.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/langgraph-react-agent-demo.git
cd langgraph-react-agent-demo

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=sk-...

# 5. Launch Jupyter
jupyter notebook notebook.ipynb
```

**Required environment variables**

| Variable                      | Description                               |
| ----------------------------- | ----------------------------------------- |
| `GROQ_API_KEY`                | API key for Groq LLM access.              |
| `TAVILY_API_KEY` _(optional)_ | If you want higher‑quota Tavily searches. |

---

## Usage

Inside the notebook you will find the following runnable sections:

1. **LLM‑only chatbot** – `graph.invoke({"messages": "Hi"})`
2. **Tool‑enabled chatbot** – add `TavilySearch` and `multiply`, then call `graph.invoke(...)`.
3. **ReAct loop** – the graph with conditional edges loops back to the LLM after each tool call.
4. **Memory‑backed agent** – instantiate `MemorySaver()` and pass `checkpointer=memory` to `graph.compile()`. Use a `config` dict with a `thread_id` to keep conversation state.

Typical interaction pattern:

```python
config = {"configurable": {"thread_id": "session-42"}}
response = graph.invoke({"messages": "What is the recent news with Sam Altman?"}, config=config)
print(response["messages"][-1].content)
```

The agent will automatically decide whether to call a tool, observe the result, and continue reasoning until it produces a final answer.

---

## Future Improvements

| Area                      | Planned Enhancement                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------------------- |
| **Additional Tools**      | Integrate APIs (e.g., weather, calculator, database) via LangChain tool wrappers.               |
| **Dynamic Tool Registry** | Load tools from a configuration file to avoid hard‑coding in the notebook.                      |
| **Streaming Responses**   | Use `graph.stream()` to emit partial LLM outputs for a more responsive UI.                      |
| **Evaluation Suite**      | Automated tests comparing agent answers against ground‑truth datasets (e.g., HotpotQA).         |
| **Deployment**            | Wrap the graph in a FastAPI/Flask endpoint or a Streamlit UI for public demo.                   |
| **Advanced Memory**       | Swap `MemorySaver` for a vector‑store (FAISS, Chroma) to enable retrieval‑augmented generation. |
| **Prompt Engineering**    | Experiment with system prompts that better guide the ReAct reasoning cycle.                     |
| **Error Handling**        | Graceful fallback when a tool fails or returns empty results.                                   |

Contributions are welcome—feel free to open issues or submit pull requests!

---

_Happy building with LangGraph!_

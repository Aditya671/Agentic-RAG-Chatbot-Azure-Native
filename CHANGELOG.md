# Changelog

All notable changes to this project will be documented in this file. This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-28

### Added

- **Asynchronous Task Processing**: Integrated **Celery** and **Redis** to create a distributed task queue. File indexing is now offloaded to a background worker, preventing UI blocking and improving system responsiveness.
- **Persistent Knowledge Graph**: Implemented `GraphRAGSystem` with **NebulaGraph** as a persistent backend, enabling scalable, multi-session relationship querying. Includes a graceful fallback to an in-memory store if NebulaGraph is unavailable.
- **Neural Reranking Layer**: Added an `LLMRerank` postprocessor to the RAG pipeline to refine search results, improving the relevance and accuracy of retrieved documents.
- **Interactive Feature Toggles**: Implemented `Switch` widgets in the Chainlit UI, allowing users to dynamically enable or disable the Neural Reranker and GraphRAG features during a session.
- **Observability Hooks**: Prepared the system for deep tracing by adding support for **LangSmith** environment variables, enabling full visibility into agent decision-making and tool usage.

### Changed

- **Agent Tooling**: Refactored the file upload tool (`upload_and_index_files_async`) to dispatch tasks to the Celery queue and introduced a new tool (`check_indexing_status_tool`) to monitor task progress.

### Security

- **Secure Code Execution**: Replaced the insecure `exec()` call with a **secure E2B sandbox** (`CodeInterpreterSandbox`). All agent-generated Python code is now executed in an isolated, sandboxed environment, mitigating code injection and other execution risks.

---

## [1.0.9] - 2026-02-27

### Added

- **Automatic Thread Titling**: Implemented `generate_thread_title` method, which uses a smaller, faster LLM (GPT-4.1-Mini) to create a concise title for the conversation based on the initial user query and assistant response.

---

## [1.0.8] - 2026-02-26

### Added

- **Specialized Coding Assistant Mode**: Introduced a `enable_coding_assistant` flag that switches the agent's system prompt to `AGENTIC_AI_CODEX_PROMPT`, optimizing its behavior for code generation, explanation, and analysis tasks.

---

## [1.0.7] - 2026-02-25

### Added

- **Token Usage Monitoring**: Integrated `llama_index.core.callbacks.TokenCountingHandler` to track prompt, completion, and total token usage for each agent interaction.
- **Metrics Endpoint**: Added a `get_token_counts` method to expose detailed token metrics, including the percentage of the context window used, for monitoring and cost control.

---

## [1.0.6] - 2026-02-24

### Added

- **Dynamic Agent Configuration**: Implemented a suite of setter methods (`set_selected_model`, `set_llm_creativity_level`, etc.) in `AsyncAgenticAiSystem` to allow for real-time adjustments of the agent's core parameters.

### Changed

- The agent now dynamically rebuilds its components (LLM, index, tools) when settings are updated via the UI, allowing for live experimentation with different models and configurations.

---

## [1.0.5] - 2026-02-23

### Added

- **Citation and Source Handling**: Implemented a citation processing layer (`get_retriever_metadata` and `utility.parse_response_sources`) to extract and display source document metadata from the RAG pipeline's output.

---

## [1.0.4] - 2026-02-22

### Added

- **Automated Context Management**: Implemented automatic conversation summarization to manage the context window efficiently, reducing token consumption in long-running conversations.

### Fixed

- **Token Overflow**: Resolved issues with long conversations exceeding model context limits by implementing the automated summarization and sliding window memory.

---

## [1.0.3] - 2026-02-21

### Added

- **Smart Ingestion Pipeline**: Created a `UserUploadedFileIndexer` that computes file hashes to prevent redundant processing and embedding of re-uploaded files.
- **Concurrency Conflicts**: Applied `nest_asyncio` to prevent event loop conflicts during the execution of nested asynchronous agentic tools.

---

## [1.0.2] - 2026-02-19

### Added

- **Internet Grounding**: Equipped the agent with a `bing_grounding_tool` to perform real-time web searches for up-to-date information.

### Security

- **Agentic Loop Safeguards**: Implemented max-loop constraints within the agentic orchestrator to prevent runaway "inference flooding," providing critical cost and rate-limit protection.

---

## [1.0.1] - 2026-02-16

### Added

- **Structured Data Analysis**: Integrated a `PandasQueryEngine` as a new agent tool to enable natural language querying of structured CSV data.


### Changed

- **Agentic Router Logic**: Enhanced the `FunctionAgent` to act as a router, dynamically selecting between the `RetrieverTool` (for PDFs) and the `PandasQueryEngine` (for CSVs).

---

## [1.0.0] - 2026-02-14

### Added

- **Agentic Core (`AsyncAgenticAiSystem`):** Initial implementation of the main agentic class to manage state, models, and tools.
- **Unstructured Data Retrieval**: Integrated the first agent tool, a `RetrieverTool` connected to **Azure AI Search**, for semantic search over PDF documents.
- **Persistent Memory Layer**: Established a high-availability data layer using **Azure Cosmos DB** to persist chat history and session state.

### Changed

- **Modular Backend Architecture**: Refactored the project into a professional `src/backend` layout, separating `orchestration`, `indexer`, and `config`.

### Security

- **Credential Management**: Migrated all secret management to **Azure Key Vault** and implemented `CredentialManager` and `DefaultAzureCredential` for secure, identity-based access.
- **User Authentication**: Implemented full **Azure AD (OAuth)** integration for secure user login.

---

## [0.0.5] - 2026-02-07

### Added

- **Application Entrypoint (`app.py`):** Initialized the core Chainlit application, establishing the user interface and entry point for the agentic system.
- **User Session Management:** Implemented basic user session handling using `chainlit.user_session` to manage application state.
- **Core Chat Hooks:** Set up fundamental Chainlit lifecycle hooks, including `@cl.on_chat_start` and `@cl.on_message`, `@cl.on_settings_update`, `@cl.on_chat_resume`, and `@cl.on_feedback`.
- **Asynchronous Streaming:** Configured the UI to handle and render asynchronous, token-by-token streaming responses.
- **Interactive Settings Panel:** Created the initial `cl.ChatSettings` panel for dynamic agent parameter adjustment.
- **File Upload Interface:** Integrated Chainlit's file upload capability.
- **Application Starter Template**: Established a foundational project template with essential configurations, directory structure, and boilerplate code to accelerate development and ensure consistency across deployments.


### Security

- **Authentication Stub:** Laid the groundwork for user authentication by integrating the `@cl.oauth_callback` decorator for Azure AD.

---

## [0.0.0] - 2026-01-28

### Added

- **Project Initialization**: Set up the initial Git repository and project scaffolding.
- **Build System**: Created `pyproject.toml` to manage dependencies, project metadata, and build configurations.
- **Development Automation**: Implemented a `Makefile` for streamlined installation, testing, and execution commands.
- **Core Logging**: Developed a centralized logger (`app_logger.py`) for consistent, timestamped logging across the application with configurable levels.
- **Architectural Blueprint**: Defined the high-level enterprise architecture, separating frontend, backend, and data layers.

---

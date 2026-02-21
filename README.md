# Agentic-RAG-Chatbot-Azure-Native
An Enterprise ready multi-agent AI assistant featuring hybrid search, automated CSV data analysis, and persistent conversation memory using Azure Cosmos DB.

---

## Key Technical Features:
- Multi-Index Orchestration: Dynamically switches between Vector indexes for unstructured PDFs and a PandasQueryEngine for structured CSV data (like Salesforce meeting records).
-  Persistent Memory Layer: Custom integration with Azure Cosmos DB for high-availability storage of chat history and user feedback.
- Smart Ingestion Pipeline: Features a UserUploadedFileIndexer that handles local file uploads, computes hashes to avoid redundant indexing, and generates automatic document summaries.
- Agentic Tools: Integrated tools for Bing Web Search (Internet grounding) and a specialized Coding Assistant mode.

---
  
## Tech Stack:
- Orchestration: LlamaIndex, Chainlit.
- LLMs: Azure OpenAI (GPT-4o, GPT-5.1).
- Database: Azure Cosmos DB, Azure AI Search (Vector Store).
- Infrastructure: Azure Blob Storage, Azure Key Vault.

---
<!--
## Architecture
1. The Model Gateway (Abstraction Layer)
Instead of hardcoding Azure OpenAI calls, the system uses a Gateway Pattern.

How it scales: You can add a new model (e.g., Anthropic or a fine-tuned Mistral) by simply updating a config file.

Benefit: Protects the application from API downtime or price hikes by allowing instant model switching.

2. The Dynamic Tool Registry
The "Agentic Tools" are no longer hard-wired into the agent.

How it scales: Tools are treated as independent modules. Adding a "Slack Notifier" or "Jira Ticketing" tool is as simple as dropping a new script into the /tools directory.

Benefit: Keeps the codebase clean and allows for "Tool-on-Demand" loading to save on token context.

3. Hybrid Storage & Memory
Short-term: Managed via Azure Cosmos DB (Session-based).

Long-term: Documents are indexed in Azure AI Search, but the architecture allows for a "Graph" layer to be added for complex relationship mapping between data points.
-->
## Technical Challenges & Design Decisions
Building a production-ready RAG system involves more than just hitting an API. Below are the key engineering hurdles I solved:
  
  1. **Structured vs. Unstructured Data Ambiguity**
     - **The Challenge**: Users often ask questions that require "joining" data across different formats (e.g., "Summarize the project notes in this PDF and compare them to the budget in the CSV"). Standard RAG fails here because vector search is poor at tabular math.
     - **The Solution**: Implemented a Router Orchestrator. I used LlamaIndex to build a `QueryEngine` that detects the intent. If the query requires calculation, it routes to a `PandasQueryEngine`; if it requires semantic meaning, it hits the Azure AI Search vector index.
  
  4. **State Management & "Memory Leak" in Conversations**
      - **The Challenge**: Storing chat history in-memory (RAM) causes data loss on server restarts and prevents horizontal scaling across multiple containers.
      - **The Solution**: Integrated **Azure Cosmos DB** as a persistent NoSQL backend.
        - **TTL (Time to Live)**: Configured for session cleanup.
        - **Partition Keys**: Used `SessionID` as the partition key to ensure millisecond latency even as the database scales to millions of conversations.
  
  5. **Preventing "Inference Flooding" (Cost & Rate Limits)**
      - **The Challenge**: Agentic loops (like Bing Search) can sometimes "hallucinate" and enter an infinite loop of API calls, draining the Azure OpenAI token quota.
      - **The Solution**: * Implemented **Max-Loop Constraints** on the Agentic Orchestrator.
      - Developed a **UserUploadedFileIndexer** that computes MD5 Hashes of files. If a user re-uploads the same 50MB PDF, the system recognizes the hash and skips the expensive embedding/ingestion process.
---

## Tech Stack Justification
| Component     | Choice          | Why not the alternative?                        |
| ------------- | --------------- | ----------------------------------------------- |
| Vector Store  | Azure AI Search | Better enterprise security and integrated "Hybrid Search" (Keyword + Vector) compared to standalone Pinecone. |
| Orchestration | LlamaIndex      | Superior "Data Agency" features for structured data compared to LangChain’s more generic chains. |
| Identity      | Azure Key Vault | Avoids "Secret Leakage" in environment variables; essential for SOC2 compliance. |

---

## Upcoming Features & Roadmap:
### Model Agnostic Infrastructure
  - [ ] **Multi-Model Routing:** Implement a LiteLLM proxy layer to seamlessly switch between Azure OpenAI, Anthropic (Claude 3.5 Sonnet), and local models via Ollama.
  - [ ] **Dynamic Model Fallback:** Automatic failover logic to high-context models (like GPT-4o) when complex reasoning is required, while using smaller models (GPT-4o-mini) for routing/summarization to optimize costs.

### Extensible Agentic Ecosystem
  - [ ] **Plug-and-Play Tool Registry:** Transition to a decorator-based tool system, allowing new Python functions or API wrappers to be registered as Agent tools with zero configuration.
  - [ ] **Code Interpreter Sandbox:** Migration to an E2B or Docker-based execution environment for safer, more robust Python data analysis.
  - [ ] **Advanced RAG (GraphRAG):** Integration of Knowledge Graphs to map entities across Salesforce records and internal PDFs for multi-hop reasoning.

### Enterprise Scalability & Observability
  - [ ] **Distributed Task Queue:** Implementation of Celery/Redis for handling long-running document ingestion and heavy data processing tasks asynchronously.
  - [ ] **Traces & Evaluation:** Integration of Arize Phoenix or LangSmith for deep tracing of agentic thought chains and RAG "Faithfulness" scoring.
  - [ ] **Kubernetes Deployment:** Helm charts for auto-scaling the Chainlit frontend and the ingestion workers based on traffic spikes.

### Security & Governance
  - [ ] **PII Redaction Layer:** Automated masking of sensitive data before it reaches the LLM provider.
  - [ ] **RBAC (Role-Based Access Control):** Granular permissions for document indexes based on user identity via Entra ID (Azure AD).


# Enterprise Agentic RAG Chatbot (Global Edition)

An advanced, production-grade AI assistant featuring multi-agent orchestration, hybrid vector search, and automated data analysis. [cite_start]This system is designed for high-availability enterprise environments, leveraging a robust Azure-native stack to provide grounded, context-aware insights across structured and unstructured data[cite: 84, 85].

---

## 🚀 Core Features

### 🧠 Hybrid Agentic Orchestration
* [cite_start]**Multi-Model Support:** Dynamically switches between models like GPT-4o and GPT-5.1 depending on reasoning requirements[cite: 127, 128].
* [cite_start]**Task-Specific Agents:** Features a `FunctionAgent` architecture that selects the best tool for the job, whether it's document retrieval, internet search, or Python-based data analysis[cite: 115, 116].
* [cite_start]**Global Context Awareness:** A sophisticated system prompt ensures the AI acts as a Technical Architect & Engineering Lead, maintaining high standards for code and data integrity across global operations[cite: 272, 273, 274].

### 🔍 Advanced RAG Pipeline
* [cite_start]**Unstructured Data (PDFs):** Uses **Azure AI Search** with semantic hybrid retrieval to query deep document repositories[cite: 91, 399].
* [cite_start]**Structured Data (CSV):** An integrated **PandasQueryEngine** allows the agent to reason over complex tabular data using natural language[cite: 102, 103, 104].
* [cite_start]**Smart Ingestion:** Automated `UserUploadedFileIndexer` that handles local file uploads, computes hashes to prevent redundant indexing, and generates automatic summaries for quick previews[cite: 148, 149, 151, 161].

### 💾 Enterprise-Grade Infrastructure
* [cite_start]**Persistent Memory:** Chat history, user feedback, and UI elements are persisted in **Azure Cosmos DB**, enabling seamless session resumption[cite: 331, 332, 386].
* [cite_start]**Secure Vaulting:** All sensitive API keys and connection strings are managed via **Azure Key Vault**[cite: 90, 137].
* [cite_start]**Real-time UI:** Built with **Chainlit**, featuring streaming responses, interactive settings, and live file upload status[cite: 1, 51, 64].

---

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Orchestration** | [LlamaIndex](https://www.llamaindex.ai/) |
| **Interface** | [Chainlit](https://chainlit.io/) |
| **LLMs** | Azure OpenAI (GPT-4o, GPT-5.1) |
| **Vector Store** | Azure AI Search |
| **Database** | Azure Cosmos DB (SQL API) |
| **File Storage** | Azure Blob Storage |
| **Security** | Azure Key Vault & Managed Identity |

---

## 🏗 Architectural Highlights

### 🔄 Dynamic Session Management
The engine partitions conversation history into "past" and "current" segments. [cite_start]It automatically summarizes older context to stay within token limits while preserving critical session data for the LLM[cite: 92, 93, 122, 123].

### 🧪 Robust Logging & Error Handling
[cite_start]A custom `setup_logger` provides granular tracking across the application, including specific suppresses for Azure SDK noise and specialized filters for third-party libraries like LlamaIndex[cite: 129, 130, 131].

---

## ⚙️ Configuration & Setup

1. **Environment Variables:**
   [cite_start]Create a `.env` file based on the provided configuration logic, including your Azure endpoints and Key Vault URLs[cite: 1, 136, 144].

2. **Authentication:**
   [cite_start]The system uses **Azure Active Directory (OAuth)** for user identity and **DefaultAzureCredential** for secure service-to-service communication[cite: 12, 13, 17, 398].

3. **Running the App:**
   ```bash
   chainlit run App.py

# Enterprise RAG Chatbot Backend Architecture

```mermaid
flowchart LR
    subgraph Retrievers
        A1[Azure Blob Retriever]
        A2[S3 Blob Retriever]
        A3[GCP Storage Retriever]
        A4[SharePoint Retriever]
    end

    subgraph Indexers
        B1[PDF Indexer]
        B2[CSV/Docx/TXT Indexer]
        B3[Llama Indexer]
    end

    subgraph Databases
        C1[Azure Cosmos DB]
        C2[MongoDB]
        C3[Postgres]
        C4[Oracle DB]
    end

    subgraph VectorStores
        D1[Azure AI Search]
        D2[Pinecone]
        D3[Weaviate]
        D4[Milvus]
    end

    subgraph Credentials
        E1[Azure Credential Manager]
        E2[AWS Credential Manager]
        E3[GCP Credential Manager]
        E4[Vault Manager]
    end

    subgraph Orchestration
        F1[LLM Loader]
        F2[Prompts]
        F3[Planner]
        F4[Multi-Agent Orchestration]
    end

    subgraph Services
        G1[Auth Service]
        G2[Logging & Monitoring]
        G3[Security Service]
    end

    subgraph Utils
        H1[App Logger]
        H2[File Utils]
    end

    subgraph Integrations
        I1[Salesforce Connector]
        I2[SAP Connector]
        I3[ServiceNow Connector]
    end

    subgraph API
        J1[FastAPI/Chainlit Server]
    end

    %% Connections
    A1 --> B1
    A2 --> B1
    A3 --> B2
    A4 --> B2

    B1 --> D1
    B1 --> D2
    B2 --> D3
    B3 --> D4

    D1 --> F1
    D2 --> F1
    D3 --> F1
    D4 --> F1

    C1 --> F1
    C2 --> F1
    C3 --> F1
    C4 --> F1

    E1 --> F1
    E2 --> F1
    E3 --> F1
    E4 --> F1

    F1 --> G1
    F1 --> G2
    F1 --> G3

    G1 --> J1
    G2 --> J1
    G3 --> J1

    H1 --> J1
    H2 --> J1

    I1 --> J1
    I2 --> J1
    I3 --> J1

    J1 --> EnterpriseUsers[(Enterprise Users)]
    J1 --> IdentitySSO[(Identity & SSO)]
```

---

This diagram illustrates the modular backend architecture of the Agentic RAG Chatbot platform, designed for enterprise deployment across multi-cloud environments.

## Key Components

- **Retrievers**: Connect to cloud/on-prem storage (Azure Blob, S3, GCP, SharePoint).
- **Indexers**: Chunk and embed documents (PDF, CSV, DOCX, TXT, DataFrame).
- **Vector Stores**: Store embeddings (Azure AI Search, Pinecone, Weaviate, MongoDB).
- **Databases**: Metadata and persistence (CosmosDB, MongoDB, Postgres, Oracle).
- **Credentials**: Secure access to cloud platforms (Azure, AWS, GCP, Vault).
- **Orchestration**: Multi-agent planning, LLM loading, prompt management.
- **Services**: Auth, logging, monitoring, security.
- **Utils**: Logging, file utilities, wrappers.
- **Integrations**: Enterprise SaaS connectors (Salesforce, SAP, ServiceNow).
- **API Server**: FastAPI/Chainlit interface for enterprise users and identity systems.

## Architecture Flow

1. **Retrievers** pull files from cloud storage.
2. **Indexers** process and chunk documents.
3. Chunks are stored in **Vector Stores** and metadata in **Databases**.
4. **Orchestration** layer manages agentic flows and LLM interactions.
5. **Services** handle enterprise-grade auth, logging, and security.
6. **Integrations** connect to enterprise platforms.
7. **API Server** exposes chatbot interface to users and identity systems.

![Enterprise Architecture](architecture.png)

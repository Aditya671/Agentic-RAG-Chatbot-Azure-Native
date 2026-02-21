# Changelog

All notable changes to this project will be documented in this file. This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-21

### Added
- [src.backend]**Global Agentic Engine**: Initial release of the `AsyncAgenticCSVChatEngine` supporting multi-agent orchestration for global enterprise data.
- [src.backend]**Multi-Model Intelligence**: Integrated support for `GPT-5.1` and `O4-Mini` models with dynamic reasoning effort settings.
- [src.backend]**Azure-Native Data Layer**: Established a persistent data layer using **Azure Cosmos DB** to store threaded conversations, user feedback, and UI elements.
- [src.backend]**Intelligent File Indexing**: Implemented `UserUploadedFileIndexer` with **SHA256 hashing** to prevent redundant processing and ensure data integrity.
- [src.backend]**Hybrid Search Capabilities**: Combined **Azure AI Search** for unstructured document retrieval with a **PandasQueryEngine** for structured CSV analysis.
- [src.backend]**Citations & Grounding**: Real-time streaming of response citations with support for PDF source rendering and page-level precision.
- [src.backend]**Internet Grounding Tool**: Integrated Bing search capabilities via Azure AI Foundry Agents for real-time web verification.
- [src.backend]**Automated Summarization**: Added background conversation summarization to optimize context window usage and reduce token costs.

### Changed
- **Modular Architecture**: Refactored the project into a professional `src/` layout, separating `auth`, `backend`, and `frontend` logic.
- [src.backend]**Enhanced Security**: Migrated all secret management to **Azure Key Vault** and implemented **Azure Managed Identity (AD)** for service-to-service authentication.
- **Build System**: Replaced simple requirements with `pyproject.toml` and a comprehensive `Makefile` for enterprise-grade deployment.

### Fixed
- [src.backend]**Token Management**: Resolved issues with conversation overflow by implementing a sliding window memory with automated summaries.
- [src.backend]**Indexing Reliability**: Improved local storage persistence fallback for vector indices when direct store calls fail.
- [src.backend]**Concurrency**: Applied `nest_asyncio` to resolve loop conflicts during nested agentic tool executions.

### Security
- [src.backend]**Input Sanitization**: Implemented prompt-injection moderation patterns in the document summary indexer to redact malicious instructions.
- [src.backend]**Credential Protection**: Removed hardcoded strings in favor of the `CredentialManager` and `DefaultAzureCredential` patterns.

---
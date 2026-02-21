import os
import uuid
from app_logger import setup_logger
import pandas as pd
import asyncio
import nest_asyncio
from typing import Optional, AsyncGenerator, List, Any, Dict
from datetime import datetime, timedelta, timezone
import ast
import re
from io import BytesIO
import json
import tempfile
from dotenv import load_dotenv

from chainlit import LlamaIndexCallbackHandler

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import Agent
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager

from llama_index.experimental.query_engine.pandas import PandasQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import (
	FunctionTool, RetrieverTool, ToolMetadata
)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.core.llms import (
	ChatMessage, TextBlock, MessageRole,
)
from llama_index.core import Settings
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from src.backend.azure_blob_file_retriever import AzureBlobFileRetriever
from src.backend.user_uploaded_file_indexer import UserUploadedFileIndexer
from src.backend.utility import parse_response_sources
from src.backend.ai_models import (
	AIModelTypes,
	MODEL_TOKEN_LIMITS,
	DEFAULT_REASONING_EFFORT
)
from src.backend.config import config
from src.backend.credential_manager import CredentialManager
from src.backend.llm_loader import load_llm, load_embed
from src.backend.index_engine import initialize_index
from src.backend.prompts import (
	AGENTIC_PANDAS_QUERY_ENGINE_RESPONSE_SYNTHESIS_PROMPT,
	AGENTIC_AI_SYSTEM_PROMPT,
	AGENTIC_PANDAS_QUERY_ENGINE_INSTRUCTION_PROMPT,
	AGENTIC_PANDAS_QUERY_ENGINE_PANDAS_PROMPT,
	AGENTIC_AI_CODEX_PROMPT
)

logger, log_filename = setup_logger('agentic_chat_engine')


class AsyncAgenticAiSystem:
	"""
	An asynchronous agentic chat engine for querying CSVs/excel/data files and documents.
	Integrates LlamaIndex, PandasQueryEngine, and FunctionAgent for structured and unstructured data analysis.
	"""

	def __init__(self,\
		selected_model: AIModelTypes = AIModelTypes.GPT51,\
		llm_creativity_level: float = 0.1, \
		similarity_top_k: int = 20,\
		reasoning_effect = 'low',
		index_name: Optional[str] = None,\
		session_id: str = str(uuid.uuid4()),\
		upload_root_dir: str = tempfile.mkdtemp(prefix="llama_index_"),\
		conversation_thread: List = [], \
		blob_bytes : Dict[str, Any] = {"bytes":bytes('', encoding="utf-8"), "metadata": '{}'},
		enable_coding_assistant=False
	):
		"""
		Initialize the AsyncAgenticAiSystem.

		Args:
			index_name (Optional[str]): Name of the index configuration to use. Defaults to environment variable INDEX_NAME or 'salesforce'.
			session_id (str): Unique session ID for memory storage. Defaults to a new UUID.

		Raises:
			ValueError: If index configuration or required settings are missing.
			Exception: For failures in loading models or initializing components.
		"""
		# Environment and Configuration
		self.env = os.getenv("ENVIRONMENT", "local")
		if self.env in ["local", "local_emulator"]:
			load_dotenv(override=True)
		self.selected_model = selected_model
		self.llm_creativity_level = llm_creativity_level
		self.similarity_top_k = similarity_top_k
		effort = "high" if (reasoning_effect == "high" and self.selected_model in DEFAULT_REASONING_EFFORT) else DEFAULT_REASONING_EFFORT.get(self.selected_model, reasoning_effect)
		self.reasoning_effect = { "reasoning_effort":  effort}
		# if self.selected_model in DEFAULT_REASONING_EFFORT:
		# 	self.reasoning_effect = { "reasoning_effort": "high"} if reasoning_effect == "high" else  { "reasoning_effort": DEFAULT_REASONING_EFFORT[self.selected_model]}
		self.index_name = index_name or os.getenv("INDEX_NAME", "aiim")
		logger.info(f"[AgenticAi] Initiating Index (KnowledgeBase): {self.index_name} for {session_id}")
		self.config = config.indexes.get(self.index_name)
		self.token_counter = TokenCountingHandler()
		self.callback_manager = CallbackManager([self.token_counter])

		if not self.config:
			raise ValueError(f"No index configuration found for '{self.index_name}'")

		# Session memory (async-capable)
		self.session_id = session_id
		self.upload_root_dir = upload_root_dir
		self.memory = Memory.from_defaults(session_id=self.session_id, token_limit=MODEL_TOKEN_LIMITS[self.selected_model])
		self.credential_manager = CredentialManager(key_vault_url=self.config.key_vault.get("url"))

		self.conversation_thread = self.set_conversation_thread(conversation_thread)
		# if self.conversation_thread != []:
		# 	self.set_memory(conversation_thread=self.conversation_thread)

		# Load LLM & embed model
		try:
			self.credential = DefaultAzureCredential()
			self.embed = load_embed(index_name=self.index_name, use_azure=True, callback_manager=self.callback_manager)
			if self.embed is None:
				raise ValueError("Failed to load embedding model")
			Settings.embed_model = self.embed
			Settings.callback_manager = CallbackManager([])
			self.llm = load_llm(
				model=AIModelTypes(self.selected_model),
				temperature=self.llm_creativity_level,
				index_name=self.index_name,
				use_azure=True,
				additional_kwargs=self.reasoning_effect,
				callback_manager=self.callback_manager
			)
			if self.llm is None:
				raise ValueError("Failed to load LLM")
			logger.info(f"""[AgenticAi] LLM set successfully for model: {self.selected_model}""")
			Settings.llm = self.llm
		except Exception as e:
			logger.error(f"[AgenticAi] Failed to load model: {str(e)}")
			raise

		# Initialize Vector Index
		try:
			self.index = self.reinitialize_index()
			if self.index is None:
				raise ValueError("Failed to initialize index")
		except Exception as e:
			logger.error(f"[AgenticAi] Failed to initialize index: {str(e)}")
			raise

		self.blob_bytes = blob_bytes
		self.enable_coding_assistant = enable_coding_assistant
		## AgenticAi - Build Tools & Agent
		self.csv_engine = self.__build_csv_engine()
		self.retriever_tool = self.__build_retriever_tool
		self.function_tool = self.__build_function_tool
		# Index User Uploaded Files)
		self.local_file_indexer = UserUploadedFileIndexer(\
			root_dir=self.upload_root_dir, index_name=self.index_name,\
			model=self.selected_model, memory=self.memory,
			similarity_top_k=self.similarity_top_k
		)

		## Actual Ai Assistant
		self.agent = self.__build_agent()

	# ----------------------- Setters ------------------------------- #
	def set_memory(self, conversation_thread):
		if isinstance(conversation_thread, list) and len(conversation_thread) > 0:
			conversation_thread.sort(key=lambda s: s.get("createdAt") or "", reverse=True)
		for step in conversation_thread:
			if step['role'] == 'assistant':
				self.memory.put(
					ChatMessage(
						role=MessageRole.ASSISTANT,
						blocks=[TextBlock(text=step['content'])]
					)
				)
			if step['role'] == 'system':
				self.memory.put(
					ChatMessage(
						role=MessageRole.SYSTEM,
						blocks=[TextBlock(text=step['content'])]
					)
				)
			if step['role'] == 'user':
				self.memory.put(
					ChatMessage(
						role=MessageRole.USER,
						blocks=[TextBlock(text=step['content'])]
					)
				)


	def set_conversation_thread(self, thread):
		now = datetime.now().replace(tzinfo=timezone.utc)
		start_of_today = datetime(now.year, now.month, now.day).replace(tzinfo=timezone.utc)
		start_of_yesterday = start_of_today - timedelta(days=1)

		# Partition thread into past and current
		past_thread = [
			m for m in thread
			if datetime.fromisoformat(m["createdAt"]).replace(tzinfo=timezone.utc) <= start_of_yesterday
		]
		current_thread = [
			m for m in thread
			if datetime.fromisoformat(m["createdAt"]).replace(tzinfo=timezone.utc) > start_of_yesterday
		]

		# Improved branching logic
		if past_thread and current_thread:
			# Summarize past, keep current
			self.conversation_thread = [
				{'role': 'system', 'content': self.__summarize_conversation(past_thread)},
				*current_thread
			]
		elif not past_thread and len(current_thread) > 8:
			# Summarize first 9 messages if only current exists and it's long
			len(current_thread)
			summary = self.__summarize_conversation(int((abs(len(current_thread) * 0.6))))
			self.conversation_thread = [
				{'role': 'system', 'content': summary},
				*current_thread[int((abs(len(current_thread) * 0.6))):]
			]
		else:
			# If thread is short or only past exists, keep as is
			self.conversation_thread = thread

		self.set_memory(conversation_thread=self.conversation_thread)
		return self.conversation_thread

	def set_selected_model(self, selected_model):
		logger.info(f"""[AgenticAi] LLM Model Changed: {selected_model}""")
		self.selected_model = selected_model
		self.llm = load_llm(
				model=AIModelTypes(self.selected_model),
				temperature=self.llm_creativity_level,
				index_name=self.index_name,
				use_azure=True,
				additional_kwargs=self.reasoning_effect,
				callback_manager = self.callback_manager
			)
		Settings.llm = self.llm
		self.local_file_indexer = UserUploadedFileIndexer(\
			root_dir=self.upload_root_dir, index_name=self.index_name,\
			model=AIModelTypes(self.selected_model), memory=self.memory,
			similarity_top_k=self.similarity_top_k
		)
		self.index = self.reinitialize_index()
		self.agent = self.__build_agent()
		return

	def set_embed_model(self):
		logger.info(f"""[AgenticAi] Embed Model set to Index: {self.index_name}""")
		self.embed = load_embed(index_name=self.index_name, use_azure=True, callback_manager=self.callback_manager)
		Settings.embed_model = self.embed
		self.index = self.reinitialize_index()
		self.agent = self.__build_agent()
		return

	def set_llm_creativity_level(self, llm_creativity_level):
		logger.info(f"""[AgenticAi] Changed Creativity Level: {llm_creativity_level}""")
		self.llm_creativity_level = llm_creativity_level
		self.llm = load_llm(
				model=AIModelTypes(self.selected_model),
				temperature=self.llm_creativity_level,
				index_name=self.index_name,
				use_azure=True,
				additional_kwargs=self.reasoning_effect,
				callback_manager = self.callback_manager
			)
		Settings.llm = self.llm
		self.local_file_indexer = UserUploadedFileIndexer(\
			root_dir=self.upload_root_dir, index_name=self.index_name,\
			model=AIModelTypes(self.selected_model), memory=self.memory,
			similarity_top_k=self.similarity_top_k
		)
		self.index = self.reinitialize_index()
		self.agent = self.__build_agent()
		return

	def set_similarity_top_k(self, similarity_top_k):
		logger.info(f"""[AgenticAi] Changed Top K: {similarity_top_k} """)
		self.similarity_top_k = similarity_top_k
		self.local_file_indexer = UserUploadedFileIndexer(\
			root_dir=self.upload_root_dir, index_name=self.index_name,\
			model=AIModelTypes(self.selected_model), memory=self.memory,
			similarity_top_k=self.similarity_top_k
		)
		self.index = self.reinitialize_index()
		self.agent = self.__build_agent()
		return

	def set_reasoning_effect(self, reasoning_effect):
		model_kwargs = {}
		logger.info(f"""[AgenticAi] Changed Reasoning Effect: {reasoning_effect} """)

		if self.selected_model in DEFAULT_REASONING_EFFORT:
			if "verbosity" in self.reasoning_effect:
				self.reasoning_effect.pop("verbosity")
			model_kwargs = { "reasoning_effort": reasoning_effect or DEFAULT_REASONING_EFFORT[self.selected_model]}
		if self.selected_model == AIModelTypes.GPT51:
			model_kwargs = { "reasoning_effort": "none", "verbosity": "high" if reasoning_effect == "high" else "low" }
		self.reasoning_effect = model_kwargs
		self.llm = load_llm(
				model=AIModelTypes(self.selected_model),
				temperature=self.llm_creativity_level,
				index_name=self.index_name,
				use_azure=True,
				additional_kwargs=model_kwargs,
				callback_manager = self.callback_manager
			)
		Settings.llm = self.llm
		self.index = self.reinitialize_index()
		self.agent = self.__build_agent()
		return

	def set_index_name(self, index_name):
		logger.info(f"""[AgenticAi] Changed Index Name: {index_name} """)
		self.index_name = index_name
		self.config = config.indexes.get(index_name)
		self.llm = load_llm(
				model=AIModelTypes(self.selected_model),
				temperature=self.llm_creativity_level,
				index_name=self.index_name,
				use_azure=True,
				additional_kwargs=self.reasoning_effect,
				callback_manager = self.callback_manager
			)
		self.index = self.reinitialize_index()
		self.set_selected_model(self.selected_model)
		self.set_embed_model()
		if self.index_name == 'capitalraising':
			self.csv_engine = self.__build_csv_engine()
		self.agent = self.__build_agent()
		return

	def set_coding_assistant(self, enable_coding_assistant=False):
		self.enable_coding_assistant=enable_coding_assistant
		self.agent = self.__build_agent()
		return

	def reinitialize_index(self):
		logger.info(f"""[AgenticAi] initialized with search_index: `{self.config.azure_ai_search.get("index_name")}` of IndexName: {self.index_name} """)
		return initialize_index(
				search_index_name=self.config.azure_ai_search.get("index_name"),
				llm=self.llm,
				embed_model=self.embed,
				embed_size=self.config.embed.get("size"),
				search_service_endpoint=self.config.azure_ai_search.get("search_service_endpoint"),
				search_service_credential=self.credential,
				old_index=False,
				aio=True,
				rag_spliter=self.config.rag.get("spliter"),
				rag_chunk_size=self.config.rag.get("chunk_size"),
				rag_chunk_overlap=self.config.rag.get("chunk_overlap"),
			)
	# ----------------------- Setters ------------------------------- #

	# LLM Output Processore for user friendly output
	# Code-Block to return preformatted code block which can be copied.
	def __safe_output_processor(self, output: str) -> Any:
		"""
		Determines if the model's output is valid Python code or just plain text.
		Executes and returns result if it's code; otherwise returns the text.
		"""
		output = output.strip()

		# Remove any markdown-style code blocks like ```python
		if output.startswith("```"):
			output = re.sub(r"^```(?:python)?", "", output).strip()
			output = re.sub(r"```$", "", output).strip()

		if self.__is_valid_python_code(output):
			try:
				local_vars = {}
				exec(output, {}, local_vars)
				return local_vars
			except Exception as e:
				return f"⚠️ Failed to execute code: {str(e)}"
		else:
			# It's just a plain text/natural language answer
			return output

	# Python Code Validator
	def __is_valid_python_code(self, code: str) -> bool:
		try:
			ast.parse(code)
			return True
		except SyntaxError:
			return False

	def __dummy_function(self, *args, **kwargs):
		# Log or return a message that indicates the tool was bypassed.
		logger.info("[AgenticAi] Dummy function invoked: Tool not available")
		return {"status": "bypassed", "message": "Tool not available, bypassed."}

	def __build_function_tool(self, fn, name: str, description: str) -> FunctionTool:
		"""
		Dynamically construct a FunctionTool from a provided function.

		This tool wraps a callable function (e.g., for uploading and indexing files),
		allowing the agent to perform specific tasks.

		Args:
			fn (Callable): The function to wrap in the tool.
			name (str): The name of the tool.
			description (str): A description of what the tool does.

		Returns:
			FunctionTool: A callable tool usable by the agent.

		Raises:
			Exception: If tool initialization fails.
		"""
		try:
			tool = FunctionTool.from_defaults(
				fn=fn,
				name=name,
				description=description,
			)
			logger.info(f"[AgenticAi] FunctionTool Created: '{name}'")
			return tool
		except Exception as e:
			logger.error(f"[AgenticAi] FunctionTool Exception: 'Tool {name}': {str(e)}")
			tool = FunctionTool.from_defaults(
				fn=self.__dummy_function,
				name=name,
				description="Dummy tool to bypass actual function call",
			)
			return tool

	def __build_retriever_tool(self, retriever, name: str, description: str) -> RetrieverTool:
		"""
		Dynamically construct a FunctionTool for querying unstructured data.

		This tool wraps a provided retriever, enabling the agent to access and reason over
		unstructured document content, such as PDF chunks stored in a vector store.

		Args:
			retriever: The retriever instance to be wrapped by the tool.
			name (str): The name of the tool.
			description (str): A description of what the tool does.

		Returns:
			FunctionTool: A callable tool usable by the agent to answer queries based on unstructured document data.

		Raises:
			Exception: If the tool initialization fails.
		"""
		try:
			tool = RetrieverTool(
				retriever=retriever,
				metadata=ToolMetadata(
					name=name,
					description=description,
				),
			)
			logger.info(f"[AgenticAi] RetrieverTool Created: '{name}'")
			return tool
		except Exception as e:
			logger.error(f"[AgenticAi] RetrieverTool Exception: 'Tool {name}': {str(e)}")
			tool = RetrieverTool(
				retriever=self.__dummy_function,
				metadata=ToolMetadata(
					name=name,
					description="Dummy tool to bypass actual function call",
				),
			)
			return tool

	def __build_csv_engine(self) -> PandasQueryEngine:
		"""
		Build a PandasQueryEngine for querying CSV data with simplified prompt templates and dynamic column info.

		Returns:
			PandasQueryEngine: Configured query engine.

		Raises:
			Exception: If metadata loading or engine creation fails.
		"""
		df, meta = self.load_csv_file(self.blob_bytes['bytes'], self.blob_bytes['metadata'])
		# Get dynamic column information from actual DataFrame with date columns highlighted
		column_info_str = f"""
			Columns ({len(df.columns)} total): {', '.join(df.columns.tolist())}
			Data types: {dict(df.dtypes)}
			DataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns

			**Date Columns:**
				- activitydate: Meeting occurrence date (datetime, already parsed)
				- createddate: Record creation date (datetime, already parsed)
		"""
		# Create DataFrame info string combining structure and column details
		df_info = f"""
			{df.head(5).to_string()}
			{column_info_str}
		"""
		pandas_prompt = PromptTemplate(
			template=AGENTIC_PANDAS_QUERY_ENGINE_PANDAS_PROMPT, \
			metadata=json.loads(meta) if isinstance(meta, str) else meta\
		).partial_format(
			df_str=df.head(5).to_string(), metadata_str=meta, column_info=column_info_str,
			instruction_str=AGENTIC_PANDAS_QUERY_ENGINE_INSTRUCTION_PROMPT.format(
				df_info=df_info, metadata_str=meta
			),
		)
		response_prompt = PromptTemplate(AGENTIC_PANDAS_QUERY_ENGINE_RESPONSE_SYNTHESIS_PROMPT)
		try:
			engine = PandasQueryEngine(
				df=df,
				instruction_str=AGENTIC_PANDAS_QUERY_ENGINE_INSTRUCTION_PROMPT.format(
					df_info=df_info, metadata_str=meta
				),
				pandas_prompt=pandas_prompt,
				response_synthesis_prompt=response_prompt,
			)
			logger.info("CSV query engine created successfully with prompts")
			return engine
		except Exception as e:
			logger.error(f"Failed to create CSV query engine: {str(e)}")
			raise

	def load_csv_file(self, csv_file_bytes_content: bytes, metadata: Any = {'description': ''}) -> pd.DataFrame:
		"""
		Method to Load CSV data from blob container.

		Returns:
			pd.DataFrame: Loaded DataFrame with parsed dates.

		Raises:
			FileNotFoundError: If the CSV file is not found.
			Exception: For other data loading errors.
		"""
		try:
			df = pd.read_csv(
				BytesIO(csv_file_bytes_content),
				encoding="latin1",
				parse_dates=["createddate", "activitydate"],
				low_memory=False,
			)
			logger.info(f"[AgenticAi] CSV File Loaded, Shape[Row, Column] {df.shape}")
			return df, metadata
		except Exception as e:
			logger.error(f"[AgenticAi] CSV Exception: {str(e)}")
			raise

	def query_local_file_index(self, query: str) -> str:
		"""
		Queries the local file index using a citation-based chat engine.
		Args:
			query (str): The user input or question to be processed.
		Returns:
			str: The response from the chat engine, or an error message if the query fails.
		Raises:
			Exception: If an error occurs while creating the chat engine or processing the query.
		"""
		try:
			chat_engine = self.local_file_indexer.create_local_citation_chat_engine()
			response = chat_engine.chat(query)  # or stream_chat, depending on your implementation
			return response.response
		except Exception as e:
			logger.error(f"[AgenticAi] User File Query Error: {e}")
			return f"Error querying local file index: {e}"

	async def upload_and_index_files(self, uploaded_files: List) -> dict[Any, Any]:
		"""
		Accepts files to upload and indexes them using UserUploadedFileIndexer[index_uploaded_files].
		Returns:
			Dictionary[FileName: Summary of File]
		Raises:
			Exception: If agent creation fails.
		"""
		try:
			result = await self.local_file_indexer.index_uploaded_files(file_list=uploaded_files)
			return result
		except Exception as e:
			logger.error(f"[AgenticAi] ")
			return f"Failed to index uploaded files: {e}"

	def bing_grounding_tool(self, query) -> Any:
		"""
		Executes a query using Azure AI Foundry GenAI Project Agent and retrieves the first response message.

		Args:
			query (str): The user query to be sent to the Azure GenAI Agent.
		Returns:
			Any: The first message object from the agent's response thread, basically an assistant response.
		Raises:
			Exception: If any part of the agent interaction fails.
		"""
		try:
			project_endpoint = self.config.ai_service.get('endpoint')
			agent_id = self.config.ai_service.get('agent_id')
			azure_project_client = AIProjectClient(
				endpoint=project_endpoint, credential=DefaultAzureCredential()
			)
			azure_agent = azure_project_client.agents.get_agent(agent_id=agent_id)
			thread = azure_project_client.agents.threads.create()
			message = azure_project_client.agents.messages.create(
				thread_id=thread.id,
				role="user",
				content=query
			)
			logger.info(f"[AgenticAi] BingGroundingTool: User Query Processed: {query}")
			run = azure_project_client.agents.runs.create_and_process(
				thread_id=thread.id,
				agent_id=azure_agent.id,
			)
			messages = azure_project_client.agents.messages.list(thread_id=thread.id)
			message_list = list(messages)
			return message_list[0]
		except Exception as e:
			logger.error(f"[AgenticAi] BingGroundingTool Exception: {str(e)}")

	def __build_agent(self) -> FunctionAgent:
		"""
		Build a FunctionAgent with GenAI Agents.
		Args:
			-
		Returns:
			FunctionAgent: Configured agent.
		Raises:
			Exception: If agent creation fails.
		"""
		try:
			im_retriever_tool = self.__build_retriever_tool(
				retriever=self.index.as_retriever(
					vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
					similarity_top_k=self.similarity_top_k,
				),
				name='im_retriever_tool',
				description="Tool for querying unstructured data from documents, specifically PDF chunks.",
			)

			upload_and_index_user_file_tool = self.__build_function_tool(
				fn=self.upload_and_index_files,
				name='upload_and_index_user_file_tool',
				description="Upload files and create an index for querying so that their content can be queried later."
			)
			query_user_upload_file_indexes_tool = self.__build_function_tool(
				fn=self.query_local_file_index,
				name='query_user_upload_file_indexes_tool',
				description="Query the index created from uploaded files or Query the content of previously uploaded and indexed files."
			)
			internet_search_tool = self.__build_function_tool(
				fn=self.bing_grounding_tool,
				name='internet_search_tool',
				description="Tool to perform Bing web search (Internet Search) via Azure AI agent."
			)
			agent_system_prompt = AGENTIC_AI_SYSTEM_PROMPT.format(\
					now_str=datetime.now().strftime("%Y-%m-%d"),
				)
			if self.enable_coding_assistant == True:
				agent_system_prompt = AGENTIC_AI_CODEX_PROMPT.format(\
					now_str=datetime.now().strftime("%Y-%m-%d"),
				)
			agent = FunctionAgent(
				tools=[im_retriever_tool, upload_and_index_user_file_tool, query_user_upload_file_indexes_tool, internet_search_tool ],
				llm=self.llm,
				system_prompt=agent_system_prompt,
				verbose=True
			)
			if self.index_name == 'capitalraising':
				csv_tool = self.__build_function_tool(\
					fn=lambda q: str(self.csv_engine.query(q)),
					name='csv_tool',
					description="Query Salesforce meeting data CSV table."
					)
				agent.tools.append(csv_tool)

			logger.info(f"[AgenticAi] FunctionAgent Created with tools")
			return agent
		except Exception as e:
			logger.error(f"[AgenticAi] FunctionAgent Exception: {str(e)}")
			raise

	# ----------------------- Agent Executors ------------------------------- #
	# Use run_agent_async if execution is synchronous as per the projects environment,
	# but asynchronous behaviour is needed
	async def run_agent_async(self, question: str) -> str:
		"""
		Run the agent asynchronously with a user query.
		Args:
			question (str): The user query to process.
		Returns:
			str: The agent's response.
		Raises:
			Exception: If the agent fails to process the query.
		"""
		try:
			logger.info(f"[AgenticAi] Current SessionId: {self.session_id}")
			# User message
			self.memory.put(
				ChatMessage(
					role=MessageRole.USER,
					blocks=[TextBlock(text=question)]
				)
			)
			logger.info(f"\n [AgenticAi] User Query: {question}")
			# Assistant message
			response = await self.agent.run(user_msg=question, chat_history= list(self.memory.get_all()))
			if 'tool_calls' in response:
				for tool in response['tool_calls']:
					logger.info(f"\n [AgenticAi] Agent Responding tool: {str(tool.tool_name)}")
			logger.info(f"[AgenticAi] Agent Response for Query: {response}")
			self.memory.put(
				ChatMessage(
					role=MessageRole.ASSISTANT,
					blocks=[TextBlock(text=response.response.blocks[0].text)]
				)
			)
			return response
		except Exception as e:
			logger.error(f"[AgenticAi] Agent Exception: {str(e)}")
			raise

	# Use run_agent if asynchronous execution is available
	async def run_agent(self, question: str) -> str:
		"""
		Run the agent with a user query, applying nested asyncio for compatibility.

		Args:
			question (str): The user query to process.

		Returns:
			str: The agent's response.

		Raises:
			Exception: If the agent fails to process the query.
		"""
		nest_asyncio.apply()
		return await self.run_agent_async(question)

	# For Streaming response on Synchronous environment
	async def stream_response(self, response):
		"""Helper function to stream response text from the async agentic engine."""
		if hasattr(response, 'response') and hasattr(response.response, 'blocks'):
			for chunk in response.response.blocks[0].text:
				yield chunk
		else:
			yield str(response)

	# For Streaming responses in chunk and prepare final response asynchronously
	async def collect_async_generator_result(self, gen: AsyncGenerator[str, None]) -> str:
		"""Consume an async generator and return the combined output as a single string."""
		result = []
		async for chunk in gen:
			result.append(chunk)
		return "".join(result)

	def get_retriever_metadata(self, response_block):
		retrieved_metadata = []
		try:
			for tool_chunk  in response_block.tool_calls:
				if 'im_retriever_tool' == tool_chunk.tool_name:
					retrieved_metadata = parse_response_sources(response_sources= tool_chunk.tool_output.raw_output)
			return json.loads(retrieved_metadata)
		except:
			return retrieved_metadata

	# Get Final response if Running on non-async context
	def get_response_async(self, question: str) -> str:
		response_block = asyncio.run(self.run_agent_async(question))
		response_stream = self.stream_response(response_block)
		response_metadata = self.get_retriever_metadata(response_block)
		response_text = asyncio.run(self.collect_async_generator_result(response_stream))
		return {'response_text': response_text, 'response_metadata': response_metadata}

	# Get Final response if Running on async context
	async def get_response(self, question: str) -> str:
		response_block = await self.run_agent_async(question)
		response_stream = await self.stream_response(response_block)
		response_metadata = self.get_retriever_metadata(response_block)
		response_text = await self.collect_async_generator_result(response_stream)
		return {'response_text': response_text, 'response_metadata': response_metadata}

	def generate_thread_title(self) -> str:
		"""
		Generate a thread title based on the first question and its answer.

		Args:
			self (cls)

		Returns:
			str: A short, descriptive title.
		"""
		first_question = ''
		first_answer = ''
		for message in self.memory.get_all():
			if message.role == MessageRole.USER and first_question == '':
				first_question = message.content or ''
			elif message.role == MessageRole.ASSISTANT and first_answer == '':
				first_answer = message.content or ''
		conversation_text = f"User: {first_question}\nAssistant: {first_answer}"
		# title = Settings.llm
		title = load_llm(model=AIModelTypes.GPT41_MINI,\
				index_name=self.index_name,\
				use_azure=True,\
				callback_manager = self.callback_manager
			).complete(\
				f'Give me a Title based on the conversation (Maximum 8 words): {conversation_text}'\
			)
		return title.text

	def __summarize_conversation(self, messages: list = [], start_idx: int = 0, end_idx: int = None) -> str:
		"""
		Summarize a set of conversation messages to reduce token length.

		Args:
			self (cls)
			start_idx (int): Start index of messages to summarize.
			end_idx (int): End index of messages to summarize. If None, summarize to the end.

		Returns:
			str: A concise summary of the selected conversation messages.
		"""
		if end_idx is None:
			end_idx = len(messages)
		conversation_snippets = []
		for msg in messages[start_idx:end_idx]:
			role = "User" if msg['role'] == MessageRole.USER else "Assistant"
			conversation_snippets.append(f"{role}: {msg['content']}")
		conversation_text = "\n".join(conversation_snippets)
		summary_prompt = (
			f"Summarize the following conversation, "
			f"preserving key context and decisions:\n{conversation_text}"
		)
		summary = load_llm(
			model=AIModelTypes.GPT41_MINI,
			index_name=self.index_name,
			use_azure=True,
			callback_manager = self.callback_manager
		).complete(summary_prompt)
		return summary.text

	def get_token_counts(self):
		return {
			"Prompt tokens": f"{self.token_counter.prompt_llm_token_count/MODEL_TOKEN_LIMITS[self.selected_model]:.2f}M",
			"Completion tokens":f" {self.token_counter.completion_llm_token_count/MODEL_TOKEN_LIMITS[self.selected_model]:.2f}M",
			"Total tokens": f"{self.token_counter.total_llm_token_count/MODEL_TOKEN_LIMITS[self.selected_model]:.2f}M",
		}


# # Example usage
# if __name__ == "__main__":
#     async def main():
#         try:
#             engine = AsyncAgenticAiSystem()
#             question = "What happened in the last 6 months for all the data?"
#             response = await engine.run_agent(question)
#             print(f"Response: {response}")
#         except Exception as e:
#             print(f"Error: {str(e)}")

#     asyncio.run(main())
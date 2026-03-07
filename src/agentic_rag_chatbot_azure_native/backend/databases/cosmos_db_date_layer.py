import datetime
import asyncio
import uuid
from typing import Optional, Dict, List, Any
from src.backend.credential_manager import CredentialManager
from azure.cosmos import CosmosClient
from chainlit import User, PersistedUser
from chainlit.data.base import BaseDataLayer
from chainlit.data.utils import queue_until_user_message
from chainlit.element import ElementDict
from chainlit.step import StepDict

from chainlit.types import (
	Feedback,
	PageInfo,
	ThreadDict,
	Pagination,
	ThreadFilter,
	PaginatedResponse
)
#* #
# Thread - A full conversation session 
# Step - A single message/action in the thread (One message in a conversation)
# Element - Visual or interactive component in a step
# *#
TTL_30_DAYS = 2592000
class CosmosDBDataLayer(BaseDataLayer):
	"""
	A Chainlit-compatible data layer implementation using Azure Cosmos DB as the backend storage.

	This class provides persistence for users, threads, steps (messages), elements (UI components),
	and feedback using Cosmos DB's SQL API. It integrates seamlessly with the Chainlit framework to
	support threaded conversations and UI interactions.

	Attributes:
		client (CosmosClient): Azure Cosmos DB client initialized with the provided credentials.
		db: Reference to the specified Cosmos DB database.
		container: Reference to the specified Cosmos DB container.

	Methods:
		get_user(identifier: str) -> Optional[PersistedUser]:
			Retrieve a user by their identifier from Cosmos DB.

		create_user(user: User) -> PersistedUser:
			Create a new user record in Cosmos DB.

		delete_user_session(id: str) -> bool:
			Stubbed method to delete a user session. Returns True by default.

		upsert_feedback(feedback: Feedback) -> str:
			Insert or update a feedback record in Cosmos DB.

		delete_feedback(feedback_id: str) -> bool:
			Stubbed method to delete feedback. Returns True by default.

		create_element(element_dict: ElementDict) -> None:
			Persist a visual or interactive element related to a step.

		get_element(thread_id: str, element_id: str) -> Optional[ElementDict]:
			Retrieve an element by thread and element ID.

		delete_element(element_id: str) -> None:
			Delete an element from Cosmos DB.

		create_step(step_dict: StepDict) -> None:
			Persist a step (message) in Cosmos DB.

		update_step(step_dict: StepDict) -> None:
			Update an existing step with new metadata or content.

		delete_step(step_id: str) -> None:
			Delete a step from Cosmos DB.

		get_thread_author(thread_id: str) -> Optional[str]:
			Retrieve the author (user ID) of a thread.

		get_thread(thread_id: str) -> Optional[ThreadDict]:
			Fetch the metadata of a thread by its ID.

		update_thread(thread_id: str, name: Optional[str], userId: Optional[str],
					metadata: Optional[Dict], tags: Optional[List[str]]) -> None:
			Update the thread's metadata, name, tags, or associated user.

		delete_thread(thread_id: str) -> None:
			Delete a thread from Cosmos DB.

		list_threads(pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse[ThreadDict]:
			Return a paginated list of threads filtered by user or metadata.

		build_debug_url() -> Optional[str]:
			Returns an optional debug URL. Returns an empty string in this implementation.

	Note:
		- All Cosmos DB operations are executed asynchronously using asyncio.to_thread() 
		to avoid blocking the event loop.
		- This implementation assumes that all stored items have a "type" field used 
		to distinguish between entities like 'user', 'thread', 'step', 'element', etc.
		- Timestamps are added during creation and update for audit purposes.
	"""
	def __init__(self,\
		credential: CredentialManager,\
		url: str,\
		database_id: str,\
		container_id: str,\
		partition_key_field: str = "partition_thread_id", \
	):
		self.client = CosmosClient(credential=credential, url=url)
		self.db = self.client.get_database_client(database_id)
		self.container = self.db.get_container_client(container_id)
		self.partition_key_field = partition_key_field
		self.user_identity = 'local_user'
		self.user_id = str(uuid.uuid4())
		self.message_step_types = ['step', 'user_message', 'assistant_message',\
			"run", "tool", "llm", "embedding", "retrieval", "rerank", "undefined"]
		self.element_types = [ "image", "text", "pdf", "tasklist", "audio", "video",
							"file", "plotly", "dataframe", "custom"]
		self.indexing_policy = {
			"indexingMode": "consistent",
			"includedPaths": [
				{ "path": "/type/?" },
				{ "path": "/userId/?" },
				{ "path": "/name/?" },
				{ "path": "/createdAt/?" }
			],
			"excludedPaths": [
				{ "path": "/*" },
				{ "path": "/\"_etag\"/?" }
			]
		}
	def _prepare_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
		"""Ensures the item has the correct partition key before persistence."""
		# Use threadId if available, otherwise fall back to the item's own ID 
		# (common for the 'thread' object itself or 'user' objects)
		pk_value = item.get("threadId") or item.get("id")
		item[self.partition_key_field] = pk_value
		return item

	def _timestamp(self):
		return datetime.datetime.now().isoformat()

	def __strip_cosmos_meta(self, obj: Dict[str, Any]) -> Dict[str, Any]:  
		for cosmos_key in ("_rid", "_self", "_etag", "_attachments", "_ts"):  
			obj.pop(cosmos_key, None)  
		return obj  
	# ----------------------
	# USER METHODS
	# ----------------------
	async def get_user(self, identifier: str):
		query = f"SELECT * FROM c WHERE c.type = 'user' AND c.identifier = '{identifier}'"
		items = await asyncio.to_thread(
			lambda: list(self.container.query_items(query=query, enable_cross_partition_query=True))
		)
		row = items[0] if items else None
		if row is not None:
			self.user_identity = str(row.get("identifier"))
			self.user_id = str(row.get("id"))
			return PersistedUser(
				id=str(row.get("id")),
				display_name=str(row['displayName']),
				identifier=str(row.get("identifier")),
				createdAt=row.get("createdAt"),
				metadata=row.get("metadata", "{}")
			)
		return None
	

	async def create_user(self, user: User):
		existing_user = await self.get_user(user.identifier)

		# --- Update Existing User ---
		if existing_user and isinstance(existing_user, User):
			self.user_identity = str(existing_user.identifier)
			self.user_id = str(existing_user.id)

			# Update metadata groups safely
			if existing_user.metadata is None:
				existing_user.metadata = {}
			existing_user.metadata['groups'] = user.metadata.get('groups')

			updated_item = {
				"id": existing_user.id,
				"identifier": existing_user.identifier,
				"type": "user",
				"displayName": existing_user.display_name,
				"metadata": existing_user.metadata,
				"createdAt": existing_user.createdAt,
				"updatedAt": existing_user.createdAt or self._timestamp(),
			}
			await asyncio.to_thread(lambda: self.container.upsert_item(updated_item))
			return PersistedUser(
				id=str(existing_user.id),
				display_name=str(existing_user.display_name),
				identifier=str(existing_user.identifier),
				createdAt=existing_user.createdAt,
				metadata=existing_user.metadata or {}
			)

		# --- Create New User ---
		now = self._timestamp()
		display_name = (
			user.metadata.get("claims", {}).get("displayName")
			if user.metadata else None
		)

		new_user = {
			"id": str(uuid.uuid4()),
			"identifier": user.identifier,
			"type": "user",
			"displayName": display_name,
			"metadata": user.metadata or {},
			"createdAt": now,
			"updatedAt": now,
		}

		self.user_identity = str(user.identifier)
		self.user_id = new_user["id"]

		await asyncio.to_thread(lambda: self.container.create_item(new_user))

		return PersistedUser(
			id=new_user["id"],
			display_name=new_user["displayName"],
			identifier=new_user["identifier"],
			createdAt=new_user["createdAt"],
			metadata=new_user["metadata"]
		)


	
	async def delete_user_session(self, id: str) -> bool:
		try:
			await asyncio.to_thread(lambda: self.container.delete_item(id, partition_key=self.partition_key_field))
			return True
		except Exception:
			return False

	# ----------------------
	# FEEDBACK METHODS
	# ----------------------
	async def get_feedback(self, step_id: str) -> List[Feedback]:
		"""
		Retrieve the feedback object belonging to a particular step from CosmosDB.
		"""
		query = f"""
			SELECT * FROM c WHERE c.type = 'feedback' AND c.forId = '{step_id}'
		"""
		items = await asyncio.to_thread(
			lambda: list(self.container.query_items(query=query, enable_cross_partition_query=True))
		)

		if not items:
			return None

		return self.__strip_cosmos_meta(items[0]) if items else None

	
	async def upsert_feedback(self, feedback: Feedback) -> str:
		existing_feedback = await self.get_feedback(feedback.forId)
		if existing_feedback:
			feedback_dict = {
				**existing_feedback,
				'comment': feedback.comment,
				"updatedAt": self._timestamp(),
			}
		else:
			feedback_dict = {
				**feedback.__dict__,
				"id": feedback.id or str(uuid.uuid4()),
				"type": "feedback",
				"userId": self.user_id,
				"updatedAt": self._timestamp(),
			}
		if self.partition_key_field not in feedback_dict:  
			feedback_dict[self.partition_key_field] = feedback.threadId or feedback_dict["id"]
		await asyncio.to_thread(lambda: self.container.upsert_item(feedback_dict))
		return feedback.id


	async def delete_feedback(self, feedback_id: str) -> bool:
		try:
			query = f"""SELECT * FROM c WHERE c.type = 'feedback' AND c.id = '{feedback_id}'"""
			items = await asyncio.to_thread(
				lambda: list(self.container.query_items(query=query, enable_cross_partition_query=True))
			)
			for item in items:
				await asyncio.to_thread(lambda: self.container.delete_item(item, partition_key=item[self.partition_key_field]))
			return True
		except Exception as e:
			print(e)
			return False


	# ----------------------
	# ELEMENT METHODS
	# ----------------------
	@queue_until_user_message()
	async def create_element(self, element_dict: ElementDict):
		element = element_dict.to_dict()
		element.setdefault("type", "element")
		element.setdefault("createdAt", self._timestamp())
		if self.partition_key_field not in element:  
			element[self.partition_key_field] = element.get("threadId") or element.get("id")
		await asyncio.to_thread(lambda: self.container.upsert_item(body=dict(element)))

	
	@queue_until_user_message()
	async def get_element(self, thread_id: str, element_id: str) -> Optional[ElementDict]:
		types_str = ", ".join([f"'{t}'" for t in self.element_types])
		query = f"""
		SELECT * FROM c WHERE c.type IN ({types_str}) AND c.threadId = '{thread_id}' AND c.id = '{element_id}'
		"""
		items = await asyncio.to_thread(
			lambda: list(self.container.query_items(query=query, enable_cross_partition_query=True))
		)
		return self.__strip_cosmos_meta(items[0]) if items else None


	@queue_until_user_message()
	async def delete_element(self, element_id: str, thread_id: str):
		try:
			items = await self.get_element(thread_id=thread_id, element_id=element_id)
			for item in items:
				await asyncio.to_thread(lambda: self.container.delete_item(item, partition_key=item[self.partition_key_field]))
			return True
		except Exception as e:
			print(e)
			return False

	# ----------------------
	# STEP METHODS
	# ----------------------
	async def get_steps(self, step_id: str) -> List[StepDict]:
		"""
		Retrieve the StepDict List from CosmosDB.
		"""
		query = f"""
			SELECT * FROM c WHERE c.id = '{step_id}'
		"""
		items = await asyncio.to_thread(
			lambda: list(self.container.query_items(query=query, enable_cross_partition_query=True))
		)

		if not items:
			return None
		items_list = [self.__strip_cosmos_meta(item) for item in items]
		return  items_list


	@queue_until_user_message()
	async def create_step(self, step_dict: StepDict):
		item = dict(step_dict)
		item.setdefault("type", "step")
		item.setdefault("createdAt", self._timestamp())
		item.setdefault("ttl", TTL_30_DAYS)
		if self.partition_key_field not in item:
			item[self.partition_key_field] = item.get("threadId") or item.get("id")
		await asyncio.to_thread(lambda: self.container.upsert_item(item))


	@queue_until_user_message()  
	async def update_step(self, step_dict: StepDict):  
		item = dict(step_dict)  
		item.setdefault("type", "step")  
		item.setdefault("updatedAt", self._timestamp())
		if self.partition_key_field not in item:  
			item[self.partition_key_field] = item.get("threadId") or item.get("id")
		if not item.get("id"):  
			print("Warning: update_step called without id")  
			return 
		await asyncio.to_thread(lambda: self.container.upsert_item(item))  


	@queue_until_user_message()
	async def delete_step(self, step_id: str, thread_id: str = None):
		try:
			# If we have the thread_id, we can delete instantly without a query
			if thread_id:
				await asyncio.to_thread(
					lambda: self.container.delete_item(
						item=step_id, 
						partition_key=thread_id
					)
				)
			else:
				# Fallback: Find it first (High RU cost)
				items = await self.get_steps(step_id)
				for item in items:
					await asyncio.to_thread(
						lambda: self.container.delete_item(
							item=item['id'], 
							partition_key=item[self.partition_key_field]
						)
					)
			return True
		except Exception as e:
			print(f"Delete failed: {e}")
			return False
		try:
			items = await self.get_steps(step_id)
			for item in items:
				await asyncio.to_thread(self.container.delete_item,item['id'], partition_key=item[self.partition_key_field])
			return True
		except Exception as e:
			print(e)
			return False

	# # ----------------------
	# # THREAD METHODS
	# # ----------------------
	async def get_thread_author(self, thread_id: str) -> Optional[str]:
		"""
		Get the userId of the author of a given thread.
		"""
		query = f"""
		SELECT c.userId, c.userIdentifier FROM c 
		WHERE c.type = 'thread' AND c.id = '{thread_id}'
		"""

		items = await asyncio.to_thread(
			lambda: list(self.container.query_items(query=query, enable_cross_partition_query=True))
		)

		if not items:
			return None

		return items[0].get("userIdentifier")


	async def delete_thread(self, thread_id: str):
		thread = await self.get_thread(thread_id)
		thread['isActive'] = False
		await asyncio.to_thread(lambda: self.container.upsert_item(thread))

	async def hard_delete_thread(self, thread_id: str):
		"""
		Performs a hard delete of all items associated with a thread ID
		to ensure no orphaned data remains in Cosmos DB.
		"""
		try:
			# 1. Query for everything sharing this threadId (steps, elements, feedback)
			# OR the thread object itself (where id = thread_id)
			query = f"SELECT c.id, c.{self.partition_key_field} FROM c WHERE c.threadId = '{thread_id}' OR c.id = '{thread_id}'"
			
			items_to_delete = await asyncio.to_thread(
				lambda: list(self.container.query_items(query=query, enable_cross_partition_query=True))
			)

			# 2. Batch delete (Cosmos DB Python SDK requires individual deletes)
			for item in items_to_delete:
				await asyncio.to_thread(
					lambda: self.container.delete_item(
						item=item['id'], 
						partition_key=item[self.partition_key_field]
					)
				)
			return True
		except Exception as e:
			print(f"Error during hard delete of thread {thread_id}: {e}")
			return False

	async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
		"""
		Retrieve a thread and all its associated steps and elements from CosmosDB.
		"""
		query = f"""
		SELECT * FROM c WHERE (c.threadId = '{thread_id}' OR c.id = '{thread_id}') 
			AND (c.type != 'thread' OR c.isActive = true)
		"""
		
		items = await asyncio.to_thread(
			lambda: list(self.container.query_items(query=query, enable_cross_partition_query=True, partition_key=thread_id))
		)

		if not items:
			return None

		thread_data = None
		steps = []
		elements = []

		for item in items:
			if item["type"] == "thread":
				thread_data = self.__strip_cosmos_meta(item)
			elif item["type"] in self.message_step_types:
				item['feedback'] = await self.get_feedback(item['id'])
				steps.append(self.__strip_cosmos_meta(item))
			elif item["type"] in self.element_types:
				elements.append(self.__strip_cosmos_meta(item))

		if not thread_data:
			return None
		steps.sort(key=lambda s: s.get("createdAt") or "")

		thread_data.setdefault("steps", steps)
		thread_data.setdefault("elements", elements)
		if isinstance(thread_data['steps'], list):
			thread_data["steps"]= steps
		if isinstance(thread_data['elements'], list):
			thread_data["elements"] = elements
		return thread_data


	async def update_thread(
		self,
		thread_id: str,
		name: Optional[str] = None,
		user_id: Optional[str] = None,
		metadata: Optional[Dict] = None,
		tags: Optional[List[str]] = None
	):
		"""
		Create or update a thread in CosmosDB.
		"""
		thread = await self.get_thread(thread_id)
		
		if not thread:
			# Create a new thread if it doesn't exist
			thread = {
				"id": thread_id,
				"type": "thread",
				"createdAt": self._timestamp(),
				"updatedAt": self._timestamp(),
				"isActive" : True
			}

		if name:
			thread["name"] = name
		else:
			if ('type' in thread and thread['type'] == 'thread') and 'name' not in thread:
				thread["name"] = f"{self._timestamp()}"
		if metadata and metadata.get("is_guest") == True:
			thread["ttl"] = TTL_30_DAYS
		thread["userId"] = user_id or thread.get("userId", self.user_id)  
		thread["userIdentifier"] = thread.get("userIdentifier", self.user_identity)
		
		if metadata:
			thread["metadata"] = metadata
		if tags:
			thread["tags"] = tags
		if self.partition_key_field not in thread:  
			thread[self.partition_key_field] = thread.get("threadId") or thread.get("id")

		return await asyncio.to_thread(lambda: self.container.upsert_item(thread))


	async def list_threads(
		self, pagination: Pagination, filters: ThreadFilter
	) -> PaginatedResponse[ThreadDict]:
		"""
		List threads filtered by userId and optionally paginated.
		"""
		query = "SELECT * FROM c WHERE c.type = 'thread' AND c.isActive = true "
		
		if filters.userId:
			query += f" AND c.userId = '{filters.userId}'"
		
		if filters.search:
			query += f" AND CONTAINS(c.name, '{filters.search}')"

		query += " ORDER BY c.createdAt DESC"

		all_threads = await asyncio.to_thread(
			lambda: list(self.container.query_items(query=query, enable_cross_partition_query=True))
		)

		start = 0
		end = start + pagination.first
		paginated_threads = all_threads[start:end]

		page_info = PageInfo(
			hasNextPage=(len(all_threads) > end),
			startCursor=None,
			endCursor=None,
		)

		return PaginatedResponse(
			data=paginated_threads,
			pageInfo=page_info
		)

	async def list_threads(
		self, pagination: Pagination, filters: ThreadFilter
	) -> PaginatedResponse[ThreadDict]:
		"""
		Optimized thread listing using indexed fields.
		"""
		# Build query parts to avoid injection and improve readability
		where_clauses = ["c.type = 'thread'", "c.isActive = true"]
		
		if filters.userId:
			where_clauses.append(f"c.userId = '{filters.userId}'")
		
		if filters.search:
			# Note: CONTAINS is still used but will be faster if 'name' is 
			# specifically included in the indexing policy.
			where_clauses.append(f"CONTAINS(c.name, '{filters.search}', true)") # true for case-insensitive

		query_str = f"SELECT * FROM c WHERE {' AND '.join(where_clauses)} ORDER BY c.createdAt DESC"

		# Optimization: Use max_item_count to limit the data returned by the DB engine
		items_iterator = await asyncio.to_thread(
			lambda: self.container.query_items(
				query=query_str,
				enable_cross_partition_query=True,
				max_item_count=pagination.first
			)
		)

		# Fetch only the requested page
		all_threads = await asyncio.to_thread(lambda: list(items_iterator))
		
		# In a real-world scenario, you'd use the continuation token 
		# provided by Cosmos for true pagination.
		paginated_threads = [self.__strip_cosmos_meta(t) for t in all_threads]

		return PaginatedResponse(
			data=paginated_threads,
			pageInfo=PageInfo(
				hasNextPage=len(paginated_threads) >= pagination.first,
				startCursor=None,
				endCursor=None,
			)
		)
	async def build_debug_url(self) -> Optional[str]:
		return ''
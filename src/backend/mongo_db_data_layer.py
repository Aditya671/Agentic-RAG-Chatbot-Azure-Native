import datetime
import uuid
from typing import Optional, Dict, List, Any
from motor.motor_asyncio import AsyncIOMotorClient
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

class MongoDBDataLayer(BaseDataLayer):
    """
    A Chainlit-compatible data layer implementation using MongoDB as the backend.
    """
    def __init__(self,
                 connection_string: str,
                 database_name: str,
                 collection_name: str = "chainlit_data"):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

        self.user_identity = 'local_user'
        self.user_id = str(uuid.uuid4())

        self.message_step_types = [
            'step', 'user_message', 'assistant_message', "run",
            "tool", "llm", "embedding", "retrieval", "rerank", "undefined"
        ]
        self.element_types = [
            "image", "text", "pdf", "tasklist", "audio", "video",
            "file", "plotly", "dataframe", "custom"
        ]

    async def initialize_indexes(self):
        """
        Call this method once during application startup to ensure
        queries for users and threads are performant.
        """
        # Index for user lookups
        await self.collection.create_index([("type", 1), ("identifier", 1)])

        # Index for listing user threads (Sorted by date)
        await self.collection.create_index([("type", 1), ("userId", 1), ("createdAt", -1)])

        # Index for thread data retrieval (steps and elements)
        await self.collection.create_index([("threadId", 1)])

    def _timestamp(self):
        return datetime.datetime.now().isoformat()

    def _format_document(self, doc: Dict) -> Dict:
        """Converts MongoDB _id to string id and removes it."""
        if not doc:
            return None
        if "_id" in doc:
            doc["id"] = str(doc.pop("_id"))
        return doc

    # ----------------------
    # USER METHODS
    # ----------------------
    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        row = await self.collection.find_one({"type": "user", "identifier": identifier})
        if row:
            self.user_identity = str(row.get("identifier"))
            self.user_id = str(row.get("_id"))
            return PersistedUser(
                id=str(row.get("_id")),
                display_name=str(row.get('displayName')),
                identifier=str(row.get("identifier")),
                createdAt=row.get("createdAt"),
                metadata=row.get("metadata", {})
            )
        return None

    async def create_user(self, user: User) -> PersistedUser:
        existing_user = await self.get_user(user.identifier)

        if existing_user:
            self.user_identity = str(existing_user.identifier)
            self.user_id = str(existing_user.id)

            metadata = existing_user.metadata or {}
            metadata['groups'] = user.metadata.get('groups')

            await self.collection.update_one(
                {"_id": existing_user.id},
                {"$set": {
                    "metadata": metadata,
                    "updatedAt": self._timestamp()
                }}
            )
            return existing_user

        now = self._timestamp()
        user_id = str(uuid.uuid4())
        display_name = user.metadata.get("claims", {}).get("displayName") if user.metadata else None

        new_user = {
            "_id": user_id,
            "identifier": user.identifier,
            "type": "user",
            "displayName": display_name,
            "metadata": user.metadata or {},
            "createdAt": now,
            "updatedAt": now,
        }

        self.user_identity = str(user.identifier)
        self.user_id = user_id

        await self.collection.insert_one(new_user)
        return PersistedUser(
            id=user_id,
            display_name=display_name,
            identifier=user.identifier,
            createdAt=now,
            metadata=new_user["metadata"]
        )

    # ----------------------
    # FEEDBACK METHODS
    # ----------------------
    async def get_feedback(self, step_id: str) -> Optional[Dict]:
        feedback = await self.collection.find_one({"type": "feedback", "forId": step_id})
        return self._format_document(feedback)

    async def upsert_feedback(self, feedback: Feedback) -> str:
        feedback_id = feedback.id or str(uuid.uuid4())
        feedback_dict = {
            "type": "feedback",
            "forId": feedback.forId,
            "threadId": feedback.threadId,
            "value": feedback.value,
            "comment": feedback.comment,
            "userId": self.user_id,
            "updatedAt": self._timestamp(),
        }

        await self.collection.update_one(
            {"type": "feedback", "forId": feedback.forId},
            {"$set": feedback_dict, "$setOnInsert": {"createdAt": self._timestamp()}},
            upsert=True
        )
        return feedback_id

    # ----------------------
    # STEP & ELEMENT METHODS
    # ----------------------
    @queue_until_user_message()
    async def create_step(self, step_dict: StepDict):
        step = dict(step_dict)
        step["_id"] = step.pop("id")
        step.setdefault("type", "step")
        step.setdefault("createdAt", self._timestamp())
        await self.collection.insert_one(step)

    @queue_until_user_message()
    async def update_step(self, step_dict: StepDict):
        step = dict(step_dict)
        step_id = step.pop("id")
        step.setdefault("updatedAt", self._timestamp())
        await self.collection.update_one({"_id": step_id}, {"$set": step})

    @queue_until_user_message()
    async def create_element(self, element_dict: ElementDict):
        element = element_dict.to_dict()
        element["_id"] = element.pop("id")
        element.setdefault("type", "element")
        element.setdefault("createdAt", self._timestamp())
        await self.collection.insert_one(element)

    # ----------------------
    # THREAD METHODS
    # ----------------------
    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        # Fetch everything related to this thread
        cursor = self.collection.find({
            "$or": [{"threadId": thread_id}, {"_id": thread_id}]
        })
        items = await cursor.to_list(length=1000)

        if not items:
            return None

        thread_data = None
        steps = []
        elements = []

        for item in items:
            item = self._format_document(item)
            if item["type"] == "thread":
                if not item.get("isActive", True): continue
                thread_data = item
            elif item["type"] in self.message_step_types:
                item['feedback'] = await self.get_feedback(item['id'])
                steps.append(item)
            elif item["type"] in self.element_types:
                elements.append(item)

        if not thread_data: return None

        steps.sort(key=lambda s: s.get("createdAt", ""))
        thread_data["steps"] = steps
        thread_data["elements"] = elements
        return thread_data

    async def update_thread(self, thread_id: str, name: Optional[str] = None,
                            user_id: Optional[str] = None, metadata: Optional[Dict] = None,
                            tags: Optional[List[str]] = None):

        update_fields = {
            "updatedAt": self._timestamp(),
            "userId": user_id or self.user_id,
            "userIdentifier": self.user_identity,
            "isActive": True,
            "type": "thread"
        }

        if name: update_fields["name"] = name
        if metadata: update_fields["metadata"] = metadata
        if tags: update_fields["tags"] = tags

        await self.collection.update_one(
            {"_id": thread_id, "type": "thread"},
            {"$set": update_fields, "$setOnInsert": {"createdAt": self._timestamp()}},
            upsert=True
        )

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse[ThreadDict]:
        query = {"type": "thread", "isActive": True}
        if filters.userId:
            query["userId"] = filters.userId
        if filters.search:
            query["name"] = {"$regex": filters.search, "$options": "i"}

        cursor = self.collection.find(query).sort("createdAt", -1).skip(0).limit(pagination.first + 1)
        all_threads = await cursor.to_list(length=pagination.first + 1)

        has_next_page = len(all_threads) > pagination.first
        data = [self._format_document(t) for t in all_threads[:pagination.first]]

        return PaginatedResponse(
            data=data,
            pageInfo=PageInfo(hasNextPage=has_next_page, startCursor=None, endCursor=None)
        )

    async def delete_thread(self, thread_id: str):
        # Soft delete
        await self.collection.update_one({"_id": thread_id}, {"$set": {"isActive": False}})

    async def build_debug_url(self) -> str:
        return ""
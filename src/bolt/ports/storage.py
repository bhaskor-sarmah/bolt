"""
Abstract session persistence repository
"""

from abc import ABC, abstractmethod
from typing import Optional
from bolt.core.schemas import AssistantMessage

class SessionStorage(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage, create tables if needed."""
        pass

    @abstractmethod
    async def save_session(self, session_id: str, compacted_history: AssistantMessage) -> None:
        """Persist the compacted memory for a given session."""
        pass

    @abstractmethod
    async def load_session(self, session_id: str) -> Optional[AssistantMessage]:
        """Load the compacted memory for a given session."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage connection."""
        pass

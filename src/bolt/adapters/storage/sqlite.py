"""
SQLite WAL repository (implements storage port)
"""

import aiosqlite
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

from bolt.ports.storage import SessionStorage
from bolt.core.schemas import AssistantMessage

logger = logging.getLogger(__name__)

class SQLiteSessionStorage(SessionStorage):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize the database and configure WAL."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        # Enable Write-Ahead Logging for better concurrency
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")

        # Create sessions table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                compacted_content TEXT,
                updated_at TIMESTAMP
            )
        """)
        await self._conn.commit()

    async def save_session(self, session_id: str, compacted_history: AssistantMessage) -> None:
        if not self._conn:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        logger.debug(f"Saving session {session_id} to SQLite.")
        await self._conn.execute("""
            INSERT INTO sessions (id, compacted_content, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                compacted_content = excluded.compacted_content,
                updated_at = excluded.updated_at
        """, (session_id, compacted_history.content, datetime.now().isoformat()))
        await self._conn.commit()

    async def load_session(self, session_id: str) -> Optional[AssistantMessage]:
        if not self._conn:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        logger.debug(f"Loading session {session_id} from SQLite.")
        async with self._conn.execute("SELECT compacted_content FROM sessions WHERE id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                return AssistantMessage(content=row[0])
            return None

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

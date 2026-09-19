import logging
from typing import List, Optional

from bolt.core.schemas import Message, UserMessage, SystemMessage, AssistantMessage, Role
from bolt.core.budgeter import TokenBudgeter
from bolt.ports.driver import ModelDriver

logger = logging.getLogger(__name__)

from bolt.ports.storage import SessionStorage

class MemoryManager:
    """
    Tier 1, Tier 2, and Tier 3 Memory.
    Manages the conversational scratchpad, compacts history when the token budget is near capacity,
    and persists it to the session storage.
    """
    def __init__(self, budgeter: TokenBudgeter, driver: ModelDriver, session_id: str, storage: SessionStorage):
        self.budgeter = budgeter
        self.driver = driver
        self.session_id = session_id
        self.storage = storage
        self.system_prompt: Optional[SystemMessage] = None
        self.scratchpad: List[Message] = []
        self.compacted_history: Optional[AssistantMessage] = None

    async def initialize(self):
        """Loads existing compacted history from storage if available."""
        self.compacted_history = await self.storage.load_session(self.session_id)
        if self.compacted_history:
            logger.info(f"Loaded existing session state for {self.session_id}")

    def set_system_prompt(self, content: str):
        self.system_prompt = SystemMessage(content=content)

    def add_message(self, message: Message):
        self.scratchpad.append(message)

    def get_full_context(self) -> List[Message]:
        """Assembles the final list of messages to send to the ModelDriver."""
        context = []
        if self.system_prompt:
            context.append(self.system_prompt)

        if self.compacted_history:
             # We inject the compacted history as an assistant message that "remembers" the past.
             # Alternatively, this could be appended to the System prompt.
            context.append(self.compacted_history)

        context.extend(self.scratchpad)
        return context

    async def check_and_compact(self):
        """
        Checks the token budget and compacts the scratchpad if necessary.
        """
        budget_status = self.budgeter.check_budget(self.scratchpad)

        if not budget_status["needs_compaction"]:
            logger.debug(f"Memory within budget. Utilization: {budget_status['utilization_pct']}%")
            return

        logger.info(f"Memory budget breached threshold ({budget_status['utilization_pct']}%). Initiating compaction...")
        await self.compact()

    async def compact(self):
        """
        Takes the current scratchpad, asks the ModelDriver to summarize it,
        stores the summary, and clears the old scratchpad (keeping the most recent turn).
        """
        if not self.scratchpad:
            return

        # Prepare a prompt for the model to summarize the history
        compaction_instruction = SystemMessage(
            content="""You are a memory compaction engine.
Your task is to read the following conversation history and create a dense, structured summary.
You must extract:
1. Key decisions made.
2. Modified file paths.
3. Known constraints or errors encountered.
Keep it factual and concise. Do not add conversational filler."""
        )

        # We need to send the history to the model
        messages_to_summarize = [compaction_instruction]
        messages_to_summarize.extend(self.scratchpad)

        try:
            # We use standard generate (not streaming) for background tasks
            logger.debug("Requesting memory compaction from model...")
            response = await self.driver.generate(
                messages=messages_to_summarize,
                max_tokens=self.budgeter.compacted_budget, # Limit output size to our budget
                temperature=0.1 # We want deterministic, factual summaries
            )

            new_summary_text = response.message.content or ""

            # If we already had compacted history, we append to it, or ideally,
            # we should have included the OLD compacted history in the summary prompt.
            # For simplicity, we just merge them here as text if both exist.
            if self.compacted_history and self.compacted_history.content:
                new_summary_text = f"Prior Context:\n{self.compacted_history.content}\n\nNew Context:\n{new_summary_text}"

            self.compacted_history = AssistantMessage(
                content=f"COMPACTED MEMORY ARCHIVE:\n{new_summary_text}"
            )

            # Keep the last 2 messages for immediate context continuity (usually user -> assistant)
            # Or if it ended on a tool, keep the tool results.
            keep_count = min(2, len(self.scratchpad))
            if keep_count > 0:
                self.scratchpad = self.scratchpad[-keep_count:]
            else:
                self.scratchpad = []

            # Save the new compacted history to Tier 3 storage
            if self.compacted_history:
                await self.storage.save_session(self.session_id, self.compacted_history)

            logger.info("Memory compaction completed successfully.")

        except Exception as e:
            logger.error(f"Failed to compact memory: {e}")
            # If compaction fails, we don't clear the scratchpad, we just hope it doesn't break the token limit yet.

    async def save_state(self):
        """Forces a save of the current compacted state to storage, e.g., on shutdown."""
        if self.compacted_history:
            await self.storage.save_session(self.session_id, self.compacted_history)
            logger.info(f"Session {self.session_id} state saved.")

import logging
import tiktoken
from typing import List, Dict, Any, Optional

from bolt.core.schemas import Message

logger = logging.getLogger(__name__)

class TokenBudgeter:
    """
    Manages the token window to ensure we don't hit max_tokens errors
    and provides signals for when compaction is necessary.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        max_context_window: int = 128000,
        system_pct: float = 0.15,
        scratchpad_pct: float = 0.50,
        compacted_pct: float = 0.20,
        headroom_pct: float = 0.15,
    ):
        self.model_name = model_name
        self.max_context_window = max_context_window

        # Verify percentages equal 1.0 (approximate due to floats)
        total_pct = system_pct + scratchpad_pct + compacted_pct + headroom_pct
        if abs(total_pct - 1.0) > 0.01:
            raise ValueError(f"Budget percentages must sum to 1.0. Got {total_pct}")

        self.system_budget = int(max_context_window * system_pct)
        self.scratchpad_budget = int(max_context_window * scratchpad_pct)
        self.compacted_budget = int(max_context_window * compacted_pct)
        self.headroom_budget = int(max_context_window * headroom_pct)

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base which is standard for recent OpenAI models
            logger.warning(f"Could not find exact encoding for {model_name}. Falling back to cl100k_base.")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: Optional[str]) -> int:
        """Counts the tokens in a single string."""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_message_tokens(self, message: Message) -> int:
        """
        Estimates the token count for a message.
        Note: Exact counts depend on the provider's specific formatting,
        but this provides a safe upper bound.
        """
        tokens_per_message = 3
        num_tokens = tokens_per_message

        if message.content:
            num_tokens += self.count_tokens(message.content)

        # Add basic tokens for roles and other structured data
        num_tokens += self.count_tokens(message.role.value)

        # If it's an AssistantMessage, we might need to count reasoning and tool calls
        if hasattr(message, "reasoning") and message.reasoning:
            num_tokens += self.count_tokens(message.reasoning)

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                num_tokens += self.count_tokens(tc.name)
                # Naive dict to string for counting, real models do custom json serializing
                num_tokens += self.count_tokens(str(tc.arguments))

        if hasattr(message, "tool_call_id") and message.tool_call_id:
             num_tokens += self.count_tokens(message.tool_call_id)

        return num_tokens

    def get_total_tokens(self, messages: List[Message]) -> int:
        """Counts total tokens in a conversation."""
        total = 0
        for msg in messages:
            total += self.count_message_tokens(msg)
        return total

    def check_budget(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Evaluates the current token consumption against the scratchpad limit.
        Returns a dict indicating if compaction is needed.
        """
        total = self.get_total_tokens(messages)

        # We consider a breach if we pass 80% of our scratchpad budget.
        # This gives us a 20% safety margin before we actually start failing.
        threshold = int(self.scratchpad_budget * 0.8)

        needs_compaction = total > threshold

        return {
            "total_tokens": total,
            "scratchpad_budget": self.scratchpad_budget,
            "threshold": threshold,
            "needs_compaction": needs_compaction,
            "utilization_pct": round((total / self.scratchpad_budget) * 100, 2) if self.scratchpad_budget else 0
        }

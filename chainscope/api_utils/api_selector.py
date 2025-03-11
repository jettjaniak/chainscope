from dataclasses import dataclass
from typing import Any, Callable, Generic

from chainscope.api_utils.anthropic_utils import ANBatchProcessor
from chainscope.api_utils.batch_processor import BatchItem, BatchProcessor, BatchResult
from chainscope.api_utils.deepseek_utils import DeepSeekBatchProcessor
from chainscope.api_utils.open_ai_utils import OABatchProcessor
from chainscope.api_utils.open_router_utils import ORBatchProcessor


@dataclass
class APIPreferences:
    open_router: bool
    open_ai: bool
    anthropic: bool
    deepseek: bool

    def __str__(self) -> str:
        return f"OpenRouter: {self.open_router}, OpenAI: {self.open_ai}, Anthropic: {self.anthropic}, DeepSeek: {self.deepseek}"

    @classmethod
    def from_args(
        cls,
        open_router: bool = False,
        open_ai: bool = False,
        anthropic: bool = False,
        deepseek: bool = False,
    ) -> "APIPreferences":
        return cls(
            open_router=open_router,
            open_ai=open_ai,
            anthropic=anthropic,
            deepseek=deepseek,
        )

    def selects_at_least_one_api(self) -> bool:
        return self.open_router or self.open_ai or self.anthropic or self.deepseek


@dataclass
class APISelector(Generic[BatchItem, BatchResult]):
    def __init__(self, preferences: APIPreferences):
        self.preferences = preferences

    def get_batch_processor(
        self,
        model_id: str,
        temperature: float,
        max_retries: int,
        process_response: Callable[
            [str | tuple[str | None, str | None], BatchItem], BatchResult | None
        ],
        max_new_tokens: int,
        rate_limiter: Any | None = None,
    ) -> BatchProcessor[BatchItem, BatchResult]:
        selected_processors = []
        if self.preferences.open_router:
            selected_processors.append(ORBatchProcessor)
        if self.preferences.open_ai:
            selected_processors.append(OABatchProcessor)
        if self.preferences.anthropic:
            selected_processors.append(ANBatchProcessor)
        if self.preferences.deepseek:
            selected_processors.append(DeepSeekBatchProcessor)

        if len(selected_processors) == 0:
            # Use OpenRouter by default when no API is selected
            selected_processors.append(ORBatchProcessor)

        available_processors = [
            p for p in selected_processors if p.is_model_supported(model_id)
        ]

        if len(available_processors) > 1:
            raise ValueError(
                f"Multiple API clients found for model {model_id}, please narrow down your API preferences: {self}"
            )
        if len(available_processors) == 0:
            raise ValueError(
                f"No API client found for model {model_id} based on API preferences: {self}"
            )

        return available_processors[0](
            model_id=model_id,
            temperature=temperature,
            max_retries=max_retries,
            process_response=process_response,
            max_new_tokens=max_new_tokens,
            rate_limiter=rate_limiter,
        )

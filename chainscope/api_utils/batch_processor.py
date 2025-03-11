from typing import Callable, Generic, TypeVar

BatchItem = TypeVar("BatchItem")  # Type of the input item
BatchResult = TypeVar("BatchResult")  # Type of the result


class BatchProcessor(Generic[BatchItem, BatchResult]):
    def __init__(
        self,
        model_id: str,
        temperature: float,
        max_retries: int,
        process_response: Callable[
            [str | tuple[str | None, str | None], BatchItem], BatchResult | None
        ],
        max_new_tokens: int,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.process_response = process_response

    async def process_batch(
        self, items: list[tuple[BatchItem, str]]
    ) -> list[tuple[BatchItem, BatchResult | None]]:
        raise NotImplementedError("Subclass must implement process_batch")

    @staticmethod
    def is_model_supported(model_id: str) -> bool:
        raise NotImplementedError("Subclass must implement is_model_supported")

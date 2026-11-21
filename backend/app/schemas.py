"""Request and response models for the public API contract."""

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints, field_validator

SessionId = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]
UserMessage = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=4000)
]
SystemPrompt = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, max_length=8000)
]


class ChatRequest(BaseModel):
    session_id: SessionId
    message: UserMessage
    system_prompt: SystemPrompt | None = None

    @field_validator("session_id")
    @classmethod
    def session_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id must not be blank")
        return value


class ChatMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata

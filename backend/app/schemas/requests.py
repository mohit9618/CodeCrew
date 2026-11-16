from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class RunRequest(BaseModel):
    repo_url: HttpUrl
    request: str = Field(min_length=5, max_length=5000)



class ApprovalRequest(BaseModel):
    decision: Literal["approve", "request_changes", "reject"]
    feedback: str = ""
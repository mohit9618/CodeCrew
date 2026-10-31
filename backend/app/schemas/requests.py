from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    request: str = Field(min_length=5)
    repo_url: str
    workspace: str
    repo_tree: str
    file_contents: str
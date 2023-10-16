from langchain.pydantic_v1 import BaseModel, Field, validator

from typing import List


class MetaData(BaseModel):
    title: str = Field(description="Title of the paper.")
    authors: List[str] = Field(description="List of the author's names")

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ResultIn(BaseModel):
    number: int = Field(..., ge=0, le=9, description="Drawn number 0-9")
    period_id: Optional[str] = Field(None, max_length=32, pattern=r"^\d{1,32}$", description="Round id as TEXT digits")
    timestamp: Optional[datetime] = Field(None, description="ISO-8601 time of the round")


class ImportIn(BaseModel):
    rows: list[ResultIn] = Field(..., min_length=1, max_length=20000, description="Oldest first")

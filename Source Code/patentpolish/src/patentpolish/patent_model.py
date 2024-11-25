from typing import List, Optional

from pydantic import BaseModel, Field


class Patent(BaseModel):
    reference_number: str
    title: str
    abstract: str
    claims: List[str]
    descriptions: List[str]
    image_paths: List[str]
    pdf_path: str
    figure_to_page_mapping: Optional[list[int]] = None

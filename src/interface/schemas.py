from pydantic import BaseModel
from typing import Optional, Dict, Any


class AnalyzerResponse(BaseModel):
    original_review: str
    fixed_review: Optional[str] = None
    sentiment: Optional[str] = None
    thread_id: str
    review_fix_skipped: bool
    product_id: int
    good_points: Dict[str, Any]
    bad_points: Dict[str, Any]
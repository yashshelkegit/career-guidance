from bson import ObjectId
from pydantic_core import core_schema
from typing import Any
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: Any) -> dict:
        return {"type": "string"}

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)



class UserEntry(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    email: EmailStr
    job_roles: str
    business_opportunities: str
    project_ideas: str
    future_trends: str
    score_data: Optional[Dict] = {}

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "email": "example@email.com",
                "job_roles": "Frontend Developer - Learn HTML, CSS, JavaScript, React",
                "business_opportunities": "Start a SaaS app, join startup incubators",
                "project_ideas": "Portfolio site, job board, skill tracker",
                "future_trends": "AI-powered frontend, no-code integrations",
                "score_data": {
                    "interests": ["web dev", "AI"],
                    "preferred_learning": "video"
                }
            }
        }
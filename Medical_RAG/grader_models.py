from pydantic import BaseModel,Field
from typing import Literal, TypedDict
from langchain_core.pydantic_v1 import validator
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from pydantic import BaseModel

class Grader(BaseModel):
    """format used to assign binary score to the retrieved documents based on the relevancy to the query """
    grade:Literal["relevant", "irrelevant"]=Field(
        ...,
        description="Use to grade the relevancy of the retrieved docuemnts"
        "If the retrieved documents are relevant to the query give it a score 'relevant' else 'irrelevant'"

    )
    @validator("grade",pre=True,allow_reuse=True)
    def validate_grade(cls, value):
        if value == "not relevant":
            return "irrelevant"
        return value
    
class HallucinationGrader(BaseModel):
    "Binary score for hallucination check in llm's response"

    grade: Literal["yes", "no"] = Field(
        ..., description="'yes' if the llm's reponse is hallucinated otherwise 'no'"
    )


class AnswerGrader(BaseModel):
    "Binary score for an answer check based on a query."

    grade: Literal["yes", "no"] = Field(
        ...,
        description="'yes' if the provided answer is an actual answer to the query otherwise 'no'",
    )


class AgentState(TypedDict):
    query:str
    response:str
    chat_history:list[BaseMessage]
    context :list[Document]
    last_source:str
    

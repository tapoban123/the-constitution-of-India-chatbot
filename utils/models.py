from pydantic import BaseModel


class DocumentDataModel(BaseModel):
    data: str
    page_nos: list[int]

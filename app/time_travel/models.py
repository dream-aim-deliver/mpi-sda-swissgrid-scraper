from pydantic import BaseModel
from typing import List, Literal, Union

class SwissgridRowSchema(BaseModel):
    timestamp: str
    model: str
    prediction: Literal["ON", "OFF"]
    confidence: float


class Error(BaseModel):
    errorName: str
    errorMessage: str

class Image(BaseModel):
    kind: str
    relativePath: str
    description: str


class KeyFrame(BaseModel):
    timestamp: str
    images: List[Union[Image, Error]]
    data: List[Union[SwissgridRowSchema, Error]]
    dataDescription: str

class Metadata(BaseModel):
    caseStudy: Literal["swissgrid"]
    relativePathsForAgent: List[str]
    keyframes: List[KeyFrame]
    imageKinds: List[str]

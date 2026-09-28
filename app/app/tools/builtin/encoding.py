import base64

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class EncodeResult(BaseModel):
    encoded: str = Field(description="The base64-encoded text.")


class Base64EncodeTool(Tool):
    metadata = ToolMetadata(
        name="base64_encode",
        description="Encode text as base64.",
        parameters=[
            ToolParameter(name="text", type="string", description="The text to encode."),
        ],
        returns=EncodeResult.__name__,
    )

    def execute(self, text: str) -> EncodeResult:
        return EncodeResult(encoded=base64.b64encode(text.encode()).decode())

    def cost(self, text: str) -> float:
        return 0.01


class DecodeResult(BaseModel):
    decoded: str = Field(description="The decoded text.")


class Base64DecodeTool(Tool):
    metadata = ToolMetadata(
        name="base64_decode",
        description="Decode a base64-encoded string back to text.",
        parameters=[
            ToolParameter(name="text", type="string", description="The base64 string to decode."),
        ],
        returns=DecodeResult.__name__,
    )

    def execute(self, text: str) -> DecodeResult:
        try:
            return DecodeResult(decoded=base64.b64decode(text).decode())
        except Exception as e:
            raise ValueError(f"invalid base64 string: '{text}'") from e

    def cost(self, text: str) -> float:
        return 0.01

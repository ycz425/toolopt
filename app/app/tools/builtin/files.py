from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class VirtualFileSystem:
    def __init__(self):
        self.files: dict[str, str] = {}
        self.reset()

    def reset(self) -> None:
        self.files = {
            "readme.txt": "Welcome! This is a scratch filesystem for notes.",
        }


class ReadFileResult(BaseModel):
    path: str = Field(description="Path of the file that was read.")
    content: str = Field(description="Contents of the file.")


class FileReadTool(Tool):
    metadata = ToolMetadata(
        name="file_read",
        description="Read the contents of a file.",
        parameters=[
            ToolParameter(name="path", type="string", description="Path of the file to read."),
        ],
        returns=ReadFileResult.__name__,
    )

    def __init__(self, fs: VirtualFileSystem):
        self.fs = fs

    def execute(self, path: str) -> ReadFileResult:
        if path not in self.fs.files:
            raise ValueError(f"file '{path}' does not exist")
        return ReadFileResult(path=path, content=self.fs.files[path])

    def cost(self, path: str) -> float:
        return 0.01

    def reset(self) -> None:
        self.fs.reset()


class WriteFileResult(BaseModel):
    path: str = Field(description="Path of the file that was written.")
    bytes_written: int = Field(description="Number of characters written.")


class FileWriteTool(Tool):
    metadata = ToolMetadata(
        name="file_write",
        description="Write content to a file, creating it if it doesn't exist or overwriting it if it does.",
        parameters=[
            ToolParameter(name="path", type="string", description="Path of the file to write."),
            ToolParameter(name="content", type="string", description="Content to write to the file."),
        ],
        returns=WriteFileResult.__name__,
    )

    def __init__(self, fs: VirtualFileSystem):
        self.fs = fs

    def execute(self, path: str, content: str) -> WriteFileResult:
        self.fs.files[path] = content
        return WriteFileResult(path=path, bytes_written=len(content))

    def cost(self, path: str, content: str) -> float:
        return 0.02

    def reset(self) -> None:
        self.fs.reset()

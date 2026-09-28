from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class Email(BaseModel):
    to: str = Field(description="Recipient address.")
    subject: str = Field(description="Email subject line.")
    body: str = Field(description="Email body text.")


class SendEmailResult(BaseModel):
    email: Email = Field(description="The email that was sent.")


class SendEmailTool(Tool):
    metadata = ToolMetadata(
        name="send_email",
        description="Send an email to a recipient.",
        parameters=[
            ToolParameter(name="to", type="string", description="Recipient address."),
            ToolParameter(name="subject", type="string", description="Email subject line."),
            ToolParameter(name="body", type="string", description="Email body text."),
        ],
        returns=SendEmailResult.__name__,
    )

    def execute(self, to: str, subject: str, body: str) -> SendEmailResult:
        return SendEmailResult(email=Email(to=to, subject=subject, body=body))

    def cost(self, to: str, subject: str, body: str) -> float:
        return 0.05

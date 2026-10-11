from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter, ToolPricing


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
        pricing=ToolPricing(base=0.05),
        parameters=[
            ToolParameter(name="to", type="string", description="Recipient address."),
            ToolParameter(name="subject", type="string", description="Email subject line."),
            ToolParameter(name="body", type="string", description="Email body text."),
        ],
        returns=SendEmailResult.__name__,
    )

    def execute(self, to: str, subject: str, body: str) -> SendEmailResult:
        return SendEmailResult(email=Email(to=to, subject=subject, body=body))


SHORT_EMAIL_MAX_BODY = 100


class SendShortEmailTool(Tool):
    metadata = ToolMetadata(
        name="send_short_email",
        description=f"Send a short email to a recipient. The body can be at most {SHORT_EMAIL_MAX_BODY} characters.",
        pricing=ToolPricing(base=0.02),
        parameters=[
            ToolParameter(name="to", type="string", description="Recipient address."),
            ToolParameter(name="subject", type="string", description="Email subject line."),
            ToolParameter(name="body", type="string", description=f"Email body text, at most {SHORT_EMAIL_MAX_BODY} characters."),
        ],
        returns=SendEmailResult.__name__,
    )

    def execute(self, to: str, subject: str, body: str) -> SendEmailResult:
        if len(body) > SHORT_EMAIL_MAX_BODY:
            raise ValueError(f"send_short_email bodies can be at most {SHORT_EMAIL_MAX_BODY} characters; got {len(body)}")
        return SendEmailResult(email=Email(to=to, subject=subject, body=body))

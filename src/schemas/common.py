from pydantic import BaseModel, Field

REQUEST_ID_EXAMPLE = "REQ_9cb01af2-493d-446d-b191-af120333f6d0"


class SuccessEnvelope(BaseModel):
    """Bagian envelope yang sama untuk semua response sukses; subclass hanya menentukan `data`."""

    status_code: int = Field(200, examples=[200])
    status_desc: str = Field("OK", examples=["OK"])
    message: str = Field("Success", examples=["Success"])
    errors: None = None
    request_id: str | None = Field(None, examples=[REQUEST_ID_EXAMPLE])

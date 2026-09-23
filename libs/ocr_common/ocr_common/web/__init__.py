"""The FastAPI toolkit every service is built with. It owns no business endpoint: only the app
factory (health probes, exception handlers, request-id middleware), the response envelope, the
API-key dependency, the shared Pydantic response models, and file intake.

Modules:
  app         create_app, add_stage_callback_webhook, database_readiness
  envelope    envelope(): the {status_code, message, data, request_id} body every endpoint returns
  security    verify_api_key: the X-API-Key dependency
  request_id  RequestIdMiddleware and get_request_id
  schemas     shared response models and the error/success example helpers for OpenAPI
  intake      reading the uploaded image or the file_url of a request
  openapi     writing openapi.yaml from a running app (python -m ocr_common.web.openapi)
"""

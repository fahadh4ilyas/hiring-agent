import os
import sys
import json
import time
import tempfile
import traceback
import timeit
from pathlib import Path
from typing import Optional

# Ensure the parent (hiring-agent) directory is on sys.path so we can import
# the top-level modules: pdf, github, evaluator, models, transform, prompt, config
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from fastapi import FastAPI, Request, File, UploadFile, Body, Depends
from fastapi.responses import ORJSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import config, LOGGER, LOGGER_ACCESS

from pdf import PDFHandler
from github import fetch_and_display_github_info
from evaluator import ResumeEvaluator
from models import JSONResume, EvaluationData, ModelProvider
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS, MODEL_PROVIDER_MAPPING
from transform import convert_json_resume_to_text, convert_github_data_to_text
from config import DEVELOPMENT_MODE  # top-level config

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hiring Agent API",
    description="Resume-to-Score pipeline: extracts structured data from PDFs, "
    "enriches with GitHub signals, and returns a fair, explainable evaluation.",
    version="1.0.0",
)

security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return ORJSONResponse(
        {
            "error": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            "status_code": 500,
        },
        status_code=500,
    )


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    client_data = ""
    if request.client:
        client_data = f"{request.client.host}:{request.client.port}"
    LOGGER_ACCESS.info(
        f'{client_data} - "{request.method.upper()} {request.url.path} '
        f'{request.url.scheme.upper()}/1.1" START'
    )

    start = timeit.default_timer()
    response: Response = await call_next(request)
    response.headers["X-Process-Time"] = f"{timeit.default_timer() - start:.6f}"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_profile(profiles, network: str):
    """Return the first profile whose network matches (case-insensitive)."""
    if not profiles:
        return None
    return next(
        (p for p in profiles if p.network and p.network.lower() == network.lower()),
        None,
    )


def _serialize_evaluation(evaluation: EvaluationData) -> dict:
    """Convert an EvaluationData to a plain dict suitable for JSON response."""
    if evaluation is None:
        return {}

    result = {}

    if hasattr(evaluation, "scores") and evaluation.scores:
        result["scores"] = {}
        for name, cat in evaluation.scores.model_dump().items():
            result["scores"][name] = {
                "score": min(cat["score"], cat["max"]),
                "max": cat["max"],
                "evidence": cat["evidence"],
            }

    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        result["bonus_points"] = evaluation.bonus_points.model_dump()

    if hasattr(evaluation, "deductions") and evaluation.deductions:
        result["deductions"] = evaluation.deductions.model_dump()

    if hasattr(evaluation, "key_strengths"):
        result["key_strengths"] = evaluation.key_strengths

    if hasattr(evaluation, "areas_for_improvement"):
        result["areas_for_improvement"] = evaluation.areas_for_improvement

    # Compute totals
    total_score = 0
    max_score = 0
    category_maxes = {
        "open_source": 35,
        "self_projects": 30,
        "production": 25,
        "technical_skills": 10,
    }

    if hasattr(evaluation, "scores") and evaluation.scores:
        for name, cat in evaluation.scores.model_dump().items():
            cat_max = category_maxes.get(name, cat["max"])
            capped = min(cat["score"], cat_max)
            total_score += capped
            max_score += cat_max

    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        total_score += evaluation.bonus_points.total

    if hasattr(evaluation, "deductions") and evaluation.deductions:
        total_score -= evaluation.deductions.total

    max_possible = max_score + 20  # 100 base + 20 bonus
    if total_score > max_possible:
        total_score = max_possible

    result["total_score"] = round(total_score, 1)
    result["max_possible_score"] = max_possible

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return ORJSONResponse({"status": "ok"})


@app.post("/score")
async def score_resume(
    request: Request,
    file: UploadFile = File(...),
    include_resume_data: bool = False,
    model: Optional[str] = Body(None),
    provider: Optional[str] = Body(None),
    base_url: Optional[str] = Body(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """
    Score a resume PDF.

    Upload a PDF resume and receive a structured evaluation with category
    scores, bonus points, deductions, strengths, and areas for improvement.

    **Body (multipart/form-data):**
    - `file` (required): the resume PDF file
    - `model` (optional): model name override (e.g. `gpt-4o`, `gemini-2.5-pro`)
    - `provider` (optional): force provider (`ollama`, `gemini`, `openai`).
      When omitted, provider is inferred from the model name.
    - `base_url` (optional): base URL for OpenAI-compatible endpoints
      (e.g. `https://api.openai.com/v1`)

    **Query params:**
    - `include_resume_data` (bool): when true, include the parsed resume
      JSON in the response.
    """
    effective_model = model or DEFAULT_MODEL

    # Extract Bearer token from Authorization header
    effective_api_key = None
    if credentials:
        effective_api_key = credentials.credentials

    # If provider is explicitly set, ensure the model mapping reflects it
    if provider:
        try:
            provider_enum = ModelProvider(provider)
            MODEL_PROVIDER_MAPPING[effective_model] = provider_enum
        except ValueError:
            return ORJSONResponse(
                {"error": f"Invalid provider '{provider}'. Use one of: {[p.value for p in ModelProvider]}"},
                status_code=400,
            )

    # Read uploaded PDF into a temporary file
    contents = await file.read()
    suffix = ".pdf"
    if file.filename and file.filename.lower().endswith(".pdf"):
        suffix = Path(file.filename).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        pdf_path = tmp.name

    try:
        start_time = time.time()

        # --- 1. PDF extraction ---
        pdf_handler = PDFHandler(
            model_name=effective_model,
            api_key=effective_api_key,
            base_url=base_url,
        )
        resume_data: Optional[JSONResume] = pdf_handler.extract_json_from_pdf(pdf_path)

        if resume_data is None:
            return ORJSONResponse(
                {"error": "Failed to extract structured data from the PDF."},
                status_code=422,
            )

        # --- 2. GitHub enrichment ---
        github_data = {}
        profiles = []
        if resume_data and hasattr(resume_data, "basics") and resume_data.basics:
            profiles = resume_data.basics.profiles or []

        github_profile = _find_profile(profiles, "Github")
        if github_profile:
            github_data = fetch_and_display_github_info(
                github_profile.url,
                model_name=effective_model,
                api_key=effective_api_key,
                base_url=base_url,
            )

        # --- 3. Evaluation ---
        model_params = MODEL_PARAMETERS.get(effective_model)
        evaluator = ResumeEvaluator(
            model_name=effective_model,
            model_params=model_params,
            api_key=effective_api_key,
            base_url=base_url,
        )

        resume_text = convert_json_resume_to_text(resume_data)
        if github_data:
            github_text = convert_github_data_to_text(github_data)
            resume_text += github_text

        evaluation = evaluator.evaluate_resume(resume_text)

        # --- 4. Build response ---
        candidate_name = file.filename or "Candidate"
        if (
            resume_data
            and hasattr(resume_data, "basics")
            and resume_data.basics
            and resume_data.basics.name
        ):
            candidate_name = resume_data.basics.name

        elapsed = round(time.time() - start_time, 2)

        response_body = {
            "candidate": candidate_name,
            "evaluation": _serialize_evaluation(evaluation),
            "elapsed_seconds": elapsed,
        }

        if include_resume_data and resume_data:
            response_body["resume_data"] = resume_data.model_dump()

        return ORJSONResponse(response_body)

    finally:
        # Clean up the temporary PDF file
        try:
            os.unlink(pdf_path)
        except OSError:
            pass

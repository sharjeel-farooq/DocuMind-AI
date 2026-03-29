import shutil
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from pydantic import BaseModel

from rag.pipeline import process_document, ask_question


router = APIRouter()

# Simple in-memory rate limiter (per client host)
RATE_LIMIT_SECONDS = 1.0
_last_call = {}


def _check_rate_limit(request: Request):
	host = request.client.host if request.client else "unknown"
	now = time.time()
	last = _last_call.get(host, 0)
	if now - last < RATE_LIMIT_SECONDS:
		raise HTTPException(status_code=429, detail="Too many requests, slow down")
	_last_call[host] = now


class QueryRequest(BaseModel):
	query: str


class EvalItem(BaseModel):
	query: str
	expected_substrings: List[str]


class EvalRequest(BaseModel):
	items: List[EvalItem]


@router.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
	if not files:
		raise HTTPException(status_code=400, detail="No files uploaded")

	temp_dir = Path("temp_uploads_api")
	temp_dir.mkdir(exist_ok=True)

	processed = []
	for file in files:
		suffix = Path(file.filename).suffix or ".txt"
		dest = temp_dir / f"{file.filename}"
		with dest.open("wb") as f:
			shutil.copyfileobj(file.file, f)

		try:
			process_document(str(dest))
			processed.append(file.filename)
		except Exception as exc:  # noqa: BLE001
			raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}: {exc}")

	return {"processed": processed}


@router.post("/query")
async def query(body: QueryRequest, request: Request):
	if not body.query.strip():
		raise HTTPException(status_code=400, detail="Query is empty")

	_check_rate_limit(request)

	start = time.time()
	try:
		answer, sources = ask_question(body.query)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=500, detail=str(exc))
	elapsed = (time.time() - start) * 1000

	source_preview = [
		{
			"source": s.get("metadata", {}).get("source", "unknown"),
			"score": s.get("score"),
			"text": s.get("text", "")[:200],
		}
		for s in sources
	]

	return {"answer": answer, "sources": source_preview, "latency_ms": round(elapsed, 1)}


@router.post("/eval")
async def eval_queries(body: EvalRequest):
	if not body.items:
		raise HTTPException(status_code=400, detail="No eval items provided")

	results = []
	for item in body.items:
		start = time.time()
		try:
			answer, _ = ask_question(item.query)
		except Exception as exc:  # noqa: BLE001
			results.append({"query": item.query, "error": str(exc)})
			continue

		elapsed = (time.time() - start) * 1000
		matched = any(sub.lower() in answer.lower() for sub in item.expected_substrings)
		results.append(
			{
				"query": item.query,
				"matched": matched,
				"latency_ms": round(elapsed, 1),
				"answer": answer,
				"expectations": item.expected_substrings,
			}
		)

	accuracy = sum(1 for r in results if r.get("matched")) / len(results)
	return {"accuracy": accuracy, "results": results}

"""
Diagnostic server: retrieval + GPT-OSS structured output.

Usage:
    uv run uvicorn src.mock_server:app --host 127.0.0.1 --port 8000

Required env:
    GPT_OSS_API_KEY

Optional env:
    GPT_OSS_HOST=https://hub.qazcode.ai
    GPT_OSS_MODEL=oss-120b
    GPT_OSS_TIMEOUT_S=60
    GPT_OSS_MAX_RETRIES=2
    GPT_OSS_RETRY_BACKOFF_S=1.5
    GPT_OSS_TEMPERATURE=0.1
    LLM_PARALLELISM=2
    CHROMA_DB_DIR=./chroma_db
    CHROMA_COLLECTION=ChunkLength-512
    EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
    ENABLE_RETRIEVAL=true
    USE_RERANKER=false
    TOP_K_CHUNKS=25
    TOP_N_PROTOCOLS=5
    TOP_CHUNKS_PER_PROTOCOL=3
    MAX_CHUNK_CHARS=2200
    MAX_PROTOCOL_CONTEXT_CHARS=12000
    QUERY_INSTRUCTION=Given a medical query, find relevant protocol chunks:
"""

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from protocol_search import ProtocolSearcher

logger = logging.getLogger("diagnostic_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

PROTOCOL_SYSTEM_PROMPT_RU = """
Ты медицинский ассистент по МКБ-10.
Твоя задача: по симптомам пациента и одному медицинскому протоколу вернуть один наиболее вероятный диагноз.

Верни СТРОГО JSON (без markdown и без текста вне JSON) в точной схеме:
{
  "diagnosis": "строка, название диагноза на русском",
  "icd10_code": "строка, код МКБ-10",
  "explanation": "строка, краткое клиническое обоснование на русском",
  "confidence": 0.0
}

Правила:
1. Используй только переданный контекст протокола.
2. Если дан список allowed_icd10_codes, код обязан быть из этого списка.
3. confidence должен быть числом от 0 до 1.
4. Никаких дополнительных полей.
""".strip()

DEFAULT_FALLBACK_CODES = ["R69", "R50.9", "R53"]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer in %s=%r; fallback=%s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float in %s=%r; fallback=%s", name, raw, default)
        return default

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid bool in %s=%r; fallback=%s", name, raw, default)
    return default


def _normalize_code(code: str) -> str:
    return re.sub(r"\s+", "", code or "").upper()


def _parse_icd_codes(raw: Any) -> list[str]:
    if raw is None:
        return []

    values: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                item_norm = item.strip()
                if item_norm:
                    values.append(item_norm)
    elif isinstance(raw, str):
        for item in raw.split(","):
            item_norm = item.strip()
            if item_norm:
                values.append(item_norm)

    unique_codes: list[str] = []
    seen = set()
    for code in values:
        norm = _normalize_code(code)
        if norm in seen:
            continue
        seen.add(norm)
        unique_codes.append(code)
    return unique_codes


def _code_is_allowed(predicted_code: str, allowed_codes: list[str]) -> bool:
    normalized_predicted = _normalize_code(predicted_code)
    if not normalized_predicted:
        return False

    for code in allowed_codes:
        normalized_allowed = _normalize_code(code)
        if normalized_predicted == normalized_allowed:
            return True
        if normalized_predicted.startswith(f"{normalized_allowed}."):
            return True
        if normalized_allowed.startswith(f"{normalized_predicted}."):
            return True
    return False


def _extract_json_payload(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Empty model response")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start : end + 1])

    raise ValueError("Model response does not contain valid JSON object")


def _build_chat_completions_url(host: str) -> str:
    url = host.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/chat/completions"


@dataclass(slots=True)
class AppSettings:
    gpt_host: str
    gpt_api_key: str
    gpt_model: str
    gpt_timeout_s: float
    gpt_max_retries: int
    gpt_retry_backoff_s: float
    gpt_temperature: float
    llm_parallelism: int
    chroma_db_dir: str
    chroma_collection: str
    embedding_model: str
    enable_retrieval: bool
    use_reranker: bool
    top_k_chunks: int
    top_n_protocols: int
    top_chunks_per_protocol: int
    max_chunk_chars: int
    max_protocol_context_chars: int
    query_instruction: str

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            gpt_host=os.getenv("GPT_OSS_HOST", "https://hub.qazcode.ai"),
            gpt_api_key=os.getenv("GPT_OSS_API_KEY", ""),
            gpt_model=os.getenv("GPT_OSS_MODEL", "oss-120b"),
            gpt_timeout_s=_env_float("GPT_OSS_TIMEOUT_S", 60.0),
            gpt_max_retries=max(0, _env_int("GPT_OSS_MAX_RETRIES", 2)),
            gpt_retry_backoff_s=max(0.1, _env_float("GPT_OSS_RETRY_BACKOFF_S", 1.5)),
            gpt_temperature=_env_float("GPT_OSS_TEMPERATURE", 0.1),
            llm_parallelism=max(1, _env_int("LLM_PARALLELISM", 2)),
            chroma_db_dir=os.getenv("CHROMA_DB_DIR", "./chroma_db"),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "ChunkLength-512"),
            # embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "./models/embedding"),
            enable_retrieval=_env_bool("ENABLE_RETRIEVAL", True),
            use_reranker=_env_bool("USE_RERANKER", False),
            top_k_chunks=max(3, _env_int("TOP_K_CHUNKS", 25)),
            top_n_protocols=max(1, _env_int("TOP_N_PROTOCOLS", 5)),
            top_chunks_per_protocol=max(1, _env_int("TOP_CHUNKS_PER_PROTOCOL", 3)),
            max_chunk_chars=max(300, _env_int("MAX_CHUNK_CHARS", 2200)),
            max_protocol_context_chars=max(
                1500, _env_int("MAX_PROTOCOL_CONTEXT_CHARS", 12000)
            ),
            query_instruction=os.getenv(
                "QUERY_INSTRUCTION",
                "Given a medical query, find relevant protocol chunks: ",
            ),
        )


class DiagnoseRequest(BaseModel):
    symptoms: Optional[str] = ""


class Diagnosis(BaseModel):
    rank: int = Field(ge=1)
    diagnosis: str = Field(min_length=1)
    icd10_code: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


class ProtocolCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis: str = Field(min_length=1, max_length=300)
    icd10_code: str = Field(min_length=1, max_length=20)
    explanation: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(slots=True)
class ProtocolContext:
    protocol_id: str
    retrieval_rank: int
    title: str
    source_file: str
    allowed_codes: list[str]
    context_text: str


class GptOssClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.chat_url = _build_chat_completions_url(settings.gpt_host)
        headers = {"Content-Type": "application/json"}
        if settings.gpt_api_key:
            headers["Authorization"] = f"Bearer {settings.gpt_api_key}"
        self.client = httpx.AsyncClient(headers=headers, timeout=settings.gpt_timeout_s)

    async def close(self) -> None:
        await self.client.aclose()

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.settings.gpt_model,
            "temperature": self.settings.gpt_temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.settings.gpt_max_retries + 1):
            try:
                response = await self.client.post(self.chat_url, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt < self.settings.gpt_max_retries:
                        delay = self.settings.gpt_retry_backoff_s * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue
                response.raise_for_status()
                response_json = response.json()
                content = (
                    response_json.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if isinstance(content, list):
                    parts: list[str] = []
                    for part in content:
                        if isinstance(part, dict):
                            parts.append(part.get("text", ""))
                        else:
                            parts.append(str(part))
                    content_text = "".join(parts)
                else:
                    content_text = str(content)
                return _extract_json_payload(content_text)
            except (
                httpx.HTTPError,
                json.JSONDecodeError,
                ValueError,
                KeyError,
                IndexError,
            ) as error:
                last_error = error
                if attempt < self.settings.gpt_max_retries:
                    delay = self.settings.gpt_retry_backoff_s * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                break

        if last_error:
            raise RuntimeError(f"GPT-OSS request failed: {last_error}") from last_error
        raise RuntimeError("GPT-OSS request failed with unknown error")


class DiagnosticEngine:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.gpt_client = GptOssClient(settings)
        self.searcher: Optional[ProtocolSearcher] = None

    async def startup(self) -> None:
        logger.info("Initializing retrieval + generation pipeline")
        if not self.settings.enable_retrieval:
            logger.warning("Retrieval is disabled via ENABLE_RETRIEVAL=false")
            self.searcher = None
            return

        try:
            self.searcher = ProtocolSearcher(
                chroma_persist_directory=self.settings.chroma_db_dir,
                collection_name=self.settings.chroma_collection,
                embedding_model_name=self.settings.embedding_model,
                sqlite_db_path="protocols.db",
                use_reranker=self.settings.use_reranker,
            )
        except Exception as error:
            logger.exception("Failed to initialize ProtocolSearcher: %s", error)
            self.searcher = None

        if not self.settings.gpt_api_key:
            logger.warning(
                "GPT_OSS_API_KEY is empty. Service will return fallback responses."
            )

    async def shutdown(self) -> None:
        await self.gpt_client.close()

    def _global_fallback(self, reason: str) -> DiagnoseResponse:
        diagnoses = [
            Diagnosis(
                rank=1,
                diagnosis="Недифференцированное состояние",
                icd10_code=DEFAULT_FALLBACK_CODES[0],
                explanation=f"Резервный ответ: {reason}.",
            ),
            Diagnosis(
                rank=2,
                diagnosis="Лихорадка неуточненная",
                icd10_code=DEFAULT_FALLBACK_CODES[1],
                explanation="Недостаточно данных для точного вывода.",
            ),
            Diagnosis(
                rank=3,
                diagnosis="Недомогание и утомляемость",
                icd10_code=DEFAULT_FALLBACK_CODES[2],
                explanation="Добавьте больше клинических деталей в описание симптомов.",
            ),
        ]
        return DiagnoseResponse(diagnoses=diagnoses)

    def _fallback_candidate(
        self, protocol: ProtocolContext, reason: str
    ) -> ProtocolCandidate:
        fallback_code = protocol.allowed_codes[0] if protocol.allowed_codes else "R69"
        diagnosis_name = (
            protocol.title.strip() if protocol.title.strip() else "Наиболее вероятный диагноз"
        )
        return ProtocolCandidate(
            diagnosis=diagnosis_name,
            icd10_code=fallback_code,
            explanation=(
                f"Резервный вывод для protocol_id={protocol.protocol_id}. "
                f"Причина: {reason}."
            ),
            confidence=0.2,
        )

    def _collect_contexts_from_chunks(self, symptoms: str) -> list[ProtocolContext]:
        if not self.searcher:
            return []

        # Preferred path: use teammate #2 pipeline output (top protocols + full context from SQLite)
        try:
            advanced = self.searcher.advanced_search(
                query=symptoms,
                instruction=self.settings.query_instruction,
                top_k_chunks=self.settings.top_k_chunks,
                final_top_protocols=self.settings.top_n_protocols,
            )
        except Exception as error:
            logger.warning("advanced_search failed; fallback to chunk grouping: %s", error)
            advanced = []

        contexts_from_advanced: list[ProtocolContext] = []
        for rank, item in enumerate(advanced, start=1):
            full_protocol = item.get("full_protocol_data") or {}
            protocol_id = item.get("protocol_id") or full_protocol.get("protocol_id")
            if not protocol_id:
                continue

            title = full_protocol.get("title", "") or ""
            source_file = full_protocol.get("source_file", "") or ""
            allowed_codes = _parse_icd_codes(full_protocol.get("icd_codes"))
            if not allowed_codes:
                allowed_codes = _parse_icd_codes(full_protocol.get("icd_codes_str"))

            protocol_text = (full_protocol.get("text", "") or "").strip()
            winning_chunk = (item.get("winning_chunk_snippet", "") or "").strip()

            if protocol_text:
                context_text = protocol_text[: self.settings.max_protocol_context_chars]
            else:
                context_text = winning_chunk[: self.settings.max_protocol_context_chars]

            if not context_text:
                continue

            contexts_from_advanced.append(
                ProtocolContext(
                    protocol_id=protocol_id,
                    retrieval_rank=rank,
                    title=title,
                    source_file=source_file,
                    allowed_codes=allowed_codes,
                    context_text=context_text,
                )
            )

        if contexts_from_advanced:
            return contexts_from_advanced

        results = self.searcher.search(
            query=symptoms,
            n_results=self.settings.top_k_chunks,
            instruction=self.settings.query_instruction,
        )

        if not results.get("ids") or not results["ids"][0]:
            return []

        grouped: dict[str, dict[str, Any]] = {}
        for doc, metadata, distance in zip(
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            if not metadata:
                continue
            protocol_id = metadata.get("protocol_id")
            if not protocol_id:
                continue

            score = 1.0 - float(distance)
            title = metadata.get("title", "") or ""
            source_file = metadata.get("source_file", "") or ""
            allowed_codes = _parse_icd_codes(metadata.get("icd_codes_str"))
            chunk_text = (doc or "")[: self.settings.max_chunk_chars]

            bucket = grouped.setdefault(
                protocol_id,
                {
                    "scores": [],
                    "title": title,
                    "source_file": source_file,
                    "allowed_codes": set(),
                    "chunks": [],
                },
            )
            bucket["scores"].append(score)
            if allowed_codes:
                bucket["allowed_codes"].update({_normalize_code(code) for code in allowed_codes})
            bucket["chunks"].append((score, chunk_text))

        ranked_protocols = sorted(
            grouped.items(),
            key=lambda item: max(item[1]["scores"]) if item[1]["scores"] else 0.0,
            reverse=True,
        )

        contexts: list[ProtocolContext] = []
        for rank, (protocol_id, bucket) in enumerate(
            ranked_protocols[: self.settings.top_n_protocols], start=1
        ):
            top_chunks = sorted(bucket["chunks"], key=lambda item: item[0], reverse=True)[
                : self.settings.top_chunks_per_protocol
            ]
            chunk_lines = []
            for idx, (score, chunk) in enumerate(top_chunks, start=1):
                chunk_lines.append(
                    f"[chunk_{idx}; relevance={score:.4f}]\n{chunk}"
                )
            context_text = "\n\n".join(chunk_lines)
            context_text = context_text[: self.settings.max_protocol_context_chars]
            allowed_codes_sorted = sorted(list(bucket["allowed_codes"]))
            contexts.append(
                ProtocolContext(
                    protocol_id=protocol_id,
                    retrieval_rank=rank,
                    title=bucket["title"],
                    source_file=bucket["source_file"],
                    allowed_codes=allowed_codes_sorted,
                    context_text=context_text,
                )
            )

        return contexts

    async def _generate_protocol_candidate(
        self, symptoms: str, protocol: ProtocolContext
    ) -> ProtocolCandidate:
        if not self.settings.gpt_api_key:
            raise RuntimeError("GPT_OSS_API_KEY is not configured")

        allowed_codes_text = ", ".join(protocol.allowed_codes) if protocol.allowed_codes else "N/A"
        user_prompt = (
            "Симптомы пациента:\n"
            f"{symptoms}\n\n"
            "Данные протокола:\n"
            f"protocol_id: {protocol.protocol_id}\n"
            f"title: {protocol.title}\n"
            f"source_file: {protocol.source_file}\n"
            f"allowed_icd10_codes: {allowed_codes_text}\n\n"
            "Ключевые фрагменты протокола:\n"
            f"{protocol.context_text}\n\n"
            "Верни только JSON по указанной схеме."
        )

        payload = await self.gpt_client.generate_json(
            system_prompt=PROTOCOL_SYSTEM_PROMPT_RU,
            user_prompt=user_prompt,
        )

        candidate = ProtocolCandidate.model_validate(payload)
        if protocol.allowed_codes and not _code_is_allowed(
            candidate.icd10_code, protocol.allowed_codes
        ):
            raise ValueError(
                f"icd10_code={candidate.icd10_code} outside protocol allowed set"
            )
        return candidate

    async def diagnose(self, symptoms: str) -> DiagnoseResponse:
        symptoms_clean = (symptoms or "").strip()
        if not symptoms_clean:
            return self._global_fallback("пустой запрос")

        contexts = self._collect_contexts_from_chunks(symptoms_clean)
        if not contexts:
            return self._global_fallback("retrieval не вернул релевантные протоколы")

        semaphore = asyncio.Semaphore(self.settings.llm_parallelism)

        async def process_one(
            protocol: ProtocolContext,
        ) -> tuple[ProtocolContext, Optional[ProtocolCandidate]]:
            async with semaphore:
                try:
                    candidate = await self._generate_protocol_candidate(
                        symptoms_clean, protocol
                    )
                    return protocol, candidate
                except (RuntimeError, ValueError, ValidationError, httpx.HTTPError) as err:
                    logger.warning(
                        "Candidate generation failed for protocol_id=%s: %s",
                        protocol.protocol_id,
                        err,
                    )
                    return protocol, None

        generated = await asyncio.gather(*[process_one(context) for context in contexts])
        generated_sorted = sorted(generated, key=lambda row: row[0].retrieval_rank)
        real_generated = [
            (protocol, candidate)
            for protocol, candidate in generated_sorted
            if candidate is not None
        ]

        if not real_generated:
            logger.warning("No real model candidates were generated for this request")
            return DiagnoseResponse(diagnoses=[])

        max_rank_index = max(len(real_generated) - 1, 1)
        best_by_code: dict[str, tuple[float, ProtocolCandidate]] = {}

        for idx, (protocol, candidate) in enumerate(real_generated):
            retrieval_score = 1.0 - (idx / max_rank_index)
            score = 0.6 * retrieval_score + 0.4 * candidate.confidence
            code_norm = _normalize_code(candidate.icd10_code)
            current = best_by_code.get(code_norm)
            if current is None or score > current[0]:
                best_by_code[code_norm] = (score, candidate)

        ranked = sorted(best_by_code.values(), key=lambda item: item[0], reverse=True)
        diagnoses: list[Diagnosis] = []

        for rank, (_, candidate) in enumerate(ranked[:3], start=1):
            diagnoses.append(
                Diagnosis(
                    rank=rank,
                    diagnosis=candidate.diagnosis,
                    icd10_code=candidate.icd10_code,
                    explanation=candidate.explanation,
                )
            )

        return DiagnoseResponse(diagnoses=diagnoses)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = AppSettings.from_env()
    engine = DiagnosticEngine(settings)
    await engine.startup()
    app.state.engine = engine

    print("\nDiagnostic Server (FastAPI)")
    print("=" * 40)
    print("Endpoint: /diagnose")
    print("Method:   POST")
    print('Body:     {"symptoms": "..."}')
    print("Docs:     /docs")
    print("=" * 40)
    print("\nPress Ctrl+C to stop\n")

    try:
        yield
    finally:
        await engine.shutdown()


app = FastAPI(title="ICD-10 Diagnostic Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = "./src/frontend"
# if frontend_dir.exists():
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
async def read_index():
    return FileResponse(f"{frontend_dir}/index.html")


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    engine: DiagnosticEngine = app.state.engine
    try:
        return await engine.diagnose(request.symptoms or "")
    except Exception as error:
        logger.exception("Pipeline execution error: %s", error)
        return engine._global_fallback("внутренняя ошибка пайплайна")

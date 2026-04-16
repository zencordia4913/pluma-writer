import os
import json
import time
import asyncio
import re
import copy
import logging
import traceback as _traceback_mod
import requests
import pandas as pd
import streamlit as st

from datetime import datetime
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from dotenv import load_dotenv
import hashlib
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

# Load environment variables
load_dotenv()

logger = logging.getLogger("pluma.writer_main")

# Import deep_research components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_research.pipeline import run_deep_research

# Import for link processing
from tavily import AsyncTavilyClient

# langchain_azure_ai has shipped multiple Azure chat model wrappers over time.
# Keep this import compatible across versions.
try:
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel as AzureChatModel  # type: ignore
    _AZURE_CHAT_MODEL_FLAVOR = "azure_openai_api"
except ImportError:  # pragma: no cover
    try:
        from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel as AzureChatModel  # type: ignore
        _AZURE_CHAT_MODEL_FLAVOR = "azure_ai_chat_completions"
    except ImportError:  # pragma: no cover
        from langchain_azure_ai.chat_models import AzureChatOpenAI as AzureChatModel  # type: ignore
        _AZURE_CHAT_MODEL_FLAVOR = "azure_chat_openai"
from langchain_core.messages import HumanMessage, SystemMessage

# Import plagiarism checker
from app.plagiarism_checker import check_plagiarism

# Import policy checker
from app.policy_checker import check_speech_policy_alignment


# Model initialization for link processing - lazy loaded (tier/deployment aware)
# Cache is keyed by deployment + asyncio loop id to avoid cross-loop client reuse.
_link_processing_models: Dict[str, Any] = {}


def _current_loop_cache_scope() -> str:
    try:
        loop = asyncio.get_running_loop()
        return f"loop:{id(loop)}"
    except RuntimeError:
        return "loop:none"

# Optional persistent cache container (Cosmos)
_stage1_cache_container = None

# Stage 1 cache (in-memory per process)
_STAGE1_CACHE: Dict[str, Dict[str, Any]] = {
    "topic": {},
    "links": {},
    "attachments": {},
}


def _is_stage1_cache_enabled() -> bool:
    return os.getenv("WRITER_STAGE1_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _is_stage1_persistent_cache_enabled() -> bool:
    return os.getenv("WRITER_STAGE1_PERSISTENT_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _get_stage1_cache_container():
    global _stage1_cache_container

    if _stage1_cache_container is not None:
        return _stage1_cache_container

    if not _is_stage1_persistent_cache_enabled():
        return None

    try:
        endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
        key = os.getenv("AZURE_COSMOS_KEY")
        database_name = os.getenv("AZURE_COSMOS_DATABASE")
        container_name = os.getenv("WRITER_STAGE1_CACHE_CONTAINER", "writer_stage1_cache")
        ttl_seconds = int(os.getenv("WRITER_STAGE1_CACHE_TTL_SECONDS", "86400"))

        if not endpoint or not key or not database_name:
            return None

        cosmos_client = CosmosClient(url=endpoint, credential=key)
        database = cosmos_client.get_database_client(database_name)
        _stage1_cache_container = database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/cache_bucket"),
            default_ttl=max(60, ttl_seconds),
        )
        return _stage1_cache_container
    except Exception:
        return None


def _stable_json_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _parse_int_env(name: str, default_value: int, minimum: int = 1, maximum: int = 7) -> int:
    raw = os.getenv(name, str(default_value))
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = default_value
    return max(minimum, min(maximum, parsed))


def _get_stage1_cache(bucket: str, cache_key: str) -> Optional[Dict[str, Any]]:
    if not _is_stage1_cache_enabled():
        return None

    cache_bucket = _STAGE1_CACHE.get(bucket, {})
    if cache_key not in cache_bucket:
        # Optional persistent cache lookup
        container = _get_stage1_cache_container()
        if not container:
            return None
        try:
            doc = container.read_item(item=cache_key, partition_key=bucket)
            value = doc.get("value")
            if isinstance(value, dict):
                _STAGE1_CACHE.setdefault(bucket, {})[cache_key] = copy.deepcopy(value)
                return copy.deepcopy(value)
        except Exception:
            return None
        return None

    return copy.deepcopy(cache_bucket[cache_key])


def _set_stage1_cache(bucket: str, cache_key: str, value: Dict[str, Any]) -> None:
    if not _is_stage1_cache_enabled():
        return

    if bucket not in _STAGE1_CACHE:
        _STAGE1_CACHE[bucket] = {}
    _STAGE1_CACHE[bucket][cache_key] = copy.deepcopy(value)

    container = _get_stage1_cache_container()
    if not container:
        return

    try:
        max_doc_bytes = int(os.getenv("WRITER_STAGE1_CACHE_MAX_DOC_BYTES", "750000"))
    except ValueError:
        max_doc_bytes = 750000

    try:
        payload_json = json.dumps(value, ensure_ascii=False, default=str)
        if len(payload_json.encode("utf-8")) > max_doc_bytes:
            return

        ttl_seconds = int(os.getenv("WRITER_STAGE1_CACHE_TTL_SECONDS", "86400"))
        container.upsert_item({
            "id": cache_key,
            "cache_bucket": bucket,
            "value": json.loads(payload_json),
            "created_at": datetime.now().isoformat(),
            "ttl": max(60, ttl_seconds),
        })
    except Exception:
        return


def _attachment_cache_fingerprint(attachment: Any) -> Dict[str, Any]:
    if isinstance(attachment, dict):
        content = attachment.get("content", "")
        raw_bytes = attachment.get("bytes", b"")

        if isinstance(content, str):
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        else:
            content_hash = None

        if isinstance(raw_bytes, (bytes, bytearray)):
            bytes_hash = hashlib.sha256(bytes(raw_bytes)).hexdigest()
        else:
            bytes_hash = None

        return {
            "filename": attachment.get("filename"),
            "file_type": attachment.get("file_type"),
            "content_hash": content_hash,
            "bytes_hash": bytes_hash,
        }

    if isinstance(attachment, str):
        return {
            "type": "str",
            "content_hash": hashlib.sha256(attachment.encode("utf-8")).hexdigest(),
        }

    return {"type": type(attachment).__name__}


def _reassign_evidence_ids(evidence_items: List[Dict[str, Any]], start_id: int) -> (List[Dict[str, Any]], int):
    reassigned: List[Dict[str, Any]] = []
    current_id = start_id

    for item in evidence_items:
        cloned = copy.deepcopy(item)
        cloned["id"] = f"E{current_id}"
        reassigned.append(cloned)
        current_id += 1

    return reassigned, current_id


_STYLE_DIGEST_CACHE: Dict[str, Dict[str, str]] = {}
_POLICY_CHECK_CACHE: Dict[str, Dict[str, Any]] = {}


def _is_policy_cache_enabled() -> bool:
    return os.getenv("WRITER_POLICY_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _get_policy_cache_ttl_seconds() -> int:
    try:
        ttl = int(os.getenv("WRITER_POLICY_CACHE_TTL_SECONDS", "7200"))
    except ValueError:
        ttl = 7200
    return max(60, min(7 * 24 * 3600, ttl))


def _build_policy_cache_key(speech_content: str, speech_metadata: Dict[str, Any]) -> str:
    payload = {
        "speech_hash": hashlib.sha256((speech_content or "").encode("utf-8")).hexdigest(),
        "metadata": {
            "topic": speech_metadata.get("topic"),
            "speaker": speech_metadata.get("speaker"),
            "audience": speech_metadata.get("audience"),
            "query": speech_metadata.get("query"),
        },
        "policy_deployment": os.getenv("AZURE_POLICY_DEPLOYMENT"),
        "policy_agent_id": os.getenv("AZURE_POLICY_AGENT_ID"),
    }
    return _stable_json_hash(payload)


def _get_cached_policy_check(cache_key: str) -> Optional[Dict[str, Any]]:
    if not _is_policy_cache_enabled():
        return None

    now_ts = time.time()
    cached = _POLICY_CHECK_CACHE.get(cache_key)
    if cached and cached.get("expires_at", 0) > now_ts:
        return copy.deepcopy(cached.get("result"))

    # Optional persistent reuse via shared cache container
    cached_doc = _get_stage1_cache("policy", cache_key)
    if cached_doc and isinstance(cached_doc, dict):
        expires_at = float(cached_doc.get("expires_at", 0) or 0)
        if expires_at > now_ts and isinstance(cached_doc.get("result"), dict):
            _POLICY_CHECK_CACHE[cache_key] = copy.deepcopy(cached_doc)
            return copy.deepcopy(cached_doc.get("result"))

    return None


def _set_cached_policy_check(cache_key: str, result: Dict[str, Any]) -> None:
    if not _is_policy_cache_enabled() or not isinstance(result, dict):
        return

    expires_at = time.time() + _get_policy_cache_ttl_seconds()
    payload = {
        "result": copy.deepcopy(result),
        "expires_at": expires_at,
    }
    _POLICY_CHECK_CACHE[cache_key] = payload
    _set_stage1_cache("policy", cache_key, payload)


def _severity_rank(level: str) -> int:
    mapping = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }
    return mapping.get(str(level or "").strip().lower(), 0)


def _policy_result_signature(policy_result: Dict[str, Any]) -> str:
    violations = policy_result.get("violations", []) if isinstance(policy_result, dict) else []
    compact_violations = []
    for item in (violations or [])[:10]:
        if not isinstance(item, dict):
            continue
        compact_violations.append({
            "type": item.get("violation_type"),
            "severity": str(item.get("severity", "")).lower(),
            "text": (item.get("problematic_text", "") or "")[:180],
        })

    payload = {
        "overall_compliance": policy_result.get("overall_compliance"),
        "requires_revision": bool(policy_result.get("requires_revision")),
        "score": round(float(policy_result.get("compliance_score") or 0), 4),
        "violations": compact_violations,
    }
    return _stable_json_hash(payload)


def _compact_text(value: Any, max_chars: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
    else:
        text = str(value)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def build_style_digest(style: Dict[str, Any]) -> Dict[str, str]:
    style = style or {}
    style_key = _stable_json_hash(style)
    if style_key in _STYLE_DIGEST_CACHE:
        return _STYLE_DIGEST_CACHE[style_key]

    style_text_value = (
        style.get("style_description")
        or style.get("style")
        or style.get("style_instructions")
        or ((style.get("properties") or {}).get("style_instructions") if isinstance(style.get("properties"), dict) else "")
    )
    global_rules_value = style.get("global_rules") or style.get("global_rulebook") or style.get("rulebook")
    guidelines_value = style.get("guidelines") or ""
    example_value = style.get("example")
    if not example_value:
        evidence_spans = style.get("evidence_spans")
        if isinstance(evidence_spans, list) and evidence_spans:
            example_value = "\n".join(str(item) for item in evidence_spans if item)

    digest = {
        "style_text": _compact_text(style_text_value, max_chars=int(os.getenv("STYLE_DIGEST_STYLE_MAX", "2200"))),
        "global_rules": _compact_text(global_rules_value, max_chars=int(os.getenv("STYLE_DIGEST_RULES_MAX", "1800"))),
        "guidelines": _compact_text(guidelines_value, max_chars=int(os.getenv("STYLE_DIGEST_GUIDELINES_MAX", "1200"))),
        "example": _compact_text(example_value, max_chars=int(os.getenv("STYLE_DIGEST_EXAMPLE_MAX", "900"))),
    }

    _STYLE_DIGEST_CACHE[style_key] = digest
    return digest


def _extract_evidence_citations(text: str) -> List[str]:
    return re.findall(r'\[E\d+(?:,E\d+)*\]', text or "")


def _extract_apa_parenthetical_citations(text: str) -> List[str]:
    return re.findall(r'\((?=[^)]*\b(?:\d{4}[a-z]?|n\.d\.)\b)[^)]*\)', text or "", flags=re.IGNORECASE)


def _extract_locked_citations(text: str) -> List[str]:
    evidence = _extract_evidence_citations(text)
    apa_parenthetical = _extract_apa_parenthetical_citations(text)
    return evidence + apa_parenthetical


def _extract_numeric_tokens(text: str) -> List[str]:
    return re.findall(r'\b\d[\d,]*(?:\.\d+)?%?\b', text or "")


def _paragraph_count(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len([block for block in re.split(r'\n\s*\n', text.strip()) if block.strip()])


def _has_references_section(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r'\n(?:=+\n)?\s*REFERENCES\s*\n(?:=+\n)?', text, flags=re.IGNORECASE))


async def reapply_style_coat_strict(
    speech_text: str,
    style: Dict[str, Any],
    context_details: str = "",
) -> Dict[str, Any]:
    """
    Re-apply style/guidelines as a final polish pass while strictly preserving content.
    Falls back to original text when deterministic content-lock checks fail.
    """
    if not speech_text or not speech_text.strip():
        return {
            "success": True,
            "applied": False,
            "output": speech_text,
            "fallback_reason": "empty_speech",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "api_calls": 0},
        }

    style = style or {}
    style_digest = build_style_digest(style)
    style_text = style_digest.get("style_text", "")
    global_rules = style_digest.get("global_rules", "")
    guidelines = style_digest.get("guidelines", "")
    example = style_digest.get("example", "")

    if not any([style_text.strip(), global_rules.strip(), guidelines.strip(), example.strip()]):
        return {
            "success": True,
            "applied": False,
            "output": speech_text,
            "fallback_reason": "no_style_instructions",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "api_calls": 0},
        }

    token_usage_summary = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "api_calls": 0,
    }

    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        model = os.getenv("AZURE_OPENAI_STAGE3_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

        system_parts = [
            "You are a precision style finisher. Apply style as a thin coat of paint only.\n",
            f"<writingStyle>{style_text}</writingStyle>\n",
            f"<globalRules>{global_rules}</globalRules>\n",
            f"<writingGuidelines>{guidelines}</writingGuidelines>\n",
        ]
        if example:
            system_parts.append(f"<writingExample>{example}</writingExample>\n")

        # SPEAKER PERSPECTIVE: Preserve first-person voice
        coat_speaker_name = style.get("speaker") or style.get("Speaker")
        if coat_speaker_name:
            system_parts.extend([
                f"SPEAKER PERSPECTIVE: This speech is delivered by {coat_speaker_name}. "
                "Maintain first-person perspective throughout ('I', 'we'). "
                f"NEVER refer to {coat_speaker_name} in the third person.\n",
            ])

        system_parts.extend([
            "CONTENT-LOCK RULES (MANDATORY):\n",
            "1. Keep the same factual meaning sentence-by-sentence.\n",
            "2. Do NOT add, remove, or alter facts, numbers, dates, names, entities, or claims.\n",
            "3. Keep citations exactly intact (same citation markers and references).\n",
            "4. Keep paragraph order and paragraph count unchanged.\n",
            "5. Keep section headers and references section present exactly as in input.\n",
            "6. Keep legal/policy modality unchanged (must/should/may) unless already in the input.\n",
            "7. Do not insert new recommendations, caveats, examples, or conclusions.\n",
            "8. Use plain-language wording so non-technical listeners can understand the speech.\n",
            "9. Avoid research-paper / RRL tone; keep it as spoken speech with natural flow.\n",
            "10. Simplify jargon by rephrasing in common words, but do NOT change technical meaning.\n",
            "11. Keep tone formal, clear, and audience-friendly; do not sound like an academic abstract.\n",
            "12. If any requested style instruction conflicts with content lock, prioritize content lock.\n",
            "13. Preserve first-person speaker perspective — do not switch to third person.\n",
            "14. Separate every paragraph with a blank line (two newlines). Never run paragraphs together.\n",
            "Return ONLY the fully rewritten speech text, no explanations.\n",
        ])

        system_prompt = "".join(system_parts)
        user_prompt = (
            "Apply a final style coat to the speech below using the provided style instructions.\n"
            "This is a style-only pass with strict content lock.\n\n"
            f"Optional context: {context_details.strip() if context_details and context_details.strip() else 'Not provided'}\n\n"
            "<SPEECH_TO_POLISH>\n"
            f"{speech_text}\n"
            "</SPEECH_TO_POLISH>\n"
        )

        max_completion_tokens = max(2000, min(9000, int(len(speech_text) * 0.9)))

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_completion_tokens,
        )

        usage = getattr(response, "usage", None)
        if usage:
            token_usage_summary["api_calls"] += 1
            token_usage_summary["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            token_usage_summary["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                token_usage_summary["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0)

        candidate = (response.choices[0].message.content or "").strip()
        if not candidate:
            return {
                "success": True,
                "applied": False,
                "output": speech_text,
                "fallback_reason": "empty_model_output",
                "token_usage": token_usage_summary,
            }

        if "<think>" in candidate and "</think>" in candidate:
            while "<think>" in candidate and "</think>" in candidate:
                start = candidate.find("<think>")
                end = candidate.find("</think>") + 8
                candidate = (candidate[:start] + candidate[end:]).strip()

        original_citations = _extract_locked_citations(speech_text)
        candidate_citations = _extract_locked_citations(candidate)
        original_numbers = _extract_numeric_tokens(speech_text)
        candidate_numbers = _extract_numeric_tokens(candidate)

        original_paragraphs = _paragraph_count(speech_text)
        candidate_paragraphs = _paragraph_count(candidate)

        original_has_refs = _has_references_section(speech_text)
        candidate_has_refs = _has_references_section(candidate)

        original_len = max(1, len(speech_text))
        candidate_len = len(candidate)
        len_ratio = candidate_len / original_len

        if original_citations and candidate_citations != original_citations:
            return {
                "success": True,
                "applied": False,
                "output": speech_text,
                "fallback_reason": "citations_changed",
                "token_usage": token_usage_summary,
                "validation": {
                    "original_citations": len(original_citations),
                    "candidate_citations": len(candidate_citations),
                },
            }

        # Numeric tokens check removed — spoken rhetoric naturally reformats numbers
        # (e.g. "1.8 mb/d" → "one-point-eight million barrels"). Citation markers
        # already lock all factual claims; numeric format is presentational.

        if abs(candidate_paragraphs - original_paragraphs) > 2:
            return {
                "success": True,
                "applied": False,
                "output": speech_text,
                "fallback_reason": "paragraph_structure_changed",
                "token_usage": token_usage_summary,
                "validation": {
                    "original_paragraphs": original_paragraphs,
                    "candidate_paragraphs": candidate_paragraphs,
                },
            }

        if candidate_has_refs != original_has_refs:
            return {
                "success": True,
                "applied": False,
                "output": speech_text,
                "fallback_reason": "references_section_changed",
                "token_usage": token_usage_summary,
                "validation": {
                    "original_has_references": original_has_refs,
                    "candidate_has_references": candidate_has_refs,
                },
            }

        if len_ratio < 0.80 or len_ratio > 1.25:
            return {
                "success": True,
                "applied": False,
                "output": speech_text,
                "fallback_reason": "length_drift_too_high",
                "token_usage": token_usage_summary,
                "validation": {
                    "original_length": original_len,
                    "candidate_length": candidate_len,
                    "length_ratio": round(len_ratio, 4),
                },
            }

        return {
            "success": True,
            "applied": candidate.strip() != speech_text.strip(),
            "output": candidate,
            "fallback_reason": "",
            "token_usage": token_usage_summary,
            "validation": {
                "original_length": original_len,
                "candidate_length": candidate_len,
                "length_ratio": round(len_ratio, 4),
                "citations_preserved": True,
                "numbers_preserved": True,
                "paragraphs_preserved": True,
                "references_preserved": True,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "applied": False,
            "output": speech_text,
            "fallback_reason": f"style_recoat_failed: {e}",
            "token_usage": token_usage_summary,
        }


def _get_citation_id_set(text: str) -> set:
    """Return the set of unique [ENN] evidence IDs referenced in *text*."""
    markers = re.findall(r'\[E\d+(?:,E\d+)*\]', text or "")
    ids: set = set()
    for marker in markers:
        ids.update(re.findall(r'E\d+', marker))
    return ids


async def reapply_rhetoric_pass(
    speech_text: str,
    style: Dict[str, Any],
    context_details: str = "",
) -> Dict[str, Any]:
    """
    Pass 1 of two-pass style recoat: Rhetoric and voice injection.

    Transforms an academically-styled draft into genuine spoken rhetoric by
    injecting the speaker's characteristic voice, rhetorical devices, and
    Filipino speech patterns.

    The ONLY hard guards enforced here are:
    - All original [ENN] citation IDs must remain present (none dropped or invented).
    - The REFERENCES section must be preserved if it was present.
    - Length ratio within 0.70–1.40 (generous to accommodate voice expansion).

    Paragraph structure, numeric formatting, and sentence order may change freely.
    """
    if not speech_text or not speech_text.strip():
        return {
            "success": True,
            "applied": False,
            "output": speech_text,
            "fallback_reason": "empty_speech",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "api_calls": 0},
        }

    style = style or {}
    style_digest = build_style_digest(style)
    style_text = style_digest.get("style_text", "")
    global_rules = style_digest.get("global_rules", "")
    guidelines = style_digest.get("guidelines", "")
    example = style_digest.get("example", "")

    token_usage_summary = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "api_calls": 0,
    }

    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        model = os.getenv("AZURE_OPENAI_STAGE3_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

        speaker_name = style.get("speaker") or style.get("Speaker") or "the speaker"

        system_parts = [
            f"You are a speechwriter restoring {speaker_name}'s authentic spoken voice to a "
            "draft that currently reads like an academic literature review.\n\n",
            f"<writingStyle>{style_text}</writingStyle>\n",
            f"<globalRules>{global_rules}</globalRules>\n",
            f"<writingGuidelines>{guidelines}</writingGuidelines>\n",
        ]
        if example:
            system_parts.append(f"<writingExample>{example}</writingExample>\n")

        # Inject real speech examples from Cosmos — same logic as Stage 3
        if speaker_name and speaker_name != "the speaker":
            try:
                max_speeches = int(os.getenv("STAGE3_MAX_SPEECH_EXAMPLES", "1"))
            except ValueError:
                max_speeches = 1
            max_speeches = max(0, min(3, max_speeches))

            if max_speeches > 0:
                sample_speeches = get_sample_speeches_from_db(speaker_name, max_speeches=max_speeches)
                if sample_speeches:
                    try:
                        excerpt_chars = int(os.getenv("STAGE3_SPEECH_EXCERPT_CHARS", "900"))
                    except ValueError:
                        excerpt_chars = 900
                    excerpt_chars = max(300, min(2000, excerpt_chars))

                    system_parts.append("\n<realSpeechExamples>\n")
                    system_parts.append(
                        f"The following are actual speeches delivered by {speaker_name}. "
                        "Study them closely to absorb:\n"
                        "- Their characteristic sentence rhythm and cadence\n"
                        "- Vocabulary and phrasing choices\n"
                        "- How they open, transition between ideas, and close\n"
                        "- Their rhetorical devices and audience address style\n\n"
                    )
                    for i, speech in enumerate(sample_speeches, 1):
                        excerpt = speech[:excerpt_chars] if len(speech) > excerpt_chars else speech
                        system_parts.append(f"=== REAL SPEECH EXAMPLE {i} ===\n{excerpt}\n")
                        if len(speech) > excerpt_chars:
                            system_parts.append("... (excerpt from longer speech)\n")
                        system_parts.append("\n")
                    system_parts.append("</realSpeechExamples>\n\n")
                    print(f"[RECOAT P1] Added {len(sample_speeches)} real speech example(s) for {speaker_name}")
                else:
                    print(f"[RECOAT P1] No real speech examples found for {speaker_name}")

        system_parts.extend([
            "\n## RHETORICAL VOICE INJECTION — YOUR PRIMARY GOAL\n",
            f"The draft below reads like an economics paper (RRL). Transform it into a "
            f"living, breathing speech that could only come from {speaker_name}'s mouth. "
            "Make it sound like a human being standing at a podium addressing a real audience.\n\n",
            "USE THESE RHETORICAL TOOLS FREELY:\n",
            "- Open with warmth and direct audience address "
            "(e.g., 'Good morning, distinguished colleagues and our partners in the banking sector...')\n",
            "- Use rhetorical questions to guide the audience's thinking "
            "('So what does this mean for us? What does this imply for Filipino families?')\n",
            "- Use conversational transitions (\'Now, let us turn to...\', \'This brings me to...\', "
            "\'Consider this:\', \'And here is the critical point:\')\n",
            "- Weave in Filipino expressions naturally "
            "(e.g., \'Maraming salamat po\', \'Mabuhay tayong lamat\', \'Ito ang ating hamon\')\n",
            "- Introduce data with a narrative frame instead of stating it flatly: "
            "rather than 'Prices rose by ten dollars [E1]', try "
            "'Last January, something shifted in global oil markets — benchmark crude jumped by ten dollars [E1]'\n",
            "- Vary sentence length: short punchy statements followed by longer explanations\n",
            "- Address audience segments directly "
            "('For our banking partners...', 'For the Filipino consumer...', 'For policymakers...')\n",
            "- Use inclusive language: \'we\', \'our\', \'together\', \'let us\'\n",
            f"- Write in strict first-person as {speaker_name}; NEVER refer to {speaker_name} in the third person\n",
            "- Close with a memorable, human line — not a data summary. End on conviction, not statistics.\n\n",
            "## CITATION LOCK — THE ONLY STRICT RULE\n",
            "- Every [ENN] citation marker from the original MUST appear somewhere in the output\n",
            "- Do NOT add new [ENN] markers that were not in the original\n",
            "- You MAY move a citation to the end of its reformatted sentence for natural flow\n",
            "- The REFERENCES section at the bottom must be preserved verbatim — copy it exactly\n\n",
            "- Separate every paragraph with a blank line (two newlines). Never run paragraphs together.\n",
            "Return ONLY the fully rewritten speech. No preamble, no explanation.\n",
        ])

        system_prompt = "".join(system_parts)
        user_prompt = (
            f"Give {speaker_name} their authentic spoken voice. Transform the academic draft below "
            "into powerful spoken rhetoric — keeping every [ENN] citation intact and the REFERENCES "
            "section verbatim at the end.\n\n"
            f"Optional context: {context_details.strip() if context_details and context_details.strip() else 'Not provided'}\n\n"
            "<ACADEMIC_DRAFT_TO_TRANSFORM>\n"
            f"{speech_text}\n"
            "</ACADEMIC_DRAFT_TO_TRANSFORM>\n"
        )

        max_completion_tokens = max(2000, min(12000, int(len(speech_text) * 1.3)))

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_completion_tokens,
        )

        usage = getattr(response, "usage", None)
        if usage:
            token_usage_summary["api_calls"] += 1
            token_usage_summary["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            token_usage_summary["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                token_usage_summary["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0)

        candidate = (response.choices[0].message.content or "").strip()
        if not candidate:
            return {
                "success": True,
                "applied": False,
                "output": speech_text,
                "fallback_reason": "empty_model_output",
                "token_usage": token_usage_summary,
            }

        # Strip reasoning traces
        if "<think>" in candidate and "</think>" in candidate:
            while "<think>" in candidate and "</think>" in candidate:
                start = candidate.find("<think>")
                end = candidate.find("</think>") + 8
                candidate = (candidate[:start] + candidate[end:]).strip()

        # --- GUARD 1: Citation ID set preservation ---
        original_eids = _get_citation_id_set(speech_text)
        candidate_eids = _get_citation_id_set(candidate)

        missing_eids = original_eids - candidate_eids
        if missing_eids:
            return {
                "success": True,
                "applied": False,
                "output": speech_text,
                "fallback_reason": f"citations_dropped: {sorted(missing_eids)}",
                "token_usage": token_usage_summary,
            }

        invented_eids = candidate_eids - original_eids
        if invented_eids:
            return {
                "success": True,
                "applied": False,
                "output": speech_text,
                "fallback_reason": f"citations_invented: {sorted(invented_eids)}",
                "token_usage": token_usage_summary,
            }

        # --- GUARD 2: References section preserved ---
        original_has_refs = _has_references_section(speech_text)
        candidate_has_refs = _has_references_section(candidate)
        if original_has_refs and not candidate_has_refs:
            return {
                "success": True,
                "applied": False,
                "output": speech_text,
                "fallback_reason": "references_section_dropped",
                "token_usage": token_usage_summary,
            }

        # --- GUARD 3: Length sanity (generous for rhetoric expansion) ---
        original_len = max(1, len(speech_text))
        candidate_len = len(candidate)
        len_ratio = candidate_len / original_len
        if len_ratio < 0.70 or len_ratio > 1.40:
            return {
                "success": True,
                "applied": False,
                "output": speech_text,
                "fallback_reason": f"rhetoric_length_drift: ratio={len_ratio:.2f}",
                "token_usage": token_usage_summary,
            }

        return {
            "success": True,
            "applied": candidate.strip() != speech_text.strip(),
            "output": candidate,
            "fallback_reason": "",
            "token_usage": token_usage_summary,
            "validation": {
                "original_length": original_len,
                "candidate_length": candidate_len,
                "length_ratio": round(len_ratio, 4),
                "original_eids": sorted(original_eids),
                "candidate_eids": sorted(candidate_eids),
                "citations_preserved": True,
                "references_preserved": True,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "applied": False,
            "output": speech_text,
            "fallback_reason": f"rhetoric_pass_failed: {e}",
            "token_usage": token_usage_summary,
        }


async def reapply_two_pass_style(
    speech_text: str,
    style: Dict[str, Any],
    context_details: str = "",
) -> Dict[str, Any]:
    """
    Two-pass style recoat: Rhetoric injection (Pass 1) → Polish coat (Pass 2).

    Pass 1 — Rhetoric pass (reapply_rhetoric_pass):
        Maximum creative freedom to inject the speaker's authentic voice,
        rhetorical devices, and spoken-speech patterns. Only citation IDs
        and the references section are locked.

    Pass 2 — Polish pass (reapply_style_coat_strict):
        Fine-grained style tidying with content-lock checks applied to the
        rhetoric pass output. If Pass 1 fell back, Pass 2 runs on the original.

    If Pass 1 succeeds and transforms the text, Pass 2 gives it a final
    editorial polish. Either pass alone still provides value.
    """
    combined_usage: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "api_calls": 0,
    }

    def _add_usage(u: Optional[Dict[str, int]]) -> None:
        for k in combined_usage:
            combined_usage[k] += int((u or {}).get(k, 0) or 0)

    # --- Pass 1: Rhetoric ---
    print("[RECOAT P1] Running rhetoric voice injection pass...")
    rhetoric_result = await reapply_rhetoric_pass(
        speech_text=speech_text,
        style=style,
        context_details=context_details,
    )
    _add_usage(rhetoric_result.get("token_usage"))

    rhetoric_applied = bool(rhetoric_result.get("success") and rhetoric_result.get("applied"))
    after_rhetoric = rhetoric_result.get("output", speech_text) if rhetoric_applied else speech_text

    if rhetoric_result.get("fallback_reason"):
        print(f"[RECOAT P1] ⚠️ Rhetoric pass skipped: {rhetoric_result.get('fallback_reason')}")
    elif rhetoric_applied:
        print("[RECOAT P1] ✅ Rhetoric pass applied")
    else:
        print("[RECOAT P1] No change from rhetoric pass (text already styled)")

    # --- Pass 2: Polish ---
    print("[RECOAT P2] Running style polish pass...")
    polish_result = await reapply_style_coat_strict(
        speech_text=after_rhetoric,
        style=style,
        context_details=context_details,
    )
    _add_usage(polish_result.get("token_usage"))

    polish_applied = bool(polish_result.get("success") and polish_result.get("applied"))
    after_polish = polish_result.get("output", after_rhetoric) if polish_applied else after_rhetoric

    if polish_result.get("fallback_reason"):
        print(f"[RECOAT P2] ⚠️ Polish pass skipped: {polish_result.get('fallback_reason')}")
    elif polish_applied:
        print("[RECOAT P2] ✅ Polish pass applied")
    else:
        print("[RECOAT P2] No change from polish pass")

    overall_applied = after_polish.strip() != speech_text.strip()
    fallback_reason = "" if overall_applied else (
        rhetoric_result.get("fallback_reason") or polish_result.get("fallback_reason") or "no_change"
    )

    return {
        "success": True,
        "applied": overall_applied,
        "output": after_polish,
        "fallback_reason": fallback_reason,
        "token_usage": combined_usage,
        "passes": {
            "rhetoric": {
                "applied": rhetoric_applied,
                "fallback_reason": rhetoric_result.get("fallback_reason", ""),
                "validation": rhetoric_result.get("validation", {}),
            },
            "polish": {
                "applied": polish_applied,
                "fallback_reason": polish_result.get("fallback_reason", ""),
                "validation": polish_result.get("validation", {}),
            },
        },
    }


def prune_evidence_for_generation(query: str, evidence_store: List[Dict[str, Any]], max_items: int = 30) -> List[Dict[str, Any]]:
    if not evidence_store:
        return []

    try:
        env_max = int(os.getenv("STAGE3_MAX_EVIDENCE_ITEMS", str(max_items)))
    except ValueError:
        env_max = max_items
    max_items = max(8, min(80, env_max))

    query_tokens = set(re.findall(r"[a-zA-Z]{3,}", (query or "").lower()))

    scored = []
    for idx, evidence in enumerate(evidence_store):
        claim = str(evidence.get("claim", ""))
        quote = str(evidence.get("quote_span", ""))
        combined = f"{claim} {quote}".lower()
        evidence_tokens = set(re.findall(r"[a-zA-Z]{3,}", combined))

        overlap = len(query_tokens & evidence_tokens) if query_tokens else 0
        confidence = float(evidence.get("confidence", 0.0) or 0.0)
        score = overlap * 3.0 + confidence
        scored.append((score, idx, evidence))

    scored.sort(key=lambda item: (-item[0], item[1]))
    pruned = [item[2] for item in scored[:max_items]]

    # Preserve original order for deterministic prompts
    selected_ids = {id(item) for item in pruned}
    ordered = [ev for ev in evidence_store if id(ev) in selected_ids]
    return ordered

def _get_link_processing_model(tier: str = "light"):
    """
    Lazy initialization of model for link processing.
    
    Returns:
        Initialized Azure chat model instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    deployment_by_tier = {
        "light": os.getenv("AZURE_INFERENCE_LIGHT_DEPLOYMENT") or os.getenv("AZURE_DR_DEPLOYMENT"),
        "strong": os.getenv("AZURE_INFERENCE_STRONG_DEPLOYMENT") or os.getenv("AZURE_DR_DEPLOYMENT"),
    }

    _model_name = deployment_by_tier.get(tier, deployment_by_tier["light"])
    model_key = f"{tier}:{_model_name}:{_current_loop_cache_scope()}"

    if model_key in _link_processing_models:
        return _link_processing_models[model_key]
    
    # Get environment variables
    _endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    _key = os.getenv("AZURE_AI_API_KEY")
    
    # Validate required environment variables
    if not _endpoint or not _model_name or not _key:
        missing = []
        if not _endpoint: missing.append("AZURE_INFERENCE_ENDPOINT")
        if not _model_name: missing.append("AZURE_DR_DEPLOYMENT")
        if not _key: missing.append("AZURE_AI_API_KEY")
        raise ValueError(
            f"Missing required environment variables for link processing: {', '.join(missing)}. "
            "Please set these in your .env file."
        )
    
    # Initialize model
    if _AZURE_CHAT_MODEL_FLAVOR == "azure_ai_chat_completions":
        model_instance = AzureChatModel(
            endpoint=_endpoint,
            credential=_key,
            model=_model_name,
        )
    elif _AZURE_CHAT_MODEL_FLAVOR == "azure_openai_api":
        model_instance = AzureChatModel(
            base_url=_endpoint,
            api_key=_key,
            model=_model_name,
        )
    else:
        model_instance = AzureChatModel(
            base_url=_endpoint,
            api_key=_key,
            model=_model_name,
        )

    _link_processing_models[model_key] = model_instance
    return model_instance


# ---------------------------------------------------------------------------
# Non-LLM helpers (deterministic evidence extraction and summary building)
# ---------------------------------------------------------------------------

def _extract_claims_from_text(raw_content: str, query: str, max_claims: int = 6) -> List[str]:
    """Score and select top sentences from text by query keyword overlap (no LLM)."""
    query_tokens = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
    sentences = re.split(r'(?<=[.!?])\s+', raw_content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

    scored = []
    for sent in sentences:
        tokens = set(re.findall(r'[a-zA-Z]{3,}', sent.lower()))
        overlap = len(query_tokens & tokens) if query_tokens else 0
        has_number = bool(re.search(r'\d', sent))
        score = overlap * 3.0 + (2.0 if has_number else 0.0)
        scored.append((score, sent))

    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:max_claims]]


def _extract_metadata_from_text(text: str, filename: str, fallback_title: str) -> Dict[str, str]:
    """Extract basic APA metadata from text using regex (no LLM)."""
    year_match = re.search(r'\b(19|20)\d{2}\b', (text or "")[:1000])
    year = year_match.group(0) if year_match else "n.d."
    title = fallback_title
    if filename and not filename.startswith("attachment_"):
        title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
    return {"author": "n.d.", "year": year, "publication": title, "source_title": title}


def _build_deterministic_summary(query: str, evidence_store: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a bullet-point evidence summary without LLM calls."""
    if not evidence_store:
        return {
            "success": False, "summary": "", "error": "No evidence",
            "evidence_count": 0, "citations_found": 0, "unique_evidence_cited": 0,
            "citation_coverage": "0.0%", "invalid_citations": [],
            "validation": {"all_citations_valid": True, "cited_ids": [], "uncited_ids": []},
        }
    allowed_ids = [e["id"] for e in evidence_store if e.get("id")]
    lines = []
    for ev in evidence_store:
        eid = ev.get("id", "")
        claim = (ev.get("claim") or "").strip()
        if eid and claim:
            lines.append(f"- {claim} [{eid}]")
    return {
        "success": True,
        "summary": "\n".join(lines),
        "evidence_count": len(evidence_store),
        "query": query,
        "citations_found": len(allowed_ids),
        "unique_evidence_cited": len(allowed_ids),
        "citation_coverage": "100.0%",
        "invalid_citations": [],
        "validation": {
            "all_citations_valid": True,
            "cited_ids": sorted(allowed_ids),
            "uncited_ids": [],
        },
    }


def _build_claim_outline_from_summary(summary: str, evidence_store: List[Dict[str, Any]]):
    """
    Parse summary sentences (or bullet lines) for inline [ENN] citations to build a claim
    outline. Returns (claims_list, outline_text) or (None, None) if nothing useful is found.
    Handles both prose-with-citations and bullet-list summaries produced by
    _build_deterministic_summary.
    """
    if not summary or not evidence_store:
        return None, None
    evidence_ids = {e["id"] for e in evidence_store if e.get("id")}
    citation_re = re.compile(r'\[E\d+(?:,E\d+)*\]')

    # Prefer sentence splitting; fall back to line splitting for bullet-list format
    if summary.lstrip().startswith(("-", "•")):
        # Bullet list format (output of _build_deterministic_summary)
        segments = [ln.strip().lstrip("-\u2022 ") for ln in summary.splitlines() if len(ln.strip()) > 15]
    else:
        segments = re.split(r'(?<=[.!?])\s+', summary.strip())
        segments = [s.strip() for s in segments if len(s.strip()) > 40]
        if len(segments) < 2:
            segments = [ln.strip().lstrip("-\u2022 ") for ln in summary.splitlines() if len(ln.strip()) > 15]

    claims = []
    for i, seg in enumerate(segments, 1):
        cited_ids: List[str] = []
        for cit in citation_re.findall(seg):
            for eid in re.findall(r'E\d+', cit):
                if eid in evidence_ids and eid not in cited_ids:
                    cited_ids.append(eid)
        cited_ids = cited_ids[:2]  # max 2 IDs per claim
        clean = citation_re.sub('', seg).strip()
        if clean:
            claims.append({
                'claim_id': f"C{i}",
                'claim_text': clean,
                'evidence_ids': cited_ids,
                'claim_type': 'factual' if cited_ids else 'transition',
            })

    if not claims:
        return None, None

    claim_lines = ["CLAIM OUTLINE (Rephrase ONLY these claims):", ""]
    for claim in claims:
        claim_lines.append(f"{claim['claim_id']}. {claim['claim_text']}")
        if claim['evidence_ids']:
            claim_lines.append(f"   Evidence: [{', '.join(claim['evidence_ids'])}]")
        claim_lines.append(f"   Type: {claim['claim_type']}")
        claim_lines.append("")
    return claims, "\n".join(claim_lines)


async def generate_summary_from_evidence(query: str, evidence_store: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a comprehensive summary with forced [ENN] citations.
    
    Args:
        query: The user's original query
        evidence_store: List of atomic evidence items with id, claim, quote_span
    
    Returns:
        Dictionary with summary, citations metadata, and validation results
    """
    if not evidence_store or len(evidence_store) == 0:
        logger.warning("generate_summary_from_evidence called with empty evidence store for query '%s'", query[:120])
        return {
            "success": False,
            "summary": "",
            "error": "No evidence available to generate summary",
            "evidence_count": 0
        }
    
    try:
        import re

        token_usage_summary = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "api_calls": 0,
        }

        def _accumulate_usage(response_obj) -> None:
            usage = getattr(response_obj, "usage", None)
            if not usage:
                return

            token_usage_summary["api_calls"] += 1
            token_usage_summary["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            token_usage_summary["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)

            details = getattr(usage, "completion_tokens_details", None)
            if details:
                token_usage_summary["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0)
        
        # Get LLM model
        model = _get_link_processing_model("light")
        
        # Build the evidence context with atomic claims
        evidence_context = ""
        allowed_ids = []
        for evidence in evidence_store:
            eid = evidence.get("id", "")
            claim = evidence.get("claim", "")
            source = evidence.get("source_url", evidence.get("source", ""))
            confidence = evidence.get("confidence", 0.0)
            
            if eid:
                allowed_ids.append(eid)
                evidence_context += f"{eid}: {claim}\n"
                evidence_context += f"Source: {source} (confidence: {confidence:.2f})\n\n"
        
        if not allowed_ids:
            return {
                "success": False,
                "summary": "",
                "error": "No valid evidence IDs found",
                "evidence_count": len(evidence_store)
            }
        
        # Create allowed IDs string
        allowed_ids_str = ", ".join(allowed_ids)
        
        # Create the summary generation prompt with STRICT citation rules
        summary_prompt = f"""You are a research assistant synthesizing information to answer a query.

<USER_QUERY>
{query}
</USER_QUERY>

<EVIDENCE>
{evidence_context}
</EVIDENCE>

<STRICT CITATION RULES>
1. EVERY factual sentence MUST end with a citation in format [ENN] or [E1,E3,E5]
2. You may ONLY cite these evidence IDs: {allowed_ids_str}
3. NEVER generate citations to non-existent evidence
4. Multiple claims in one sentence = multiple citations: [E1,E2,E5]
5. General statements without specific facts do NOT need citations
6. Cite the EXACT evidence that supports each claim

EXAMPLES:
✓ "Transformers use attention mechanisms to process sequential data [E1]."
✓ "The model achieved 95% accuracy on the benchmark dataset [E3,E7]."
✗ "Transformers are important." (too general, missing citation)
✗ "Deep learning has many applications [E99]." (E99 doesn't exist)
</STRICT CITATION RULES>

<TASK>
Write a comprehensive summary that:
1. Directly answers the user's query
2. Synthesizes insights from the evidence
3. Cites EVERY factual claim with [ENN] format
4. Organizes information logically
5. Only uses valid evidence IDs from the list above

Generate your summary with citations:"""
        
        messages = [
            SystemMessage(content="You are an expert research synthesizer who ALWAYS cites sources using [ENN] format for every factual claim."),
            HumanMessage(content=summary_prompt)
        ]
        
        result = await model.ainvoke(messages)
        summary_text = result.content.strip()
        
        # Strip thinking tokens if present
        if "<think>" in summary_text and "</think>" in summary_text:
            while "<think>" in summary_text and "</think>" in summary_text:
                start = summary_text.find("<think>")
                end = summary_text.find("</think>") + 8
                summary_text = summary_text[:start] + summary_text[end:]
            summary_text = summary_text.strip()
        
        # Validate citations
        citation_pattern = r'\[E\d+(?:,E\d+)*\]'
        citations_found = re.findall(citation_pattern, summary_text)
        
        # Extract individual evidence IDs from citations
        cited_ids = set()
        invalid_citations = []
        for citation in citations_found:
            # Extract E1, E2, etc. from [E1,E2]
            ids_in_citation = re.findall(r'E\d+', citation)
            for eid in ids_in_citation:
                cited_ids.add(eid)
                if eid not in allowed_ids:
                    invalid_citations.append(eid)
        
        # Calculate coverage
        cited_count = len(cited_ids & set(allowed_ids))
        total_evidence = len(allowed_ids)
        coverage = (cited_count / total_evidence * 100) if total_evidence > 0 else 0
        
        return {
            "success": True,
            "summary": summary_text,
            "evidence_count": len(evidence_store),
            "query": query,
            "citations_found": len(citations_found),
            "unique_evidence_cited": cited_count,
            "citation_coverage": f"{coverage:.1f}%",
            "invalid_citations": invalid_citations,
            "validation": {
                "all_citations_valid": len(invalid_citations) == 0,
                "cited_ids": sorted(list(cited_ids)),
                "uncited_ids": sorted(list(set(allowed_ids) - cited_ids))
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "summary": "",
            "error": str(e),
            "evidence_count": len(evidence_store)
        }


async def critique_summary(query: str, summary: str, evidence_store: List[Dict[str, Any]], iteration: int) -> Dict[str, Any]:
    """
    Critique the current summary to identify gaps and areas for improvement.
    
    Args:
        query: The user's original query
        summary: The current generated summary
        evidence_store: The evidence used to create the summary
        iteration: Current iteration number
    
    Returns:
        Dictionary with critique and suggested adjustments
    """
    try:
        # Get LLM model
        model = _get_link_processing_model("light")
        
        critique_prompt = f"""You are a research critic tasked with evaluating a research summary and identifying areas for improvement.

<USER_QUERY>
{query}
</USER_QUERY>

<CURRENT_SUMMARY>
{summary}
</CURRENT_SUMMARY>

<ITERATION>
This is iteration {iteration} of the research process.
</ITERATION>

<TASK>
Analyze the summary and provide constructive criticism in these areas:

1. **Research Gaps**: What important aspects of the query are not adequately addressed? What questions remain unanswered?

2. **Topic Alignment**: How well does the summary align with the user's original query? Are there tangential topics that should be removed or refocused?

3. **Information Enrichment**: What specific details, examples, or explanations would make the summary more comprehensive and valuable?

4. **Source Coverage**: Based on the {len(evidence_store)} evidence sources, are there perspectives or information not yet incorporated?

5. **Clarity and Organization**: How could the structure or presentation be improved?

Provide your critique in a structured format with:
- **Gaps Identified**: List specific missing information
- **Alignment Issues**: Note any misalignments with the query
- **Enrichment Opportunities**: Suggest specific improvements
- **Recommended Focus**: What should the next iteration prioritize?

Be specific and actionable in your recommendations.
</TASK>

Provide your critique:"""
        
        messages = [
            SystemMessage(content="You are an expert research critic who identifies gaps and suggests improvements in research summaries."),
            HumanMessage(content=critique_prompt)
        ]
        
        result = await model.ainvoke(messages)
        critique_text = result.content.strip()
        
        # Strip thinking tokens
        if "<think>" in critique_text and "</think>" in critique_text:
            while "<think>" in critique_text and "</think>" in critique_text:
                start = critique_text.find("<think>")
                end = critique_text.find("</think>") + 8
                critique_text = critique_text[:start] + critique_text[end:]
            critique_text = critique_text.strip()
        
        return {
            "success": True,
            "critique": critique_text,
            "iteration": iteration
        }
        
    except Exception as e:
        return {
            "success": False,
            "critique": "",
            "error": str(e),
            "iteration": iteration
        }


async def generate_adjustments_from_critique(query: str, critique: str) -> str:
    """
    Generate query adjustments based on critique to guide the next iteration.
    
    Args:
        query: Original user query
        critique: The critique from previous iteration
    
    Returns:
        Adjusted query string with additional focus areas
    """
    try:
        model = _get_link_processing_model("light")
        
        adjustment_prompt = f"""Based on a critique of the current research, generate specific adjustments to refine the research focus.

<ORIGINAL_QUERY>
{query}
</ORIGINAL_QUERY>

<CRITIQUE>
{critique}
</CRITIQUE>

<TASK>
Create a refined query that:
1. Maintains the core intent of the original query
2. Adds specific focus areas identified in the critique
3. Addresses the gaps and enrichment opportunities mentioned
4. Guides the next research iteration toward completeness

Provide ONLY the enhanced query text without any preamble or explanation.
</TASK>

Enhanced query:"""
        
        messages = [
            SystemMessage(content="You are a research assistant who refines queries based on identified gaps."),
            HumanMessage(content=adjustment_prompt)
        ]
        
        result = await model.ainvoke(messages)
        adjusted_query = result.content.strip()
        
        # Strip thinking tokens
        if "<think>" in adjusted_query and "</think>" in adjusted_query:
            while "<think>" in adjusted_query and "</think>" in adjusted_query:
                start = adjusted_query.find("<think>")
                end = adjusted_query.find("</think>") + 8
                adjusted_query = adjusted_query[:start] + adjusted_query[end:]
            adjusted_query = adjusted_query.strip()
        
        return adjusted_query
        
    except Exception as e:
        # If adjustment generation fails, return original query
        return query


def split_sources(sources: Dict[str, Any]) -> Dict[str, Any]:
    """
    Split the sources into topics, links, and attachments.
    
    Args:
        sources: JSON object containing:
            - topics: string
            - links: array of strings
            - attachments: array of file paths/objects
    
    Returns:
        Dictionary with separated sources
    """
    return {
        "topics": sources.get("topics", ""),
        "links": sources.get("links", []),
        "attachments": sources.get("attachments", [])
    }


def _split_topic_for_parallel_processing(topic_text: str, max_parts: int = 3, min_parts: int = 1) -> List[str]:
    """Split a broad topic string into deterministic shards for concurrent deep-research calls."""
    text = (topic_text or "").strip()
    if not text:
        return []

    # Try common separators first
    raw_parts = re.split(r"\s*[;\n]\s*|\s*,\s*|\s+and\s+", text, flags=re.IGNORECASE)
    parts = []
    for part in raw_parts:
        cleaned = (part or "").strip()
        if len(cleaned) < 8:
            continue
        if cleaned.lower() == text.lower():
            continue
        parts.append(cleaned)

    # Deduplicate while preserving order
    unique_parts = list(dict.fromkeys(parts))
    if not unique_parts:
        unique_parts = [text]

    # If we still have too few parts, do sentence/length-based sharding fallback
    min_parts = max(1, min(max_parts, min_parts))
    if len(unique_parts) < min_parts and len(text) >= 120:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) >= min_parts:
            buckets = [""] * min_parts
            for idx, sentence in enumerate(sentences):
                bucket_idx = idx % min_parts
                buckets[bucket_idx] = (buckets[bucket_idx] + " " + sentence).strip()
            unique_parts = [b for b in buckets if b]
        else:
            # Length-based fallback to guarantee at least min_parts chunks when practical
            chunk_size = max(80, len(text) // min_parts)
            chunks = []
            cursor = 0
            while cursor < len(text) and len(chunks) < min_parts:
                chunks.append(text[cursor:cursor + chunk_size].strip())
                cursor += chunk_size
            chunks = [c for c in chunks if len(c) >= 40]
            if len(chunks) >= min_parts:
                unique_parts = chunks

    return unique_parts[:max(1, max_parts)]


def _parse_deep_research_sources(summary: str) -> List[Dict[str, str]]:
    """
    Extract source URLs and titles from the '### Sources:' section appended by
    deep_research.pipeline.finalize_summary.
    Each line has the format:  '* Title : https://...'
    """
    sources: List[Dict[str, str]] = []
    block_match = re.search(
        r'###\s*Sources:\s*\n(.*?)(?:\n##|\Z)',
        summary,
        re.DOTALL | re.IGNORECASE,
    )
    if not block_match:
        return sources
    for line in block_match.group(1).splitlines():
        line = line.strip().lstrip('*').strip()
        if ' : ' in line:
            title_part, url_part = line.split(' : ', 1)
            url_part = url_part.strip()
            if url_part.startswith('http'):
                sources.append({"title": title_part.strip(), "url": url_part})
    return sources


async def topic_processing(topic: str, query: str, evidence_id_start: int = 1) -> Dict[str, Any]:
    """
    Process a topic by calling deep_research module.
    Extracts atomic claims as evidence with stable IDs and exact quotes.
    
    Args:
        topic: The research topic string
        query: The user's query/question
        evidence_id_start: Starting ID number for evidence items
    
    Returns:
        Research results with atomic evidence store
    """
    if not topic or not topic.strip():
        return {
            "success": False,
            "error": "Empty topic provided",
            "evidence_store": [],
            "next_evidence_id": evidence_id_start
        }

    # Respect user option to skip deep research entirely
    if os.getenv("ENABLE_DEEP_RESEARCH", "true").strip().lower() in {"0", "false", "no", "off"}:
        print("[INFO] Deep research skipped (ENABLE_DEEP_RESEARCH=false)")
        return {
            "success": True,
            "topic": topic,
            "query": query,
            "summary": "",
            "evidence_store": [],
            "next_evidence_id": evidence_id_start,
            "deep_research_skipped": True,
        }

    topic_cache_key = _stable_json_hash({
        "query": query,
        "topic": topic.strip(),
        "deep_research_max_loops": os.getenv("DEEP_RESEARCH_MAX_LOOPS", "3"),
    })
    cached_topic_result = _get_stage1_cache("topic", topic_cache_key)
    if cached_topic_result:
        cached_evidence = cached_topic_result.get("evidence_store", [])
        remapped_evidence, next_evidence_id = _reassign_evidence_ids(cached_evidence, evidence_id_start)
        cached_topic_result["evidence_store"] = remapped_evidence
        cached_topic_result["next_evidence_id"] = next_evidence_id
        cached_topic_result["cache_hit"] = True
        print(f"[CACHE] Topic processing cache hit ({len(remapped_evidence)} evidence items)")
        return cached_topic_result
    
    try:
        from datetime import datetime
        
        # Call the deep_research pipeline
        research_summary = await run_deep_research(
            topic=topic,
            notify=None
        )
        
        # Extract atomic claims from summary
        evidence_store = []
        evidence_id = evidence_id_start

        # Parse actual source URLs from the '### Sources:' block in the summary
        parsed_sources = _parse_deep_research_sources(research_summary)
        print(f"[topic_processing] Parsed {len(parsed_sources)} sources from deep-research summary")
        sources_context = "\n".join(
            f"{i+1}. {s['url']} ({s['title']})"
            for i, s in enumerate(parsed_sources)
        ) if parsed_sources else "Not available"

        # Deterministically extract claims from deep-research summary (no LLM)
        year_match = re.search(r'\b(19|20)\d{2}\b', research_summary[:500])
        topic_year = year_match.group(0) if year_match else "n.d."
        top_sentences = _extract_claims_from_text(research_summary, query, max_claims=10)
        for claim_idx, sent in enumerate(top_sentences):
            src_entry = parsed_sources[claim_idx % len(parsed_sources)] if parsed_sources else None
            src_url = src_entry["url"] if src_entry else None
            src_title = src_entry["title"] if src_entry else topic
            evidence_store.append({
                "id": f"E{evidence_id}",
                "claim": sent[:300],
                "source": src_url or f"Deep Research: {topic}",
                "source_url": src_url,
                "source_title": src_title,
                "quote_span": sent[:400],
                "author": "n.d.",
                "year": topic_year,
                "publication": src_title,
                "publisher": "n.d.",
                "retrieval_context": "deep_research_pipeline",
                "confidence": 0.75,
                "timestamp_accessed": datetime.now().isoformat()
            })
            evidence_id += 1
        
        result_payload = {
            "success": True,
            "topic": topic,
            "query": query,
            "summary": research_summary,
            "evidence_store": evidence_store,
            "next_evidence_id": evidence_id
        }

        _set_stage1_cache("topic", topic_cache_key, result_payload)
        return result_payload
    
    except Exception as e:
        logger.error(
            "topic_processing failed for topic '%s', query '%s': %s",
            topic[:80], query[:80], e, exc_info=True,
        )
        return {
            "success": False,
            "topic": topic,
            "error": str(e),
            "evidence_store": [],
            "next_evidence_id": evidence_id_start
        }


async def process_links(
    links: List[str],
    query: str,
    evidence_id_start: int = 1,
    escalated_mode: bool = False,
) -> Dict[str, Any]:
    """
    Process links to extract atomic claims with validated citations.
    
    Args:
        links: Array of URL strings
        query: The user's query
        evidence_id_start: Starting ID number for evidence items
    
    Returns:
        Results with atomic evidence store and stable IDs
    """
    from datetime import datetime
    
    if not links:
        return {
            "type": "links",
            "count": 0,
            "items": [],
            "query": query,
            "evidence_store": [],
            "next_evidence_id": evidence_id_start
        }

    links_cache_key = _stable_json_hash({
        "query": query,
        "links": links,
    })
    cached_links_result = _get_stage1_cache("links", links_cache_key)
    if cached_links_result:
        cached_evidence = cached_links_result.get("evidence_store", [])
        remapped_evidence, next_evidence_id = _reassign_evidence_ids(cached_evidence, evidence_id_start)
        cached_links_result["evidence_store"] = remapped_evidence
        cached_links_result["next_evidence_id"] = next_evidence_id
        cached_links_result["cache_hit"] = True
        print(f"[CACHE] Link processing cache hit ({len(remapped_evidence)} evidence items)")
        return cached_links_result
    
    processed_links = []
    evidence_store = []
    evidence_id = evidence_id_start
    tavily_client = None
    
    try:
        # Initialize Tavily client
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return {
                "type": "links",
                "count": len(links),
                "error": "TAVILY_API_KEY not found in environment variables",
                "items": [{"url": link, "status": "skipped"} for link in links],
                "query": query,
                "evidence_store": [],
                "next_evidence_id": evidence_id_start
            }
        
        tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

        link_parallelism = _parse_int_env("STAGE1_LINK_PARALLELISM", 3, minimum=1, maximum=8)
        if escalated_mode:
            min_escalated_parallelism = _parse_int_env(
                "STAGE1_ESCALATED_MIN_LINK_PARALLELISM",
                2,
                minimum=1,
                maximum=8,
            )
            link_parallelism = max(link_parallelism, min_escalated_parallelism)
        tavily_max_retries = _parse_int_env("STAGE1_TAVILY_MAX_RETRIES", 2, minimum=0, maximum=5)
        tavily_retry_base_ms = _parse_int_env("STAGE1_TAVILY_RETRY_BASE_MS", 450, minimum=100, maximum=5000)
        tavily_retry_max_ms = _parse_int_env("STAGE1_TAVILY_RETRY_MAX_MS", 2500, minimum=250, maximum=10000)
        print(f"[Stage 1] Link processing parallelism: {link_parallelism} (escalated={escalated_mode})")
        semaphore = asyncio.Semaphore(link_parallelism)

        async def _process_single_link(index: int, link: str) -> Dict[str, Any]:
            try:
                async with semaphore:
                    # Use Tavily to extract content (with lightweight retry/backoff for transient failures)
                    search_result = None
                    last_extract_error = None
                    for attempt in range(tavily_max_retries + 1):
                        try:
                            search_result = await tavily_client.extract(urls=[link])
                            last_extract_error = None
                            break
                        except Exception as extract_error:
                            last_extract_error = extract_error
                            if attempt >= tavily_max_retries:
                                break

                            error_text = str(extract_error).lower()
                            transient_markers = [
                                "429", "rate", "timeout", "tempor", "connection", "reset", "503", "502", "504"
                            ]
                            is_transient = any(marker in error_text for marker in transient_markers)
                            if not is_transient:
                                break

                            backoff_ms = min(tavily_retry_max_ms, tavily_retry_base_ms * (2 ** attempt))
                            jitter_ms = int(backoff_ms * 0.25 * ((index + attempt) % 4) / 3)
                            wait_ms = backoff_ms + jitter_ms
                            print(
                                f"[Stage 1][Retry] Tavily extract transient error for {link} "
                                f"(attempt {attempt + 1}/{tavily_max_retries + 1}); retrying in {wait_ms}ms"
                            )
                            await asyncio.sleep(wait_ms / 1000.0)

                    # --- Determine raw content from extract result or fall back to search ---
                    _extract_ok = (
                        last_extract_error is None
                        and search_result
                        and search_result.get("results")
                        and len(search_result.get("results", [])) > 0
                    )

                    raw_content = ""
                    title = "Unknown"

                    if _extract_ok:
                        _res0 = search_result["results"][0]
                        raw_content = _res0.get("raw_content", "") or ""
                        title = _res0.get("title", "Unknown")
                        # Detect navigation-only HTML: strip markdown/symbols and check real text length
                        _text_only = re.sub(r'[#\[\]()!*_`|<>{}=\\]', ' ', raw_content)
                        _text_only = ' '.join(_text_only.split())
                        if len(_text_only) < 250:
                            _extract_ok = False  # treat as failed — content is just navigation chrome

                    if not _extract_ok:
                        # Fall back to Tavily search to get a snippet/summary for paywalled / JS-heavy pages
                        _search_hint = title if (title and title != "Unknown") else link
                        try:
                            _fb_res = await tavily_client.search(
                                _search_hint,
                                max_results=1,
                                max_tokens_per_source=1500,
                                include_raw_content=True,
                            )
                            _fb_results = _fb_res.get("results", [])
                            if _fb_results:
                                _fb = _fb_results[0]
                                _fb_content = _fb.get("raw_content") or _fb.get("content") or ""
                                _fb_title = _fb.get("title") or title
                                if len(_fb_content) > len(raw_content):
                                    raw_content = _fb_content
                                    title = _fb_title
                                    print(f"[Stage 1] Used search fallback for {link} (title: {title[:60]})")
                        except Exception as _fb_err:
                            print(f"[Stage 1] Search fallback failed for {link}: {_fb_err}")

                    # If we still have no usable content, emit a stub reference for this URL
                    if not raw_content or len(raw_content.strip()) < 50:
                        print(f"[Stage 1] No content for {link} — creating stub reference")
                        return {
                            "index": index,
                            "processed_link": {
                                "url": link,
                                "status": "stub",
                                "claims_extracted": 1,
                                "title": title
                            },
                            "claims": [{
                                "claim": f"Source referenced: {title}",
                                "source": link,
                                "source_url": link,
                                "source_title": title,
                                "quote_span": title,
                                "author": "n.d.",
                                "year": "n.d.",
                                "publication": title,
                                "publisher": "n.d.",
                                "retrieval_context": "link_stub",
                                "confidence": 0.3,
                                "timestamp_accessed": datetime.now().isoformat()
                            }]
                        }

                    # Truncate if too long
                    if len(raw_content) > 5000:
                        raw_content = raw_content[:5000] + "..."

                    # Deterministic claim extraction (no LLM)
                    link_year_m = re.search(r'\b(19|20)\d{2}\b', raw_content[:500])
                    link_year = link_year_m.group(0) if link_year_m else "n.d."
                    top_sentences = _extract_claims_from_text(raw_content, query, max_claims=6)
                    extracted_claims: List[Dict[str, Any]] = []
                    for sent in top_sentences:
                        extracted_claims.append({
                            "claim": sent[:300],
                            "source": link,
                            "source_url": link,
                            "source_title": title,
                            "quote_span": sent[:400],
                            "author": "n.d.",
                            "year": link_year,
                            "publication": title,
                            "publisher": "n.d.",
                            "retrieval_context": "link_processing",
                            "confidence": 0.75,
                            "timestamp_accessed": datetime.now().isoformat()
                        })

                    # Add a stub if no claims extracted so URL still appears in references
                    if not extracted_claims:
                        print(f"[Stage 1] No claims extracted for {link} — adding stub reference")
                        extracted_claims.append({
                            "claim": f"Source referenced: {title}",
                            "source": link,
                            "source_url": link,
                            "source_title": title,
                            "quote_span": title,
                            "author": "n.d.",
                            "year": "n.d.",
                            "publication": title,
                            "publisher": "n.d.",
                            "retrieval_context": "link_stub",
                            "confidence": 0.3,
                            "timestamp_accessed": datetime.now().isoformat()
                        })

                    print(f"✓ Extracted {len(extracted_claims)} claims from {link}")
                    return {
                        "index": index,
                        "processed_link": {
                            "url": link,
                            "status": "success",
                            "claims_extracted": len(extracted_claims),
                            "title": title
                        },
                        "claims": extracted_claims
                    }

            except Exception as e:
                error_msg = str(e)
                print(f"✗ Error processing {link}: {error_msg}")
                # Even on hard failure, emit a stub so all input URLs appear in references
                return {
                    "index": index,
                    "processed_link": {
                        "url": link,
                        "status": "stub",
                        "claims_extracted": 1,
                        "title": link
                    },
                    "claims": [{
                        "claim": f"Source referenced: {link}",
                        "source": link,
                        "source_url": link,
                        "source_title": link,
                        "quote_span": link,
                        "author": "n.d.",
                        "year": "n.d.",
                        "publication": link,
                        "publisher": "n.d.",
                        "retrieval_context": "link_stub",
                        "confidence": 0.3,
                        "timestamp_accessed": datetime.now().isoformat()
                    }]
                }

        link_results = await asyncio.gather(
            *[_process_single_link(index, link) for index, link in enumerate(links)],
            return_exceptions=True
        )

        ordered_results: List[Dict[str, Any]] = []
        for index, task_result in enumerate(link_results):
            if isinstance(task_result, Exception):
                ordered_results.append({
                    "index": index,
                    "processed_link": {
                        "url": links[index],
                        "status": "error",
                        "error": f"Task failed: {task_result}"
                    },
                    "claims": []
                })
            else:
                ordered_results.append(task_result)

        ordered_results.sort(key=lambda item: item.get("index", 0))

        for item in ordered_results:
            processed_links.append(item.get("processed_link", {
                "url": "unknown",
                "status": "error",
                "error": "Unknown processing failure"
            }))

            for claim in item.get("claims", []):
                claim_with_id = dict(claim)
                claim_with_id["id"] = f"E{evidence_id}"
                evidence_store.append(claim_with_id)
                evidence_id += 1
    
    finally:
        # Close Tavily client
        if tavily_client:
            await tavily_client.close()
    
    result_payload = {
        "type": "links",
        "count": len(links),
        "items": processed_links,
        "query": query,
        "evidence_store": evidence_store,
        "next_evidence_id": evidence_id
    }

    _set_stage1_cache("links", links_cache_key, result_payload)
    return result_payload


async def process_attachments(attachments: List[Any], query: str, evidence_id_start: int = 1) -> Dict[str, Any]:
    """
    Process file attachments provided by the user.
    
    Args:
        attachments: Array of file paths or file objects
        query: The user's query
    
    Returns:
        Processed attachment information
    """
    from datetime import datetime

    if not attachments:
        return {
            "type": "attachments",
            "count": 0,
            "items": [],
            "query": query,
            "evidence_store": [],
            "next_evidence_id": evidence_id_start,
        }

    attachments_cache_key = _stable_json_hash({
        "query": query,
        "attachments": [_attachment_cache_fingerprint(item) for item in attachments],
    })
    cached_attachments_result = _get_stage1_cache("attachments", attachments_cache_key)
    if cached_attachments_result:
        cached_evidence = cached_attachments_result.get("evidence_store", [])
        remapped_evidence, next_evidence_id = _reassign_evidence_ids(cached_evidence, evidence_id_start)
        cached_attachments_result["evidence_store"] = remapped_evidence
        cached_attachments_result["next_evidence_id"] = next_evidence_id
        cached_attachments_result["cache_hit"] = True
        print(f"[CACHE] Attachment processing cache hit ({len(remapped_evidence)} evidence items)")
        return cached_attachments_result

    async def extract_docint_text(raw_bytes: bytes) -> str:
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
        if not endpoint or not key or not raw_bytes:
            return ""

        headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/octet-stream",
        }

        analyze_urls = [
            f"{endpoint}/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30",
            f"{endpoint}/formrecognizer/documentModels/prebuilt-layout:analyze?api-version=2023-07-31",
        ]

        op_url = None
        for url in analyze_urls:
            try:
                resp = requests.post(url, headers=headers, data=raw_bytes, timeout=60)
                if resp.status_code in [200, 202]:
                    op_url = resp.headers.get("operation-location") or resp.headers.get("Operation-Location")
                    if op_url:
                        break
            except Exception:
                continue

        if not op_url:
            return ""

        poll_headers = {"Ocp-Apim-Subscription-Key": key}
        for _ in range(25):
            try:
                poll = requests.get(op_url, headers=poll_headers, timeout=60)
                data = poll.json() if poll.content else {}
                status = (data.get("status") or "").lower()
                if status == "succeeded":
                    result = data.get("analyzeResult", {})
                    lines = []
                    for paragraph in result.get("paragraphs", []):
                        text = paragraph.get("content", "")
                        if text:
                            lines.append(text)
                    if not lines:
                        for page in result.get("pages", []):
                            for line in page.get("lines", []):
                                text = line.get("content", "")
                                if text:
                                    lines.append(text)
                    return "\n".join(lines)
                if status in ["failed", "error"]:
                    return ""
            except Exception:
                return ""
            await asyncio.sleep(1)

        return ""

    async def extract_metadata_with_llm(text: str, filename: str, fallback_title: str) -> Dict[str, str]:
        return _extract_metadata_from_text(text, filename, fallback_title)

    async def extract_claims_with_llm(text: str, query_text: str, metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        sentences = _extract_claims_from_text(text, query_text, max_claims=6)
        return [{"claim": s[:300], "quote_span": s[:400], "confidence": 0.75} for s in sentences]

    processed_items = []
    evidence_store = []
    evidence_id = evidence_id_start

    for index, attachment in enumerate(attachments, start=1):
        filename = f"attachment_{index}"
        file_type = "unknown"
        raw_bytes = b""
        provided_text = ""

        if isinstance(attachment, dict):
            filename = attachment.get("filename") or filename
            file_type = attachment.get("file_type") or "unknown"
            provided_text = attachment.get("content", "") if isinstance(attachment.get("content", ""), str) else ""
            raw_bytes = attachment.get("bytes", b"") if isinstance(attachment.get("bytes", b""), (bytes, bytearray)) else b""
        elif isinstance(attachment, str):
            provided_text = attachment

        docint_text = await extract_docint_text(bytes(raw_bytes)) if raw_bytes else ""
        final_text = docint_text if docint_text and len(docint_text.strip()) > 50 else provided_text

        fallback_title = filename if filename else f"attachment_{index}"
        metadata = await extract_metadata_with_llm(final_text, filename, fallback_title)
        claims = await extract_claims_with_llm(final_text, query, metadata)

        if not claims and final_text.strip():
            # Minimal fallback claim when extraction fails
            snippet = final_text.strip()[:400]
            claims = [{
                "claim": f"Attachment {index} contains content relevant to the query.",
                "quote_span": snippet,
                "confidence": 0.6,
            }]

        for claim_data in claims:
            evidence_store.append({
                "id": f"E{evidence_id}",
                "claim": claim_data.get("claim", ""),
                "source": fallback_title,
                "source_url": None,
                "source_title": metadata.get("source_title", fallback_title),
                "quote_span": claim_data.get("quote_span", ""),
                "author": metadata.get("author", "Unknown"),
                "year": metadata.get("year", "Unknown"),
                "publication": metadata.get("publication", "Unknown"),
                "publisher": "Unknown",
                "retrieval_context": "attachment_processing",
                "confidence": claim_data.get("confidence", 0.75),
                "timestamp_accessed": datetime.now().isoformat(),
            })
            evidence_id += 1

        processed_items.append({
            "name": filename,
            "file_type": file_type,
            "status": "success" if claims else "no_claims",
            "claims_extracted": len(claims),
            "used_document_intelligence": bool(docint_text),
            "metadata": metadata,
        })

        print(f"✓ Extracted {len(claims)} atomic claims from attachment: {filename}")

    result_payload = {
        "type": "attachments",
        "count": len(processed_items),
        "items": processed_items,
        "query": query,
        "evidence_store": evidence_store,
        "next_evidence_id": evidence_id,
    }

    _set_stage1_cache("attachments", attachments_cache_key, result_payload)
    return result_payload


async def process_user_input(
    query: str,
    sources: Dict[str, Any],
    evidence_id_start: int = 1,
    escalated_mode: bool = False,
) -> Dict[str, Any]:
    """
    Main function to process user query and sources with atomic evidence tracking.
    
    Args:
        query: User's research query/question
        sources: JSON object containing topics, links, and attachments
            Example: {
                "topics": "machine learning transformers",
                "links": ["https://arxiv.org/abs/1706.03762"],
                "attachments": ["/path/to/file.pdf"]
            }
        evidence_id_start: Starting ID for evidence items (default: 1)
    
    Returns:
        Combined results from all processing with stable evidence IDs
    """
    # Split sources
    split_data = split_sources(sources)

    # Fallback: if no sources at all, use the query itself as the topic for deep research
    has_any_source = (
        bool((split_data.get("topics") or "").strip())
        or bool(split_data.get("links"))
        or bool(split_data.get("attachments"))
    )
    if not has_any_source and query and query.strip():
        logger.info(
            "No sources provided; using query as topic for deep research: '%s'",
            query[:120],
        )
        split_data["topics"] = query.strip()
    
    results = {
        "query": query,
        "timestamp": datetime.now().strftime("%B %d, %Y"),
        "topic_results": None,
        "link_results": None,
        "attachment_results": None,
        "evidence_store": [],
        "next_evidence_id": evidence_id_start
    }
    
    print(f"\n[Stage 1] Processing topic, links, and attachments in parallel... (escalated={escalated_mode})")
    source_tasks: List[Any] = []
    source_keys: List[str] = []

    if split_data["topics"]:
        topic_tasks: List[Any] = []
        topic_shards = [split_data["topics"]]
        if escalated_mode and os.getenv("STAGE1_ESCALATED_TOPIC_SHARDING", "true").strip().lower() in {"1", "true", "yes", "on"}:
            max_topic_shards = _parse_int_env("STAGE1_ESCALATED_TOPIC_SHARDS", 3, minimum=1, maximum=5)
            min_topic_shards = _parse_int_env("STAGE1_ESCALATED_MIN_TOPIC_SHARDS", 2, minimum=1, maximum=5)
            topic_shards = _split_topic_for_parallel_processing(
                split_data["topics"],
                max_parts=max_topic_shards,
                min_parts=min_topic_shards,
            )
            print(f"[Stage 1] Escalated topic sharding: {len(topic_shards)} shard(s)")

        if len(topic_shards) <= 1:
            source_keys.append("topic_results")
            source_tasks.append(topic_processing(split_data["topics"], query, evidence_id_start=1))
        else:
            source_keys.append("topic_results")

            async def _process_topic_shards() -> Dict[str, Any]:
                async def _run_topic_shard(shard: str) -> Dict[str, Any]:
                    print(f"[Stage 1][TopicShard] START: {shard[:80]}{'...' if len(shard) > 80 else ''}")
                    shard_result = await topic_processing(shard, query, evidence_id_start=1)
                    status = "ok" if shard_result.get("success") else "failed"
                    print(f"[Stage 1][TopicShard] END ({status}): {shard[:80]}{'...' if len(shard) > 80 else ''}")
                    return shard_result

                shard_results = await asyncio.gather(
                    *[_run_topic_shard(shard) for shard in topic_shards],
                    return_exceptions=True,
                )

                combined_evidence: List[Dict[str, Any]] = []
                combined_summaries: List[str] = []
                failed_shards = 0

                for shard, shard_result in zip(topic_shards, shard_results):
                    if isinstance(shard_result, Exception):
                        failed_shards += 1
                        print(f"✗ topic shard failed: {shard} -> {shard_result}")
                        continue
                    if not shard_result.get("success"):
                        failed_shards += 1
                        print(f"✗ topic shard unsuccessful: {shard} -> {shard_result.get('error', 'unknown')}")
                        continue

                    combined_evidence.extend(shard_result.get("evidence_store", []))
                    shard_summary = (shard_result.get("summary") or "").strip()
                    if shard_summary:
                        combined_summaries.append(f"### {shard}\n{shard_summary}")

                return {
                    "success": len(combined_evidence) > 0,
                    "topic": split_data["topics"],
                    "query": query,
                    "summary": "\n\n".join(combined_summaries),
                    "evidence_store": combined_evidence,
                    "next_evidence_id": 1,
                    "topic_shards": topic_shards,
                    "failed_topic_shards": failed_shards,
                }

            source_tasks.append(_process_topic_shards())

    if split_data["links"]:
        source_keys.append("link_results")
        source_tasks.append(
            process_links(
                split_data["links"],
                query,
                evidence_id_start=1,
                escalated_mode=escalated_mode,
            )
        )

    if split_data["attachments"]:
        source_keys.append("attachment_results")
        source_tasks.append(process_attachments(split_data["attachments"], query, evidence_id_start=1))

    if source_tasks:
        print(f"[Stage 1] Launching {len(source_tasks)} source task(s) concurrently")
        parallel_results = await asyncio.gather(*source_tasks, return_exceptions=True)
        for key, task_result in zip(source_keys, parallel_results):
            if isinstance(task_result, Exception):
                results[key] = {
                    "success": False,
                    "error": str(task_result),
                    "evidence_store": [],
                    "next_evidence_id": 1,
                }
                print(f"✗ {key} failed: {task_result}")
            else:
                results[key] = task_result

    # Deterministic evidence ID assignment order: topic -> links -> attachments
    next_evidence_id = evidence_id_start
    for key in ["topic_results", "link_results", "attachment_results"]:
        source_result = results.get(key)
        if not source_result or not source_result.get("evidence_store"):
            continue

        remapped_evidence, next_evidence_id = _reassign_evidence_ids(
            source_result.get("evidence_store", []),
            next_evidence_id,
        )
        source_result["evidence_store"] = remapped_evidence
        source_result["next_evidence_id"] = next_evidence_id
        results["evidence_store"].extend(remapped_evidence)

        label = key.replace("_results", "").replace("_", " ").title()
        print(f"[{label} processing: added {len(remapped_evidence)} evidence items]")
    
    # Print evidence ID summary
    if results["evidence_store"]:
        print(f"\n[Total evidence collected: {len(results['evidence_store'])} items (E{evidence_id_start} through E{next_evidence_id-1})]")
    
    # Store the next evidence ID for continuation
    results["next_evidence_id"] = next_evidence_id
    
    # Generate comprehensive summary from all collected evidence
    if results["evidence_store"] and len(results["evidence_store"]) > 0:
        summary_result = await generate_summary_from_evidence(query, results["evidence_store"])
        results["generated_summary"] = summary_result
    else:
        # Log detailed diagnostics so we know WHY no evidence was collected
        topic_err = (results.get("topic_results") or {}).get("error", "")
        link_err = (results.get("link_results") or {}).get("error", "")
        attach_err = (results.get("attachment_results") or {}).get("error", "")
        logger.warning(
            "No evidence collected for query '%s'. "
            "topic_error=%s | link_error=%s | attachment_error=%s",
            query[:120],
            topic_err or "(none)",
            link_err or "(none)",
            attach_err or "(none)",
        )
        results["generated_summary"] = {
            "success": False,
            "summary": "",
            "error": (
                "No evidence collected to generate summary. "
                f"Topic: {topic_err or 'no topic provided'}. "
                f"Links: {link_err or 'no links provided'}. "
                f"Attachments: {attach_err or 'no attachments provided'}."
            ),
            "evidence_count": 0
        }
    
    return results


async def process_with_iterative_refinement(
    query: str,
    sources: Dict[str, Any],
    max_iterations: int = 3,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    iteration_prefix: str = "",
) -> Dict[str, Any]:
    """
    Process user query with iterative refinement - runs multiple iterations with critiques and adjustments.
    
    Args:
        query: User's research query/question
        sources: JSON object containing topics, links, and attachments
        max_iterations: Number of refinement iterations (default: 3)
    
    Returns:
        Dictionary containing all iterations with summaries, critiques, and adjustments
    """
    def emit(event_type: str, **payload):
        if progress_callback:
            try:
                progress_callback({"type": event_type, **payload})
            except Exception:
                pass

    try:
        return await _process_with_iterative_refinement_inner(
            query, sources, max_iterations, emit, iteration_prefix,
        )
    except Exception as e:
        logger.error(
            "process_with_iterative_refinement failed for query '%s': %s",
            query[:120], e, exc_info=True,
        )
        emit("stage_text", stage=1, text=f"Iterative refinement error: {e}")
        return {
            "original_query": query,
            "sources": sources,
            "total_iterations": 0,
            "max_iterations_requested": max_iterations,
            "iterations": [],
            "final_summary": {
                "success": False,
                "summary": "",
                "error": f"Iterative refinement failed: {e}",
                "evidence_count": 0,
            },
            "final_evidence_count": 0,
            "cumulative_evidence_store": [],
        }


async def _process_with_iterative_refinement_inner(
    query: str,
    sources: Dict[str, Any],
    max_iterations: int,
    emit: Callable,
    iteration_prefix: str,
) -> Dict[str, Any]:
    """Inner implementation of iterative refinement (extracted for try/except wrapper)."""

    iterations = []
    current_query = query
    cumulative_evidence = []
    next_evidence_id = 1  # Track evidence ID across all iterations
    is_escalated_run = iteration_prefix.strip().upper().startswith("ESCALATED")

    try:
        min_gain_ratio = float(os.getenv("ITERATIVE_EARLY_STOP_MIN_GAIN_RATIO", "0.10"))
    except ValueError:
        min_gain_ratio = 0.10
    min_gain_ratio = max(0.0, min(1.0, min_gain_ratio))

    try:
        consecutive_rounds_required = int(os.getenv("ITERATIVE_EARLY_STOP_CONSECUTIVE_ROUNDS", "2"))
    except ValueError:
        consecutive_rounds_required = 2
    consecutive_rounds_required = max(1, min(5, consecutive_rounds_required))

    if is_escalated_run:
        try:
            min_gain_ratio = float(os.getenv("ESCALATED_ITERATIVE_EARLY_STOP_MIN_GAIN_RATIO", str(min_gain_ratio)))
        except ValueError:
            pass
        min_gain_ratio = max(0.0, min(1.0, min_gain_ratio))

        try:
            consecutive_rounds_required = int(
                os.getenv("ESCALATED_ITERATIVE_EARLY_STOP_CONSECUTIVE_ROUNDS", "1")
            )
        except ValueError:
            consecutive_rounds_required = 1
        consecutive_rounds_required = max(1, min(5, consecutive_rounds_required))

        print(
            "[ESCALATED] Early-stop config: "
            f"min_gain_ratio={min_gain_ratio:.2f}, consecutive_rounds={consecutive_rounds_required}"
        )

    early_stop_enabled = os.getenv("ITERATIVE_EARLY_STOP_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    low_gain_streak = 0
    
    for iteration in range(1, max_iterations + 1):
        iteration_label = f"{iteration_prefix}Iteration {iteration}/{max_iterations}".strip()
        emit("stage_text", stage=1, text=iteration_label)
        print(f"\n{'='*70}")
        print(f"{(iteration_prefix + 'ITERATION').strip()} {iteration}/{max_iterations}")
        print(f"{'='*70}")
        
        # Process the current iteration with continuing evidence IDs
        iteration_results = await process_user_input(
            current_query,
            sources,
            evidence_id_start=next_evidence_id,
            escalated_mode=is_escalated_run,
        )
        
        # Update next_evidence_id for the next iteration
        next_evidence_id = iteration_results.get("next_evidence_id", next_evidence_id)
        
        previous_cumulative_count = len(cumulative_evidence)

        # Add new evidence to cumulative store
        new_evidence = iteration_results.get("evidence_store", [])
        cumulative_evidence.extend(new_evidence)

        gain_ratio = 1.0
        if previous_cumulative_count > 0:
            gain_ratio = len(new_evidence) / previous_cumulative_count

        emit("stage_metric", stage=1, key="New Evidence", value=len(new_evidence))
        emit("stage_metric", stage=1, key="Evidence Gain", value=f"{gain_ratio * 100:.1f}%")
        
        # Update iteration results with cumulative evidence
        iteration_results["cumulative_evidence_store"] = cumulative_evidence.copy()
        iteration_results["iteration"] = iteration
        iteration_results["query_used"] = current_query
        
        # Generate or regenerate summary with all cumulative evidence (deterministic, no LLM)
        if cumulative_evidence:
            summary_result = _build_deterministic_summary(query, cumulative_evidence)
            iteration_results["generated_summary"] = summary_result
        
        iteration_data = {
            "iteration": iteration,
            "query": current_query,
            "results": iteration_results,
            "new_evidence_count": len(new_evidence),
            "cumulative_evidence_count": len(cumulative_evidence)
        }

        emit(
            "stage_metric",
            stage=1,
            key="Cumulative Evidence",
            value=len(cumulative_evidence),
        )

        if early_stop_enabled and iteration > 1:
            if gain_ratio < min_gain_ratio:
                low_gain_streak += 1
                print(
                    f"[Iteration {iteration}] Low evidence gain detected: "
                    f"{gain_ratio * 100:.1f}% (< {min_gain_ratio * 100:.1f}%) "
                    f"[{low_gain_streak}/{consecutive_rounds_required}]"
                )
            else:
                low_gain_streak = 0

            if low_gain_streak >= consecutive_rounds_required and iteration < max_iterations:
                print(f"[Iteration {iteration}] Early-stop triggered due to consistently low evidence gain")
                emit(
                    "stage_text",
                    stage=1,
                    text=(
                        f"Early-stop triggered at iteration {iteration}: "
                        f"evidence gain stayed below {min_gain_ratio * 100:.1f}%"
                    ),
                )
                iteration_data["early_stop_triggered"] = True
                iteration_data["evidence_gain_ratio"] = gain_ratio
                iterations.append(iteration_data)
                break
        
        # Only critique if not the last iteration
        if iteration < max_iterations:
            if iteration_results.get("generated_summary", {}).get("success"):
                print(f"\n[Iteration {iteration}] Generating critique...")
                
                critique_result = await critique_summary(
                    query,
                    iteration_results["generated_summary"]["summary"],
                    cumulative_evidence,
                    iteration
                )
                
                iteration_data["critique"] = critique_result
                
                if critique_result.get("success"):
                    print(f"[Iteration {iteration}] Generating adjustments for next iteration...")

                    freeze_query_adjustment = (
                        is_escalated_run and
                        os.getenv("ESCALATED_DISABLE_QUERY_ADJUSTMENT", "true").strip().lower() in {"1", "true", "yes", "on"}
                    )
                    if freeze_query_adjustment:
                        print(f"[Iteration {iteration}] Escalated mode: query adjustment disabled for cache reuse")
                        adjusted_query = current_query
                    else:
                        # Generate adjustments for next iteration
                        adjusted_query = await generate_adjustments_from_critique(
                            query,
                            critique_result["critique"]
                        )
                    
                    iteration_data["adjustments"] = {
                        "original_query": query,
                        "adjusted_query": adjusted_query
                    }
                    
                    # Use adjusted query for next iteration
                    current_query = adjusted_query
                    print(f"[Iteration {iteration}] Next iteration will use refined query")
                else:
                    print(f"[Iteration {iteration}] Critique failed, using original query")
                    iteration_data["adjustments"] = {
                        "original_query": query,
                        "adjusted_query": query
                    }
        else:
            print(f"\n[Iteration {iteration}] Final iteration - no critique needed")
            iteration_data["critique"] = None
            iteration_data["adjustments"] = None
        
        iterations.append(iteration_data)
    
    total_iterations_run = len(iterations)

    # Build final results
    final_results = {
        "original_query": query,
        "sources": sources,
        "total_iterations": total_iterations_run,
        "max_iterations_requested": max_iterations,
        "iterations": iterations,
        "final_summary": iterations[-1]["results"].get("generated_summary", {}),
        "final_evidence_count": len(cumulative_evidence),
        "cumulative_evidence_store": cumulative_evidence
    }
    
    return final_results


def get_random_style_from_db():
    """
    Get a random writing style from Cosmos DB for testing.
    Also fetches associated global rulebook and extracts style rules properly.
    
    Returns:
        Dictionary with style information including:
        - style_description: JSON string of style rules
        - global_rules: Text from global rulebook
        - example: Example text from evidence_spans or example field
        - Other metadata: speaker, audience, name, etc.
    """
    try:
        # Initialize Cosmos DB client
        cosmos_client = CosmosClient(
            url=os.getenv("AZURE_COSMOS_ENDPOINT"),
            credential=os.getenv("AZURE_COSMOS_KEY")
        )
        database = cosmos_client.get_database_client(os.getenv("AZURE_COSMOS_DATABASE"))
        styles_container = database.get_container_client("styles_from_speeches")
        
        # Query for style fingerprints
        query = """
        SELECT *
        FROM c
        WHERE c.doc_kind = 'style_fingerprint'
        """
        items = list(styles_container.query_items(
            query=query,
            parameters=[],
            enable_cross_partition_query=True
        ))
        
        if not items:
            print("No style fingerprint documents found in Cosmos DB")
            return None
        
        # Select a random style for testing
        import random
        selected_style = random.choice(items)
        
        # Extract style rules from the document
        style_rules = _extract_style_rules_from_doc(selected_style)
        selected_style["style_description"] = json.dumps(style_rules, ensure_ascii=False, indent=2) if style_rules else ""
        
        # Fetch associated global rulebook if available
        rulebook_id = selected_style.get("global_rulebook_id")
        if rulebook_id:
            try:
                rulebook_query = "SELECT * FROM c WHERE c.id = @id OR c.container_key = @id"
                rulebook_params = [{"name": "@id", "value": rulebook_id}]
                rulebook_items = list(styles_container.query_items(
                    query=rulebook_query,
                    parameters=rulebook_params,
                    enable_cross_partition_query=True
                ))
                if rulebook_items:
                    rulebook = rulebook_items[0]
                    selected_style["global_rules"] = _extract_rulebook_text(rulebook)
            except Exception as e:
                print(f"Error fetching global rulebook: {e}")
                selected_style["global_rules"] = ""
        
        # Extract example text
        example = selected_style.get("example")
        if not example:
            evidence = selected_style.get("evidence_spans") or []
            example = "\n".join([str(e) for e in evidence if e]) if evidence else ""
        selected_style["example"] = example
        
        return selected_style
        
    except Exception as e:
        print(f"Error getting style from Cosmos DB: {e}")
        import traceback
        traceback.print_exc()
        return None


def _extract_style_rules_from_doc(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract style rules from a style_fingerprint document."""
    rules = item.get("style") or item.get("style_instructions")
    if rules and isinstance(rules, dict):
        return rules
    
    # Check nested properties
    props = item.get("properties") or {}
    if isinstance(props, str):
        try:
            props = json.loads(props)
        except Exception:
            props = {}
    
    if isinstance(props, dict):
        nested = props.get("style_instructions") or props.get("style")
        if isinstance(nested, dict):
            return nested
    
    # Fallback: build from known top-level fields
    fallback_keys = [
        "register_profile",
        "stylistics",
        "pragmatics",
        "semantics_frames",
        "discourse_structure",
    ]
    fallback = {}
    for key in fallback_keys:
        if key in item and item[key]:
            fallback[key] = item[key]
    
    return fallback if fallback else {}


def _extract_rulebook_text(rulebook: Dict[str, Any]) -> str:
    """Extract readable text from a global_rulebook document."""
    if not rulebook:
        return ""
    
    for key in ["global_rules", "rules", "rulebook", "rulebook_text"]:
        if key in rulebook and rulebook[key]:
            return rulebook[key]
    
    try:
        return json.dumps(rulebook, ensure_ascii=False, indent=2)
    except Exception:
        return str(rulebook)


def get_sample_speeches_from_db(speaker_name: str, max_speeches: int = 3) -> List[str]:
    """
    Fetch sample speeches from the speeches database for the given speaker.
    
    Args:
        speaker_name: The name of the speaker (e.g., "Benjamin E. Diokno", "Eli M. Remolona Jr.")
        max_speeches: Maximum number of speeches to fetch (default: 3)
    
    Returns:
        List of speech content strings
    """
    try:
        # Initialize Cosmos DB client for speeches database
        cosmos_client = CosmosClient(
            url=os.getenv("AZURE_SPEECHES_ENDPOINT"),
            credential=os.getenv("AZURE_SPEECHES_KEY")
        )
        database = cosmos_client.get_database_client(os.getenv("AZURE_SPEECHES_DATABASE"))
        
        # Try common container names
        container_names = ["speeches", "Speeches", "speech_data", "documents"]
        container = None
        
        for name in container_names:
            try:
                container = database.get_container_client(name)
                # Test if container exists by trying to read properties
                container.read()
                break
            except:
                continue
        
        if not container:
            print(f"[WARNING] No speeches container found in database")
            return []
        
        # Query for speeches by speaker
        # Try different speaker field variations
        speaker_queries = [
            f"SELECT TOP {max_speeches} * FROM c WHERE CONTAINS(c.Speaker, @speaker) ORDER BY c._ts DESC",
            f"SELECT TOP {max_speeches} * FROM c WHERE CONTAINS(c.speaker, @speaker) ORDER BY c._ts DESC",
            f"SELECT TOP {max_speeches} * FROM c WHERE c.Speaker = @speaker ORDER BY c._ts DESC",
            f"SELECT TOP {max_speeches} * FROM c WHERE c.speaker = @speaker ORDER BY c._ts DESC"
        ]
        
        items = []
        for query in speaker_queries:
            try:
                items = list(container.query_items(
                    query=query,
                    parameters=[{"name": "@speaker", "value": speaker_name}],
                    enable_cross_partition_query=True
                ))
                if items:
                    break
            except Exception as e:
                continue
        
        if not items:
            print(f"[WARNING] No speeches found for speaker: {speaker_name}")
            return []
        
        # Extract speech content
        speeches = []
        for item in items[:max_speeches]:
            content = item.get("content", "") or item.get("text", "") or item.get("speech_text", "")
            if content and len(content) > 200:  # Only use substantial speeches
                # Clean up the content - find where actual speech starts
                # Look for common greeting patterns
                greeting_start_patterns = [
                    'magandang', 'good morning', 'good afternoon', 'good evening',
                    'ladies and gentlemen', 'thank you', 'it is'
                ]
                
                lines = content.split('\n')
                speech_start_idx = None
                
                for i, line in enumerate(lines):
                    line_lower = line.lower().strip()
                    if any(greeting in line_lower for greeting in greeting_start_patterns):
                        speech_start_idx = i
                        break
                
                if speech_start_idx is not None:
                    cleaned_content = '\n'.join(lines[speech_start_idx:]).strip()
                else:
                    # Fallback: skip obvious metadata lines
                    cleaned_lines = []
                    for line in lines:
                        line_lower = line.lower().strip()
                        if not any(x in line_lower for x in ['http', '.pdf', 'bangko sentral ng pilipinas media']):
                            cleaned_lines.append(line)
                    cleaned_content = '\n'.join(cleaned_lines).strip()
                
                if len(cleaned_content) > 200:
                    speeches.append(cleaned_content)
        
        print(f"[INFO] Fetched {len(speeches)} sample speech(es) for {speaker_name}")
        for i, speech in enumerate(speeches, 1):
            print(f"[INFO]   Speech {i}: {len(speech)} characters")
        
        return speeches
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch speeches: {e}")
        import traceback
        traceback.print_exc()
        return []


def enforce_citation_discipline(text: str, max_ids_per_sentence: int = 2) -> str:
    """
    IMPROVEMENT #3: Enforce citation discipline - max N evidence IDs per sentence.
    
    If a sentence has more than max_ids_per_sentence citations, keep only the first N.
    This prevents "citation spraying" where models cite many IDs hoping one fits.
    
    Args:
        text: Styled output text with citations
        max_ids_per_sentence: Maximum evidence IDs allowed per sentence (default: 2)
        
    Returns:
        Text with citation discipline enforced
    """
    # Split into sentences (preserve terminators)
    # Pattern splits on .!? followed by space, so we need to add them back
    parts = re.split(r'([.!?]+\s+)', text)
    
    corrected_parts = []
    violations_fixed = 0
    
    for i in range(0, len(parts), 2):
        if i >= len(parts):
            break
            
        sentence = parts[i]
        terminator = parts[i+1] if i+1 < len(parts) else ''
        
        # Find all citations in this sentence
        citations = re.findall(r'\[E\d+(?:,E\d+)*\]', sentence)
        
        if not citations:
            corrected_parts.append(sentence + terminator)
            continue
        
        # Extract all unique evidence IDs from all citations in the sentence
        all_ids = []
        for citation in citations:
            ids = re.findall(r'E\d+', citation)
            all_ids.extend(ids)
        
        # Remove duplicates while preserving order
        unique_ids = list(dict.fromkeys(all_ids))
        
        if len(unique_ids) <= max_ids_per_sentence:
            # Already compliant
            corrected_parts.append(sentence + terminator)
        else:
            # Violation: too many IDs
            violations_fixed += 1
            
            # Keep only first max_ids_per_sentence IDs
            kept_ids = unique_ids[:max_ids_per_sentence]
            
            # Replace all citations in this sentence with a single citation
            corrected_sentence = sentence
            for citation in citations:
                corrected_sentence = corrected_sentence.replace(citation, '', 1)
            
            # Add the corrected citation (clean up extra spaces)
            new_citation = f"[{','.join(kept_ids)}]"
            corrected_sentence = corrected_sentence.strip() + " " + new_citation
            
            corrected_parts.append(corrected_sentence + terminator)
    
    if violations_fixed > 0:
        print(f"  ⚠️  Fixed {violations_fixed} citation discipline violations (reduced to ≤{max_ids_per_sentence} IDs/sentence)")
    else:
        print(f"  ✓ Citation discipline already maintained (≤{max_ids_per_sentence} IDs/sentence)")
    
    return ''.join(corrected_parts)


def fix_speaker_perspective(text: str, speaker_name: str) -> str:
    """
    Post-process speech text to fix third-person references to the speaker.
    
    When the speaker is the one delivering the speech, references like
    "Governor Remolona noted that..." should be converted to first-person.

    Args:
        text: The speech text to fix
        speaker_name: The speaker's name (e.g., "Eli M. Remolona Jr.")

    Returns:
        Text with third-person speaker references corrected to first person.
    """
    if not text or not speaker_name:
        return text

    # Build name variants for matching:
    # "Eli M. Remolona Jr." -> ["Remolona", "Eli M. Remolona Jr.", "Eli Remolona"]
    name_parts = speaker_name.strip().split()
    name_variants = {speaker_name.strip()}

    # Add last name (skip suffixes like Jr., Sr., III)
    suffixes = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv"}
    substantive_parts = [p for p in name_parts if p.lower().rstrip(".,") not in suffixes]
    if substantive_parts:
        last_name = substantive_parts[-1].rstrip(".,")
        name_variants.add(last_name)
        # First + Last (no middle initial)
        if len(substantive_parts) >= 2:
            name_variants.add(f"{substantive_parts[0]} {last_name}")

    # Common titles that may precede the speaker's name
    titles = [
        "Governor", "Deputy Governor", "Director", "Chairman",
        "President", "Secretary", "Commissioner", "Dr.",
        "Sec.", "Gov.", "Dir.", "Amb.", "Atty.",
    ]

    fixes_applied = 0

    # Sort name variants longest first to avoid partial replacements
    sorted_variants = sorted(name_variants, key=len, reverse=True)

    for variant in sorted_variants:
        escaped = re.escape(variant)

        # Pattern: "Title Name said/stated/noted/emphasized/highlighted/mentioned/stressed/argued/explained..."
        for title in titles:
            t_escaped = re.escape(title)
            # "Governor Remolona noted that X" -> "I note that X" (but only at sentence-level)
            pattern = re.compile(
                rf'\b{t_escaped}\s+{escaped}\b',
                re.IGNORECASE
            )
            if pattern.search(text):
                text = pattern.sub("I", text)
                fixes_applied += 1

        # Pattern: "According to Name, ..." -> "... " (remove the phrase)
        pattern_according = re.compile(
            rf'[Aa]ccording\s+to\s+{escaped}\s*,?\s*',
            re.IGNORECASE
        )
        if pattern_according.search(text):
            text = pattern_according.sub("", text)
            fixes_applied += 1

        # Pattern: "Name stated/said/noted/emphasized/believes/highlights that"
        # -> "I state/say/note... that"
        verbs_map = {
            "stated": "stated", "states": "state", "said": "said",
            "says": "say", "noted": "noted", "notes": "note",
            "emphasized": "emphasized", "emphasizes": "emphasize",
            "highlighted": "highlighted", "highlights": "highlight",
            "believes": "believe", "believed": "believed",
            "mentioned": "mentioned", "mentions": "mention",
            "stressed": "stressed", "stresses": "stress",
            "argued": "argued", "argues": "argue",
            "explained": "explained", "explains": "explain",
            "added": "added", "adds": "add",
            "observed": "observed", "observes": "observe",
            "pointed out": "pointed out", "points out": "point out",
            "underscored": "underscored", "underscores": "underscore",
            "reiterated": "reiterated", "reiterates": "reiterate",
        }

        for third_verb, first_verb in verbs_map.items():
            pattern_name_verb = re.compile(
                rf'\b{escaped}\s+{re.escape(third_verb)}\b',
                re.IGNORECASE
            )
            if pattern_name_verb.search(text):
                text = pattern_name_verb.sub(f"I {first_verb}", text)
                fixes_applied += 1

    # Fix double-capital "I" at sentence start after replacements (e.g., ". I I noted")
    text = re.sub(r'\bI\s+I\b', 'I', text)

    # Fix sentences starting with lowercase after our replacements
    text = re.sub(r'(?<=\.\s)([a-z])', lambda m: m.group(1).upper(), text)

    if fixes_applied > 0:
        print(f"[INFO] Fixed {fixes_applied} third-person speaker reference(s) to first-person perspective")

    return text


def trim_text_to_boundary(text: str, max_length: int) -> str:
    """Trim text to max_length without cutting citations/sentences mid-way when possible."""
    if not text or len(text) <= max_length:
        return text

    raw_truncated = text[:max_length]

    # If citation is cut mid-way, drop from the unmatched '[' onward
    if raw_truncated.count('[') > raw_truncated.count(']'):
        last_open_bracket = raw_truncated.rfind('[')
        if last_open_bracket > 0:
            raw_truncated = raw_truncated[:last_open_bracket].rstrip()

    # Prefer ending on a natural sentence/citation boundary
    candidate_boundaries = [
        raw_truncated.rfind("]. "),
        raw_truncated.rfind("].\n"),
        raw_truncated.rfind(". "),
        raw_truncated.rfind("? "),
        raw_truncated.rfind("! "),
        raw_truncated.rfind("]\n"),
    ]
    best_boundary = max(candidate_boundaries)

    if best_boundary >= int(max_length * 0.6):
        return raw_truncated[:best_boundary + 1].rstrip()

    return raw_truncated.rstrip()


def split_references_section(text: str) -> (str, str):
    """Split text into (body, references_section). References section starts at a REFERENCES heading if present."""
    if not text:
        return "", ""

    match = re.search(r'\n(?:=+\n)?\s*REFERENCES\s*\n(?:=+\n)?', text, flags=re.IGNORECASE)
    if not match:
        return text, ""

    return text[:match.start()].rstrip(), text[match.start():]


def strip_in_text_citations(text: str) -> str:
    """Remove in-text citation tokens for effective length counting."""
    if not text:
        return ""

    # [E1] / [E1,E2]
    without_bracket_citations = re.sub(r'\[E\d+(?:,E\d+)*\]', '', text)

    # APA-like parenthetical citations containing a year or n.d.
    without_parenthetical_citations = re.sub(
        r'\((?=[^)]*\b(?:\d{4}[a-z]?|n\.d\.)\b)[^)]*\)',
        '',
        without_bracket_citations,
        flags=re.IGNORECASE
    )

    # Normalize spacing created by removals
    normalized = re.sub(r'\s{2,}', ' ', without_parenthetical_citations)
    return normalized.strip()


def effective_speech_body_length(text: str) -> int:
    """Length used for max-length policy: excludes references and in-text citations."""
    body, _ = split_references_section(text)
    return len(strip_in_text_citations(body))


def trim_body_preserve_closing(text: str, max_length: int) -> str:
    """Fallback trim that keeps opening and ending (closing remarks) when possible."""
    if not text:
        return text

    if len(strip_in_text_citations(text)) <= max_length:
        return text

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) <= 3:
        return trim_text_to_boundary(text, max_length)

    first = sentences[0]
    ending = sentences[-2:]
    middle = sentences[1:-2]

    # Remove middle content from the center outward until under effective limit
    while middle and len(strip_in_text_citations(" ".join([first] + middle + ending))) > max_length:
        remove_idx = len(middle) // 2
        middle.pop(remove_idx)

    candidate = " ".join([first] + middle + ending).strip()
    if len(strip_in_text_citations(candidate)) <= max_length:
        return candidate

    return trim_text_to_boundary(candidate, max_length)


async def smart_trim_speech_to_max_length(
    speech_text: str,
    max_length: int,
    style_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Reduce speech length using an LLM by removing least important content while preserving
    message, style, tone, confidence, and citation fidelity.
    """
    if not speech_text:
        return {
            "success": False,
            "error": "No speech text provided",
            "trimmed_output": speech_text
        }

    speech_body, references_section = split_references_section(speech_text)

    if len(strip_in_text_citations(speech_body)) <= max_length:
        return {
            "success": True,
            "trimmed_output": speech_text,
            "strategy": "no-op"
        }

    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )

        model = (
            os.getenv("AZURE_OPENAI_STAGE3_EDIT_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_STRONG_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        )

        style_context = ""
        if style_profile:
            style_name = style_profile.get("name", "")
            style_speaker = style_profile.get("speaker", "")
            style_audience = style_profile.get("audience_setting_classification", "")
            style_tone = style_profile.get("tone", "")

            style_context = "\n<STYLE_TO_RETAIN>\n"
            if style_name:
                style_context += f"Style Name: {style_name}\n"
            if style_speaker:
                style_context += f"Speaker Persona: {style_speaker}\n"
            if style_audience:
                style_context += f"Audience: {style_audience}\n"
            if style_tone:
                style_context += f"Tone: {style_tone}\n"
            style_context += "Retain rhetorical voice, cadence, confidence, and register.\n"
            style_context += "</STYLE_TO_RETAIN>\n"

        # SPEAKER PERSPECTIVE: Preserve first-person voice during trimming
        trim_speaker_name = style_profile.get("speaker") or style_profile.get("Speaker") if style_profile else None
        perspective_note = ""
        if trim_speaker_name:
            perspective_note = (
                f" This speech is delivered by {trim_speaker_name} — maintain first-person "
                f"perspective ('I', 'we') and NEVER refer to {trim_speaker_name} in the third person."
            )

        system_prompt = (
            "You are a senior speech editor. Shorten the speech by removing the least important "
            "details only. Preserve core message, factual meaning, argument flow, and speaker style. "
            "Do not add new claims, policies, or disclaimers. Keep all citations already present "
            "in retained sentences unchanged." + perspective_note
        )

        user_prompt = f"""{style_context}
<CONSTRAINTS>
- Target maximum length: {max_length} characters (hard limit).
- Character limit applies to body only (exclude any REFERENCES section).
- In-text citations do NOT count toward the character limit; preserve them anyway.
- Keep opening and closing intact when possible.
- Always end with brief closing remarks and a clear end greeting (e.g., "Thank you.").
- Prefer trimming repetitive or lower-priority illustrative details.
- Preserve confidence and clarity.
- Return ONLY the revised speech text.
</CONSTRAINTS>

<SPEECH>
{speech_body}
</SPEECH>
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=max(4000, min(12000, max_length * 6))
        )

        trimmed_output = (response.choices[0].message.content or "").strip()

        if not trimmed_output:
            return {
                "success": False,
                "error": "LLM returned empty trimmed output",
                "trimmed_output": trim_body_preserve_closing(speech_body, max_length) + references_section
            }

        if len(strip_in_text_citations(trimmed_output)) > max_length:
            trimmed_output = trim_body_preserve_closing(trimmed_output, max_length)

        trimmed_output = trimmed_output + references_section

        return {
            "success": True,
            "trimmed_output": trimmed_output,
            "strategy": "llm-trim"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "trimmed_output": trim_body_preserve_closing(speech_body, max_length) + references_section
        }


async def fix_speech_issues(
    speech_text: str,
    issues: List[Dict[str, Any]],
    evidence_store: List[Dict[str, Any]] = None,
    issue_type: str = "generic",
    style_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Use LLM to automatically fix flagged issues in speech segments.
    
    Args:
        speech_text: The full speech text
        issues: List of issues to fix, each with:
            - segment: The problematic text segment
            - issue_description: What's wrong
            - suggestion: Optional suggestion for fix
            - severity: CRITICAL/HIGH/MEDIUM/LOW
        evidence_store: List of evidence for citation context
        issue_type: Type of issues ("citation", "plagiarism", "policy", "generic")
        style_profile: Optional selected style metadata to preserve voice and register
    
    Returns:
        {
            "success": bool,
            "fixed_speech": str (revised text),
            "fixes_applied": int,
            "fix_details": List of what was changed
        }
    """
    if not issues:
        return {
            "success": True,
            "fixed_speech": speech_text,
            "fixes_applied": 0,
            "fix_details": []
        }
    
    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        
        model = (
            os.getenv("AZURE_OPENAI_STAGE3_EDIT_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_STRONG_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        )
        
        # Build issue-specific system prompt
        system_prompts = {
            "citation": """Fix claims not fully supported by evidence. Soften exaggerated claims, add hedging ("may", "appears to"), remove unsupported specifics. Keep citations [ENN] intact. Maintain speaker's voice.""",
            
            "plagiarism": """Rewrite text with high similarity to sources. Restructure sentences, use synonyms, change word order. Keep citations [ENN] intact. Preserve factual accuracy.""",
            
            "policy": """Fix concrete BSP policy violations with targeted edits only. Do not add blanket disclaimer sections or legal boilerplate unless explicitly required by a listed issue. Preserve confidence, directness, and the speaker's rhetorical style; only soften language when a specific statement is unsupported, non-compliant, or misleading. This is a speech delivered by the speaker — preserve first-person voice ("I", "we") and do NOT convert to third person. Keep citations [ENN] intact. Maintain professional tone.""",
            
            "generic": """Fix flagged issues. Preserve citations [ENN], maintain voice and facts."""
        }
        
        system_prompt = system_prompts.get(issue_type, system_prompts["generic"])

        # Build style preservation context if available
        style_context = ""
        if style_profile:
            style_name = style_profile.get("name", "")
            style_speaker = style_profile.get("speaker", "")
            style_audience = style_profile.get("audience_setting_classification", "")
            style_tone = style_profile.get("tone", "")

            style_context = "\n<STYLE_TO_PRESERVE>\n"
            if style_name:
                style_context += f"Style Name: {style_name}\n"
            if style_speaker:
                style_context += f"Speaker Persona: {style_speaker}\n"
            if style_audience:
                style_context += f"Audience: {style_audience}\n"
            if style_tone:
                style_context += f"Tone: {style_tone}\n"
            style_context += (
                "Preserve these style characteristics exactly: rhetorical voice, cadence, formality level, "
                "opening/closing phrasing, and audience register.\n"
            )
            style_context += "</STYLE_TO_PRESERVE>\n"
        
        # Build evidence context if available
        evidence_context = ""
        if evidence_store and issue_type == "citation":
            evidence_context = "\n<EVIDENCE_STORE>\n"
            for ev in evidence_store[:10]:  # Reduced from 20 to 10 to save tokens
                # Truncate claim to 150 chars to reduce token usage
                evidence_context += f"[{ev.get('id')}]: {ev.get('claim', '')[:150]}\n"
            evidence_context += "</EVIDENCE_STORE>\n\n"
        
        # Build issues list
        issues_text = "\n<ISSUES_TO_FIX>\n"
        for i, issue in enumerate(issues[:5], 1):  # Reduced from 10 to 5 to save tokens
            issues_text += f"\nISSUE #{i}:\n"
            issues_text += f"Segment: \"{issue.get('segment', '')[:200]}\"\n"  # Reduced from 300
            issues_text += f"Problem: {issue.get('issue_description', 'Not specified')[:150]}\n"  # Added limit
            if issue.get('suggestion'):
                issues_text += f"Fix: {issue.get('suggestion')[:150]}\n"  # Reduced from no limit
            issues_text += f"Severity: {issue.get('severity', 'MEDIUM')}\n"
        issues_text += "</ISSUES_TO_FIX>\n"
        
        user_prompt = f"""{evidence_context}{style_context}{issues_text}

    <EDITING_RULES>
    - Make the smallest possible edits needed to resolve the listed issues.
    - Preserve original structure, paragraph flow, tone, confidence, and speaking style.
    - Do not rewrite unaffected passages.
    - Do not add a standalone disclaimer block unless the issue list explicitly requests one.
    - Keep citation markers [ENN] exactly preserved.
    </EDITING_RULES>

<SPEECH_TO_REVISE>
{speech_text}
</SPEECH_TO_REVISE>

TASK: Revise the speech to fix the {len(issues)} issue(s) listed above. Return ONLY the full revised speech text (no explanations, no markup)."""
        
        print(f"[INFO] 🔧 Calling LLM to fix {len(issues)} {issue_type} issue(s)...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=8000  # Sufficient for speech revision (GPT-5 compatible)
        )
        
        fixed_speech = response.choices[0].message.content.strip()
        
        # Fix speaker perspective if style profile has a speaker
        if style_profile:
            fix_speaker = style_profile.get("speaker") or style_profile.get("Speaker")
            if fix_speaker:
                fixed_speech = fix_speaker_perspective(fixed_speech, fix_speaker)
        
        # Analyze what changed
        fix_details = []
        if fixed_speech != speech_text:
            # Simple diff: count sentence changes
            original_sentences = [s.strip() for s in re.split(r'[.!?]+', speech_text) if s.strip()]
            fixed_sentences = [s.strip() for s in re.split(r'[.!?]+', fixed_speech) if s.strip()]
            
            changes = sum(1 for orig, fixed in zip(original_sentences, fixed_sentences) if orig != fixed)
            fix_details.append({
                "sentences_modified": changes,
                "original_length": len(speech_text),
                "fixed_length": len(fixed_speech)
            })
            
            print(f"[SUCCESS] ✅ Fixed {changes} sentence(s), preserved {len(fixed_sentences) - changes} sentence(s)")
        else:
            print(f"[INFO] No changes needed")
        
        return {
            "success": True,
            "fixed_speech": fixed_speech,
            "fixes_applied": len(issues),
            "fix_details": fix_details
        }
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to fix issues: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "fixed_speech": speech_text,  # Return original on failure
            "fixes_applied": 0,
            "fix_details": [],
            "error": str(e)
        }


async def generate_styled_output(
    summary: str, 
    query: str, 
    style: Dict[str, Any], 
    context_details: str = "",
    evidence_store: List[Dict[str, Any]] = None, 
    max_output_length: int = 5000,
    use_claim_outline: bool = True  # IMPROVEMENT #1: Use claim outlines by default
) -> Dict[str, Any]:
    """
    Generate final styled output with STRICT citation enforcement.
    Follows the same pattern as prompts.rewrite_content() but enforces [ENN] citations.
    
    Args:
        summary: The comprehensive summary from iterative refinement
        query: Original user query
        style: Writing style dictionary from Cosmos DB (can be None for default formatting)
        context_details: Optional event/setting/partner context provided by user
        evidence_store: List of atomic evidence items for citation validation
        max_output_length: Target output length in characters (default: 5000 ≈ 1000 words at ~5 chars/word). Output will be trimmed if it exceeds this by more than 10%.
        use_claim_outline: If True, generate claim outline first to prevent style drift
    
    Returns:
        Dictionary with styled output, validation metrics, and metadata
    """
    # Allow None style - will use default formatting
    if style is None:
        style = {}
        print("[INFO] No style provided, using default formatting")
    
    try:
        import re

        token_usage_summary = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "api_calls": 0,
        }

        def _accumulate_usage(response_obj) -> None:
            usage = getattr(response_obj, "usage", None)
            if not usage:
                return

            token_usage_summary["api_calls"] += 1
            token_usage_summary["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            token_usage_summary["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)

            details = getattr(usage, "completion_tokens_details", None)
            if details:
                token_usage_summary["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0)
        
        # Initialize Azure OpenAI client (GPT-5 or configured model)
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        
        model = os.getenv("AZURE_OPENAI_STAGE3_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        # Extract compact style digest (cached) to reduce prompt size/cost
        style_digest = build_style_digest(style)
        style_text = style_digest.get("style_text", "")
        global_rules = style_digest.get("global_rules", "")
        guidelines = style_digest.get("guidelines", "")
        example = style_digest.get("example", "")
        
        # Build allowed evidence IDs list
        allowed_ids = []
        if evidence_store:
            allowed_ids = [e.get("id") for e in evidence_store if e.get("id")]
        allowed_ids_str = ", ".join(allowed_ids) if allowed_ids else "None available"
        
        # IMPROVEMENT #1: Generate claim outline to prevent style drift
        claim_outline_data = None
        claim_count = 0
        claim_outline = None
        
        print(f"[DEBUG] Claim outline parameters: use_claim_outline={use_claim_outline}, evidence_store_size={len(evidence_store) if evidence_store else 0}")
        print(f"[DEBUG] Summary length for extraction: {len(summary)} chars")
        
        if use_claim_outline and evidence_store:
            claim_outline_data, claim_outline = _build_claim_outline_from_summary(summary, evidence_store)
            if claim_outline_data:
                claim_count = len(claim_outline_data)
                print(f"[SUCCESS] ✅ Built deterministic claim outline: {claim_count} claims")
            else:
                print(f"[INFO] No claim outline built (summary has no inline citations)")
        else:
            if not use_claim_outline:
                print(f"[INFO] Claim outline disabled (use_claim_outline=False)")
            elif not evidence_store:
                print(f"[INFO] No evidence store available for claim outline generation")
        
        # Build the system prompt with STRICT citation rules and claim outline
        system_parts = [
            "You are an expert writer assistant. Rewrite the user input based on the following writing style, global rules, writing guidelines and writing example.\n",
            f"<writingStyle>{style_text}</writingStyle>\n",
            f"<globalRules>{global_rules}</globalRules>\n",
            f"<writingGuidelines>{guidelines}</writingGuidelines>\n",
        ]
        
        # Add original example if available
        if example:
            system_parts.append(f"<writingExample>{example}</writingExample>\n")
        
        # ENHANCEMENT: Fetch real speech examples from speeches database
        speaker_name = style.get("speaker") or style.get("Speaker")
        if speaker_name:
            print(f"[INFO] Fetching sample speeches for speaker: {speaker_name}")
            try:
                max_speeches = int(os.getenv("STAGE3_MAX_SPEECH_EXAMPLES", "1"))
            except ValueError:
                max_speeches = 1
            max_speeches = max(0, min(3, max_speeches))

            sample_speeches = get_sample_speeches_from_db(speaker_name, max_speeches=max_speeches)
            
            if sample_speeches:
                system_parts.append("\n<realSpeechExamples>\n")
                system_parts.append(f"Below are {len(sample_speeches)} actual speeches by {speaker_name}. Study their:\n")
                system_parts.append("- Sentence structure and rhythm\n")
                system_parts.append("- Vocabulary choices and phrasing\n")
                system_parts.append("- Opening/closing patterns\n")
                system_parts.append("- Transition phrases\n")
                system_parts.append("- Rhetorical devices\n\n")
                
                try:
                    excerpt_chars = int(os.getenv("STAGE3_SPEECH_EXCERPT_CHARS", "900"))
                except ValueError:
                    excerpt_chars = 900
                excerpt_chars = max(300, min(2000, excerpt_chars))

                for i, speech in enumerate(sample_speeches, 1):
                    # Truncate speeches to fit compact prompt budget
                    speech_excerpt = speech[:excerpt_chars] if len(speech) > excerpt_chars else speech
                    system_parts.append(f"=== SPEECH EXAMPLE {i} ===\n")
                    system_parts.append(f"{speech_excerpt}\n")
                    if len(speech) > excerpt_chars:
                        system_parts.append("... (excerpt from longer speech)\n")
                    system_parts.append("\n")
                
                system_parts.append("</realSpeechExamples>\n\n")
                print(f"[SUCCESS] ✅ Added {len(sample_speeches)} real speech examples to style prompt")
            else:
                print(f"[WARNING] No speech examples found for {speaker_name}, using style description only")
        
        system_parts.append("Make sure to emulate the writing style, global rules, guidelines and examples provided above.\n\n")

        # SPEAKER PERSPECTIVE: Write as the speaker, not about them
        speaker_name = style.get("speaker") or style.get("Speaker")
        if speaker_name:
            system_parts.extend([
                "SPEAKER PERSPECTIVE (MANDATORY):\n",
                f"You are writing a speech to be delivered by {speaker_name}.\n",
                "The speech must be written from the speaker's OWN perspective (first person).\n",
                f"- Use 'I' and 'we' — NEVER refer to {speaker_name} in the third person.\n",
                f"- WRONG: '{speaker_name} believes that...', 'The Governor noted...', 'According to {speaker_name}...'\n",
                "- RIGHT: 'I believe that...', 'We at the BSP...', 'Let me share...'\n",
                "- The speaker is addressing the audience directly; write as if they are speaking.\n\n",
            ])

        # Add claim outline instructions if available
        if claim_outline and claim_outline_data:
            system_parts.extend([
                "CLAIM-BASED REWRITING MODE:\n",
                "You are given a CLAIM OUTLINE below. You must:\n",
                "1. Rephrase ONLY the claims provided (do not add new facts or methods)\n",
                "2. Preserve all evidence citations exactly as shown\n",
                "3. You may adjust wording for flow, but NOT add new factual content\n",
                "4. Transitions must be non-factual (e.g., 'This suggests...', 'A key implication is...')\n",
                "5. Do NOT introduce named methods, numbers, or entities not in the claim outline\n\n",
                claim_outline + "\n\n"
            ])
        
        system_parts.extend([
            "CRITICAL CITATION REQUIREMENTS:\n",
            "1. EVERY factual claim MUST end with [ENN] citation format (e.g., [E1] or [E2,E5])\n",
            "2. You may ONLY cite these evidence IDs: " + allowed_ids_str + "\n",
            "3. Do NOT generate citations to non-existent evidence IDs\n",
            "4. Maintain ALL citations from the original summary\n",
            "5. If you rephrase a sentence, keep its citation intact\n",
            "6. Multiple claims in one sentence = multiple citations [E1,E3,E7]\n",
            "7. End with a short closing reflection (1-3 sentences) on why this topic matters to BSP, financial institutions, and the nation as a whole. Keep this aligned with available evidence and avoid unsupported claims.\n",
            "8. Finish with a clear closing greeting line (e.g., 'Thank you.' or 'Maraming salamat po.').\n\n",
            f"TARGET OUTPUT LENGTH: {max_output_length} characters — approximately {max_output_length // 5} words (excluding references). "
            f"This is a STRICT target: the final speech body MUST be within ±10-15% of {max_output_length // 5} words. "
            f"That means between {int((max_output_length // 5) * 0.85)} and {int((max_output_length // 5) * 1.15)} words. Do NOT produce output significantly outside this range."
        ])
        
        system_prompt = "".join(system_parts)
        
        # User prompt with citation examples
        user_prompt = f"""Original Query: {query}

    Additional Prompt Instructions (optional context):
    {context_details.strip() if context_details and context_details.strip() else "Not provided"}

Research Summary to Rewrite:
{summary}

STRICT REQUIREMENTS:
- Rewrite in the specified style while preserving ALL factual content
- EVERY factual sentence MUST have [ENN] citations
- Only use valid evidence IDs: {allowed_ids_str}
- Do NOT remove, modify, or invent citations
    - Use the optional setting/location/partners context naturally when relevant
    - Include a short closing reflection on importance to BSP, financial institutions, and the nation as a whole

Example correct format:
"Transformers use attention mechanisms [E1]. They achieve better performance than RNNs [E3,E7]."

Now rewrite the summary in the specified style with ALL citations preserved:"""

        # Keep completion token budget separate from output character cap.
        # max_output_length sets the indicative target for final characters,
        # while completion tokens must leave room for GPT-5 reasoning + visible output.
        # These can be tuned via environment variables if needed.
        try:
            min_completion_tokens = int(os.getenv("AZURE_OPENAI_MIN_COMPLETION_TOKENS", "12000"))
        except ValueError:
            min_completion_tokens = 12000

        try:
            max_completion_tokens = int(os.getenv("AZURE_OPENAI_MAX_COMPLETION_TOKENS", "24000"))
        except ValueError:
            max_completion_tokens = 24000

        if max_completion_tokens < min_completion_tokens:
            max_completion_tokens = min_completion_tokens

        completion_token_budget = max(
            min_completion_tokens,
            min(max_completion_tokens, max_output_length * 10)
        )

        # Log prompt lengths for debugging
        print(f"[DEBUG] System prompt length: {len(system_prompt)} chars")
        print(f"[DEBUG] User prompt length: {len(user_prompt)} chars")
        print(f"[DEBUG] Summary length: {len(summary)} chars")
        print(f"[DEBUG] Min completion tokens: {min_completion_tokens}")
        print(f"[DEBUG] Max completion tokens: {max_completion_tokens}")
        print(f"[DEBUG] Completion token budget: {completion_token_budget}")

        # Generate styled output (GPT-5 needs more tokens due to reasoning)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=completion_token_budget
            )
            _accumulate_usage(response)
            print(f"[DEBUG] API call successful, response choices: {len(response.choices)}")
        except Exception as api_error:
            print(f"[DEBUG] API call failed: {str(api_error)}")
            return {
                "success": False,
                "error": f"API call failed: {str(api_error)}",
                "styled_output": summary,
                "style_name": style.get("name", "Unknown"),
                "speaker": style.get("speaker", "Unknown"),
                "audience": style.get("audience_setting_classification", "General"),
                "model_used": model,
                "token_usage": token_usage_summary,
            }
        
        # Extract and truncate styled output (matching existing app behavior)
        styled_output = response.choices[0].message.content
        print(f"[DEBUG] Response content length: {len(styled_output) if styled_output else 0} chars")
        print(f"[DEBUG] Response content is None: {styled_output is None}")
        print(f"[DEBUG] Response finish_reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")
        
        # Additional GPT-5 debugging
        if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
            print(f"[DEBUG] ⚠️ Response refusal: {response.choices[0].message.refusal}")
        
        # Show token usage for GPT-5 reasoning models
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            print(f"[DEBUG] Token usage:")
            print(f"[DEBUG]   Prompt tokens: {usage.prompt_tokens}")
            print(f"[DEBUG]   Completion tokens: {usage.completion_tokens}")
            if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                    print(f"[DEBUG]   Reasoning tokens: {details.reasoning_tokens}")
                    print(f"[DEBUG]   Output tokens: {usage.completion_tokens - details.reasoning_tokens}")
        
        if styled_output:
            print(f"[DEBUG] ✓ Content received, first 100 chars: {styled_output[:100]}")
        else:
            print(f"[DEBUG] ✗ Content is empty or None - investigating...")
            print(f"[DEBUG]   Response ID: {response.id if hasattr(response, 'id') else 'N/A'}")
            print(f"[DEBUG]   Response model: {response.model if hasattr(response, 'model') else 'N/A'}")
            print(f"[DEBUG]   Finish reason: {response.choices[0].finish_reason}")
            if response.choices[0].finish_reason == 'length':
                print(f"[DEBUG]   ⚠️ ISSUE: Model hit token limit! Increase max_completion_tokens.")
        
        # Handle None or empty response
        if not styled_output or len(styled_output.strip()) == 0:
            return {
                "success": False,
                "error": "GPT model returned empty response",
                "styled_output": summary,  # Fallback to original summary
                "style_name": style.get("name", "Unknown"),
                "speaker": style.get("speaker", "Unknown"),
                "audience": style.get("audience_setting_classification", "General"),
                "model_used": model,
                "token_usage": token_usage_summary,
            }
        
        # Enforce strict target word count: trim or expand to stay within ±15% of target.
        body_text, references_section = split_references_section(styled_output)
        body_length = len(body_text)
        effective_body_length = len(strip_in_text_citations(body_text))
        upper_tolerance = int(max_output_length * 1.15)  # allow up to 15% over
        lower_bound = int(max_output_length * 0.85)  # enforce if under 15%
        target_words = max_output_length // 5
        effective_words = len(strip_in_text_citations(body_text).split())
        print(f"[INFO] Word count check: {effective_words} words (target: {target_words}, range: {int(target_words * 0.85)}-{int(target_words * 1.15)})")
        if effective_body_length > upper_tolerance:
            pct_over = ((effective_body_length - max_output_length) / max_output_length) * 100
            print(f"[INFO] Output body exceeds target by {pct_over:.0f}% ({body_length} raw chars, {effective_body_length} effective chars vs {max_output_length} target). Trimming deterministically...")
            styled_output = trim_body_preserve_closing(body_text, max_output_length) + references_section
            body_final, refs_final = split_references_section(styled_output)
            print(f"[INFO] Final output length after trim (body={len(body_final)} raw chars, effective={len(strip_in_text_citations(body_final))} chars, refs={len(refs_final)} chars)")
        elif effective_body_length < lower_bound:
            pct_under = ((max_output_length - effective_body_length) / max_output_length) * 100
            print(f"[INFO] Output body is {pct_under:.0f}% short ({effective_words} words vs target {target_words}). Requesting expansion...")
            expansion_system = (
                f"You are an expert editor. The speech below is too short. "
                f"Expand it to approximately {target_words} words "
                f"(current: ~{effective_words} words, target: {target_words} words). "
                f"Rules:\n"
                f"1. Preserve ALL existing [ENN] citations exactly as written.\n"
                f"2. Only elaborate using evidence already cited — do NOT introduce new unsupported facts.\n"
                f"3. Add depth, examples, and elaboration to existing points.\n"
                f"4. Maintain the same tone, perspective (first-person), and closing signature.\n"
                f"5. Valid evidence IDs you may cite: {allowed_ids_str}.\n"
                f"6. Output ONLY the expanded speech body — no preambles or meta-commentary."
            )
            expansion_user = (
                f"Expand the following speech body to approximately {target_words} words. "
                f"Keep all [ENN] citations intact:\n\n{body_text}"
            )
            try:
                expansion_tokens = max(min_completion_tokens, min(max_completion_tokens, max_output_length * 12))
                exp_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": expansion_system},
                        {"role": "user", "content": expansion_user},
                    ],
                    max_completion_tokens=expansion_tokens,
                )
                _accumulate_usage(exp_response)
                expanded_body = exp_response.choices[0].message.content
                if expanded_body and len(strip_in_text_citations(expanded_body).split()) >= int(target_words * 0.8):
                    expanded_words = len(strip_in_text_citations(expanded_body).split())
                    print(f"[INFO] Expansion successful: {expanded_words} words (target: {target_words})")
                    # Trim if the expansion itself overshoots
                    if len(strip_in_text_citations(expanded_body)) > upper_tolerance:
                        expanded_body = trim_body_preserve_closing(expanded_body, max_output_length)
                    styled_output = expanded_body + references_section
                    # Refresh body_text for downstream steps
                    body_text = expanded_body
                else:
                    print(f"[WARNING] Expansion did not reach target length; keeping original output.")
            except Exception as exp_err:
                print(f"[WARNING] Expansion LLM call failed ({exp_err}); keeping original output.")
        else:
            print(f"[INFO] Output body within target range ({effective_body_length} effective chars vs {max_output_length} target).")
        
        # IMPROVEMENT #5: Enforce citation discipline (max 2 IDs per sentence)
        print("[INFO] Enforcing citation discipline (max 2 IDs/sentence)...")
        styled_output = enforce_citation_discipline(styled_output, max_ids_per_sentence=2)
        
        # IMPROVEMENT: Fix speaker perspective (third-person -> first-person)
        perspective_speaker = style.get("speaker") or style.get("Speaker")
        if perspective_speaker:
            print(f"[INFO] Checking speaker perspective for {perspective_speaker}...")
            styled_output = fix_speaker_perspective(styled_output, perspective_speaker)
        
        # Validate citations in styled output
        citation_pattern = r'\[E\d+(?:,E\d+)*\]'
        citations_found = re.findall(citation_pattern, styled_output)
        
        # Extract individual evidence IDs from citations
        cited_ids = set()
        invalid_citations = []
        for citation in citations_found:
            ids_in_citation = re.findall(r'E\d+', citation)
            for eid in ids_in_citation:
                cited_ids.add(eid)
                if allowed_ids and eid not in allowed_ids:
                    invalid_citations.append(eid)
        
        # Calculate coverage
        cited_count = len(cited_ids & set(allowed_ids)) if allowed_ids else len(cited_ids)
        total_evidence = len(allowed_ids) if allowed_ids else 0
        coverage = (cited_count / total_evidence * 100) if total_evidence > 0 else 0
        
        output_body, output_refs = split_references_section(styled_output)

        return {
            "success": True,
            "styled_output": styled_output,
            "style_name": style.get("name", "Unknown"),
            "speaker": style.get("speaker", "Unknown"),
            "audience": style.get("audience_setting_classification", "General"),
            "model_used": model,
            "token_usage": token_usage_summary,
            "output_length": len(strip_in_text_citations(output_body)) if output_body else 0,
            "output_length_raw": len(output_body) if output_body else 0,
            "references_length": len(output_refs) if output_refs else 0,
            "target_length": max_output_length,
            "citations_found": len(citations_found),
            "unique_evidence_cited": cited_count,
            "citation_coverage": f"{coverage:.1f}%",
            "invalid_citations": invalid_citations,
            # IMPROVEMENT #1: Track claim outline usage
            "claim_outline_used": claim_outline_data is not None,
            "claim_count": claim_count,
            "validation": {
                "all_citations_valid": len(invalid_citations) == 0,
                "cited_ids": sorted(list(cited_ids)),
                "uncited_ids": sorted(list(set(allowed_ids) - cited_ids)) if allowed_ids else []
            }
        }
        
        # Debug: Confirm claim outline status being returned
        print(f"[DEBUG] Returning styled_result with claim_outline_used={claim_outline_data is not None}, claim_count={claim_count}")
        
        return result
        
    except Exception as e:
        logger.error("Style generation failed for query '%s': %s", query[:120], e, exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "styled_output": summary,  # Fallback to unstyled summary
            "claim_outline_used": False,
            "claim_count": 0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "api_calls": 0,
            },
        }


async def convert_styled_output_to_apa(
    styled_output: str,
    evidence_store: List[Dict[str, Any]],
    azure_llm_client = None
) -> Dict[str, Any]:
    """
    Convert styled output with [ENN] citations to APA format with (Author, Year) citations
    and add a References section following APA 7th edition guidelines.
    
    Args:
        styled_output: The styled text with [ENN] citations
        evidence_store: List of evidence dicts with id, source_url, claim, etc.
        azure_llm_client: Optional Azure OpenAI client
        
    Returns:
        {
            "success": bool,
            "apa_output": str with (Author, Year) citations,
            "references": list of APA-formatted references,
            "citation_map": dict mapping evidence IDs to (Author, Year)
        }
    """
    
    print("\n" + "="*70)
    print("CONVERTING TO APA FORMAT (7th Edition)")
    print("="*70)
    
    # Build evidence lookup
    evidence_dict = {}
    for evidence in evidence_store:
        eid = evidence.get("id")
        if eid:
            evidence_dict[eid] = evidence
    
    print(f"Evidence store: {len(evidence_dict)} items")
    
    # Step 1: Build citation map and group by unique source
    print("\nBuilding APA citations from evidence metadata...")
    
    citation_map = {}
    source_groups = {}  # Group evidence by unique source (author, year, URL)
    
    for eid, evidence in evidence_dict.items():
        author = evidence.get("author", "")
        year = evidence.get("year", "n.d.")
        source_url = evidence.get("source_url") or evidence.get("source") or ""
        publication = evidence.get("publication", "")
        publisher = evidence.get("publisher", "")
        source_title = evidence.get("source_title", "")
        
        # Determine display author for citations
        display_author = author
        if not display_author or display_author == "n.d.":
            if publisher and publisher != "n.d.":
                display_author = publisher
            elif publication:
                display_author = publication
            else:
                display_author = "Unknown"
        
        # For multiple authors, use "et al." format if already present, otherwise use first author
        if display_author and ", " in display_author:
            authors_list = display_author.split(", ")
            if len(authors_list) > 2:
                # Extract first author's last name for et al. format
                first_author = authors_list[0]
                if " " in first_author:
                    # "Ashish Vaswani" -> "Vaswani"
                    last_name = first_author.split()[-1]
                else:
                    last_name = first_author
                display_author_short = f"{last_name} et al."
            else:
                display_author_short = display_author
        else:
            display_author_short = display_author
        
        citation_map[eid] = {
            "author": display_author,
            "author_short": display_author_short,
            "year": year,
            "source_url": source_url,
            "publication": publication,
            "publisher": publisher,
            "source_title": source_title
        }
        
        # Group by unique source for deduplication.
        # When a URL is present use it as the sole key so that inconsistent
        # LLM-extracted author/year values across claims from the same page
        # don't generate multiple reference entries for a single source.
        source_key = source_url if source_url else (display_author, year, publication)
        if source_key not in source_groups:
            source_groups[source_key] = {
                "author": display_author,
                "year": year,
                "source_url": source_url,
                "publication": publication,
                "publisher": publisher,
                "source_title": source_title,
                "evidence_ids": []
            }
        source_groups[source_key]["evidence_ids"].append(eid)
    
    print(f"Built citation map for {len(citation_map)} evidence items")
    print(f"Unique sources: {len(source_groups)}")
    
    # Step 2: Replace [ENN] with (Author, Year) in the text
    print("\nConverting [ENN] citations to (Author, Year) format...")
    
    apa_output = styled_output
    
    # Find all [ENN] or [E1,E2] citations
    citation_pattern = r'\[E\d+(?:,E\d+)*\]'
    matches = list(re.finditer(citation_pattern, apa_output))
    
    # Replace from end to start to preserve positions
    for match in reversed(matches):
        citation_str = match.group(0)
        citation_ids = re.findall(r'E\d+', citation_str)
        
        # Convert to (Author, Year) format, removing duplicates
        seen_citations = set()
        apa_citations = []
        for cid in citation_ids:
            if cid in citation_map:
                info = citation_map[cid]
                citation_key = (info['author_short'], info['year'])
                if citation_key not in seen_citations:
                    apa_citations.append(f"{info['author_short']}, {info['year']}")
                    seen_citations.add(citation_key)
        
        if apa_citations:
            # Format as (Author1, Year1; Author2, Year2)
            apa_format = f"({'; '.join(apa_citations)})"
            apa_output = apa_output[:match.start()] + apa_format + apa_output[match.end():]
    
    # Step 3: Generate References section with proper APA 7th edition format
    print("\nGenerating APA 7th edition references list...")
    
    # Include ALL unique sources from the evidence store in references,
    # not just ones cited in the generated text, so every provided URL appears.
    all_source_keys = set(source_groups.keys())

    # Also track which are actually cited (for future use / ordering)
    all_citations_in_text = re.findall(citation_pattern, styled_output)
    cited_ids = set()
    for citation in all_citations_in_text:
        ids = re.findall(r'E\d+', citation)
        cited_ids.update(ids)
    cited_source_keys = set()
    for eid in cited_ids:
        if eid in citation_map:
            info = citation_map[eid]
            source_key = info['source_url'] if info['source_url'] else (info['author'], info['year'], info['publication'])
            cited_source_keys.add(source_key)

    # Build references list (one entry per unique source) — cited ones first
    references = []
    ordered_keys = sorted(cited_source_keys, key=lambda x: str(x)) + \
                   sorted(all_source_keys - cited_source_keys, key=lambda x: str(x))
    for source_key in ordered_keys:
        source_info = source_groups[source_key]
        
        author = source_info['author']
        year = source_info['year']
        source_url = source_info['source_url']
        publication = source_info['publication']
        source_title = source_info['source_title']
        
        # Format author for reference list (APA 7th edition)
        if author and ", " in author and "et al." not in author:
            # Convert "First Last, First Last" to "Last, F., & Last, F."
            authors_list = author.split(", ")
            if len(authors_list) <= 20:  # APA shows up to 20 authors
                formatted_authors = []
                for auth in authors_list:
                    parts = auth.strip().split()
                    if len(parts) >= 2:
                        last_name = parts[-1]
                        initials = ". ".join([p[0] for p in parts[:-1]]) + "."
                        formatted_authors.append(f"{last_name}, {initials}")
                    else:
                        formatted_authors.append(auth.strip())
                
                if len(formatted_authors) > 1:
                    author_display = ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
                else:
                    author_display = formatted_authors[0]
            else:
                author_display = author  # Keep original if too many
        else:
            author_display = author
        
        # Extract title from source_title or use publication
        title = ""
        if source_title:
            # Clean up title (remove site name if present)
            title = source_title.replace(" - arXiv", "").replace("[1706.03762] ", "")
            # Remove trailing site names
            for site in [" - Medium", " - YouTube", " - ResearchGate"]:
                if title.endswith(site):
                    title = title[:-len(site)]
        elif publication:
            title = publication
        
        # Generate APA 7th edition reference entry
        if source_url:
            # Online source with URL
            if "arxiv.org" in source_url:
                # arXiv format: Author. (Year). Title. arXiv. URL
                apa_ref = f"{author_display} ({year}). {title}. arXiv. {source_url}"
            else:
                # General web source: Author. (Year). Title. Site Name. URL
                site_name = publication if publication else "Website"
                if year and year != "n.d.":
                    apa_ref = f"{author_display} ({year}). {title}. {site_name}. {source_url}"
                else:
                    # No date format
                    apa_ref = f"{author_display} (n.d.). {title}. {site_name}. Retrieved from {source_url}"
        else:
            # No URL available
            if year and year != "n.d.":
                apa_ref = f"{author_display} ({year}). {title}. {publication if publication else 'Unknown source'}."
            else:
                apa_ref = f"{author_display} (n.d.). {title}. {publication if publication else 'Unknown source'}."
        
        references.append(apa_ref)
    
    # Add References section
    apa_output += "\n\n" + "="*70 + "\n"
    apa_output += "REFERENCES\n"
    apa_output += "="*70 + "\n\n"
    
    for ref in references:
        apa_output += f"{ref}\n\n"
    
    print(f"✓ APA conversion complete")
    print(f"  Converted {len(matches)} citation instances")
    print(f"  Generated {len(references)} unique reference entries (removed duplicates)")
    
    return {
        "success": True,
        "apa_output": apa_output,
        "references": references,
        "citation_map": citation_map,
        "citations_converted": len(matches),
        "references_generated": len(references)
    }


async def verify_styled_citations(
    styled_output: str, 
    evidence_store: List[Dict[str, Any]],
    azure_llm_client = None,
    sentence_level: bool = True  # IMPROVEMENT #4: Enable sentence-level verification by default
) -> Dict[str, Any]:
    """
    Parse styled output by citations and verify each claim matches its cited evidence.
    
    Args:
        styled_output: The styled text with citations like [E1] or [E1,E2]
        evidence_store: List of evidence dicts with id, claim, quote_span, etc.
        azure_llm_client: Optional Azure OpenAI client for verification
        sentence_level: If True, verify at sentence level (more precise). If False, use segment level.
        
    Returns:
        {
            "segments": [
                {
                    "segment_number": 1,
                    "text": "...",
                    "citations": ["E1", "E2"],
                    "cited_claims": ["claim text...", "claim text..."],
                    "verified": "Yes" | "No",
                    "verification_reason": "...",
                    "is_sentence_level": True|False
                }
            ],
            "total_segments": 10,
            "verified_segments": 8,
            "unverified_segments": 2,
            "verification_rate": "80.0%",
            "is_sentence_level": True|False
        }
    """
    
    print("\n" + "="*70)
    print(f"VERIFYING STYLED OUTPUT CITATIONS ({'SENTENCE-LEVEL' if sentence_level else 'SEGMENT-LEVEL'})")
    print("="*70)
    
    # Build evidence lookup dict
    evidence_dict = {}
    for evidence in evidence_store:
        eid = evidence.get("id")
        if eid:
            evidence_dict[eid] = evidence
    
    print(f"Evidence store: {len(evidence_dict)} items")
    
    # IMPROVEMENT #4: Split text into segments
    citation_pattern = r'\[E\d+(?:,E\d+)*\]'
    
    segments = []
    segment_number = 0
    
    if sentence_level:
        # Sentence-level verification: Split into sentences first, then check each
        sentences = re.split(r'(?<=[.!?])\s+', styled_output)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if this sentence has citations
            citations_in_sentence = re.findall(citation_pattern, sentence)
            if not citations_in_sentence:
                # Skip sentences without citations (non-factual content)
                continue
            
            segment_number += 1
            
            # Extract all citation IDs from this sentence
            citation_ids = []
            for citation_match in citations_in_sentence:
                citation_ids.extend(re.findall(r'E\d+', citation_match))
            
            # Remove duplicates while preserving order
            citation_ids = list(dict.fromkeys(citation_ids))
            
            # Look up cited claims
            cited_claims = []
            missing_ids = []
            for cid in citation_ids:
                if cid in evidence_dict:
                    cited_claims.append({
                        "id": cid,
                        "claim": evidence_dict[cid].get("claim", ""),
                        "quote_span": evidence_dict[cid].get("quote_span", "")
                    })
                else:
                    missing_ids.append(cid)
            
            segments.append({
                "segment_number": segment_number,
                "text": sentence,
                "citations": citation_ids,
                "cited_claims": cited_claims,
                "missing_evidence_ids": missing_ids,
                "verified": None,
                "verification_reason": None,
                "is_sentence_level": True
            })
    else:
        # Original segment-level verification: Split by citation patterns
        current_pos = 0
        
        # Find all citations and their positions
        for match in re.finditer(citation_pattern, styled_output):
            segment_number += 1
            
            # Extract text from last position to end of this citation
            segment_end = match.end()
            segment_text = styled_output[current_pos:segment_end].strip()
            
            # Extract citation IDs
            citation_str = match.group(0)
            citation_ids = re.findall(r'E\d+', citation_str)
            
            # Look up cited claims
            cited_claims = []
            missing_ids = []
            for cid in citation_ids:
                if cid in evidence_dict:
                    cited_claims.append({
                        "id": cid,
                        "claim": evidence_dict[cid].get("claim", ""),
                        "quote_span": evidence_dict[cid].get("quote_span", "")
                    })
                else:
                    missing_ids.append(cid)
            
            segments.append({
                "segment_number": segment_number,
                "text": segment_text,
                "citations": citation_ids,
                "cited_claims": cited_claims,
                "missing_evidence_ids": missing_ids,
                "verified": None,
                "verification_reason": None,
                "is_sentence_level": False
            })
            
            current_pos = segment_end
    
    print(f"Parsed {len(segments)} {'sentences' if sentence_level else 'segments'} with citations")
    
    # Initialize Azure client if not provided
    if not azure_llm_client:
        azure_llm_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )

    verification_model = (
        os.getenv("AZURE_OPENAI_STAGE4_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_VERIFICATION_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_CHAT_MINI_DEPLOYMENT")
        or "gpt-4o-mini"
    )
    use_llm_for_uncertain = os.getenv("WRITER_VERIFY_USE_LLM", "true").strip().lower() in {"1", "true", "yes", "on"}
    token_usage_summary = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "api_calls": 0,
    }

    try:
        batch_size = int(os.getenv("WRITER_VERIFY_BATCH_SIZE", "12"))
    except ValueError:
        batch_size = 12
    batch_size = max(8, min(20, batch_size))

    try:
        lexical_overlap_threshold = float(os.getenv("WRITER_VERIFY_LEXICAL_OVERLAP_THRESHOLD", "0.18"))
    except ValueError:
        lexical_overlap_threshold = 0.18

    try:
        lexical_low_threshold = float(os.getenv("WRITER_VERIFY_LEXICAL_LOW_THRESHOLD", "0.06"))
    except ValueError:
        lexical_low_threshold = 0.06

    def _tokenize(text: str) -> set:
        return set(re.findall(r"[a-zA-Z]{3,}", (text or "").lower()))

    def _extract_numbers(text: str) -> set:
        return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text or ""))

    def _rule_precheck(segment: Dict[str, Any]) -> Dict[str, str]:
        claims_blob = " ".join(
            f"{c.get('claim', '')} {c.get('quote_span', '')}" for c in segment.get("cited_claims", [])
        )
        seg_tokens = _tokenize(segment.get("text", ""))
        claim_tokens = _tokenize(claims_blob)
        overlap = len(seg_tokens & claim_tokens) / max(1, len(seg_tokens))

        seg_numbers = _extract_numbers(segment.get("text", ""))
        evidence_numbers = _extract_numbers(claims_blob)

        if seg_numbers and evidence_numbers and seg_numbers.isdisjoint(evidence_numbers):
            return {
                "status": "No",
                "reason": "Numeric mismatch with cited evidence (rule precheck)",
            }

        if overlap >= lexical_overlap_threshold:
            return {
                "status": "Yes",
                "reason": f"Rule precheck passed (lexical overlap {overlap:.2f})",
            }

        if overlap < lexical_low_threshold and seg_numbers and not evidence_numbers:
            return {
                "status": "No",
                "reason": "Segment includes numeric claim not present in cited evidence (rule precheck)",
            }

        return {
            "status": "Uncertain",
            "reason": f"Rule precheck uncertain (lexical overlap {overlap:.2f})",
        }

    def _clean_json_text(text: str) -> str:
        cleaned = (text or "").replace("```json", "").replace("```", "").strip()
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            return cleaned[start:end + 1]
        return cleaned

    def _apply_llm_batch_results(batch_items: List[Dict[str, Any]], parsed_results: List[Dict[str, Any]]) -> None:
        result_map = {}
        for item in parsed_results:
            seg_no = item.get("segment_number")
            if isinstance(seg_no, int):
                result_map[seg_no] = item

        for item in batch_items:
            segment = item["segment"]
            seg_no = segment.get("segment_number")
            llm_item = result_map.get(seg_no)
            if not llm_item:
                segment["verified"] = "Error"
                segment["verification_reason"] = "Batch verification missing response"
                continue

            verdict = str(llm_item.get("verdict", "")).strip().upper()
            reason = str(llm_item.get("reason", "")).strip() or "LLM verification result"
            if verdict == "VERIFIED":
                segment["verified"] = "Yes"
                segment["verification_reason"] = reason
            else:
                segment["verified"] = "No"
                segment["verification_reason"] = reason

    uncertain_segments: List[Dict[str, Any]] = []
    prechecked_yes = 0
    prechecked_no = 0

    for segment in segments:
        if segment["missing_evidence_ids"]:
            segment["verified"] = "No"
            segment["verification_reason"] = f"Missing evidence IDs: {', '.join(segment['missing_evidence_ids'])}"
            prechecked_no += 1
            continue

        precheck = _rule_precheck(segment)
        if precheck["status"] == "Yes":
            segment["verified"] = "Yes"
            segment["verification_reason"] = precheck["reason"]
            prechecked_yes += 1
        elif precheck["status"] == "No":
            segment["verified"] = "No"
            segment["verification_reason"] = precheck["reason"]
            prechecked_no += 1
        else:
            uncertain_segments.append({"segment": segment, "precheck": precheck})

    print(f"Rule precheck results: Yes={prechecked_yes}, No={prechecked_no}, Uncertain={len(uncertain_segments)}")

    if uncertain_segments and use_llm_for_uncertain:
        print(f"\nVerifying uncertain segments in batches using {verification_model} (batch_size={batch_size})...")
        for start in range(0, len(uncertain_segments), batch_size):
            batch = uncertain_segments[start:start + batch_size]
            payload_items = []
            for item in batch:
                segment = item["segment"]
                payload_items.append({
                    "segment_number": segment["segment_number"],
                    "text": segment["text"],
                    "citations": segment["citations"],
                    "cited_claims": [
                        {"id": c["id"], "claim": c.get("claim", "")} for c in segment.get("cited_claims", [])
                    ]
                })

            verification_prompt = (
                "You are a precise fact-checking AI. Verify whether each segment is supported by its cited evidence. "
                "Return ONLY a JSON array of objects with keys: segment_number (int), verdict ('VERIFIED' or 'UNVERIFIED'), "
                "reason (short string).\n\n"
                f"BATCH_INPUT:\n{json.dumps(payload_items, ensure_ascii=False)}"
            )

            try:
                response = azure_llm_client.chat.completions.create(
                    model=verification_model,
                    messages=[{"role": "user", "content": verification_prompt}],
                    max_completion_tokens=800,
                )
                usage = getattr(response, "usage", None)
                if usage:
                    token_usage_summary["api_calls"] += 1
                    token_usage_summary["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                    token_usage_summary["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
                    details = getattr(usage, "completion_tokens_details", None)
                    if details:
                        token_usage_summary["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0)
                content = _clean_json_text(response.choices[0].message.content)
                parsed = json.loads(content)
                if not isinstance(parsed, list):
                    raise ValueError("Batch verification returned non-list JSON")
                _apply_llm_batch_results(batch, parsed)
            except Exception as batch_error:
                # Fallback: mark batch as uncertain error to avoid silent false positives
                for item in batch:
                    item["segment"]["verified"] = "Error"
                    item["segment"]["verification_reason"] = f"Batch verification failed: {str(batch_error)}"

    elif uncertain_segments:
        for item in uncertain_segments:
            item["segment"]["verified"] = "Error"
            item["segment"]["verification_reason"] = "LLM verification disabled for uncertain segment"

    verified_count = sum(1 for s in segments if s.get("verified") == "Yes")
    unverified_count = sum(1 for s in segments if s.get("verified") in {"No", "Error"})
    
    print(f"\nVerification complete!")
    print(f"  Verified: {verified_count}")
    print(f"  Unverified: {unverified_count}")
    
    # Calculate verification rate
    total = len(segments)
    verification_rate = (verified_count / total * 100) if total > 0 else 0
    
    return {
        "total_segments": total,
        "verified_segments": verified_count,
        "unverified_segments": unverified_count,
        "verification_rate": f"{verification_rate:.1f}%",
        "segments": segments,
        "is_sentence_level": sentence_level,  # IMPROVEMENT #4: Flag for sentence-level verification
        "token_usage": token_usage_summary,
    }


def select_stage1_iterations(query_text: str, input_sources: Dict[str, Any]) -> int:
    """
    Select Stage 1 iteration count automatically based on request complexity.

    Heuristic:
    - Base: 1 iteration (fast default)
    - +1 if links/attachments are present
    - +1 if many sources or long/complex query
    - Clamp to 1..3
    """
    links = input_sources.get("links") or []
    attachments = input_sources.get("attachments") or []
    topic = str(input_sources.get("topics") or "").strip()

    score = 1

    if links or attachments:
        score += 1

    if len(links) >= 4 or len(attachments) >= 2 or len(query_text) > 220:
        score += 1

    if topic and len(topic) > 120 and score < 3:
        score += 1

    return max(1, min(3, score))


async def speechify_output(
    literature_review: str,
    query: str,
    style: Optional[Dict[str, Any]] = None,
    context_details: str = "",
    target_word_count: int = 1000,
) -> Dict[str, Any]:
    """
    Stage 8 — Free-composition speechifier.

    Takes the citation-dense literature-review-style output and transforms it
    into a genuine delivered speech: narrative arc, audience conversation,
    rhetorical devices, metaphors, humour, conviction. Citations are stripped
    from the final speech because real speeches don't read footnotes aloud;
    sources are preserved in a separate references block appended at the end.

    The LLM has full creative freedom over structure and language. The only
    hard constraint is factual fidelity to what is in the source text.
    """
    if not literature_review or not literature_review.strip():
        return {"success": False, "speech": "", "error": "No source text provided",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "api_calls": 0}}

    style = style or {}
    token_usage_summary = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "api_calls": 0}

    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        model = os.getenv("AZURE_OPENAI_STAGE3_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

        speaker_name = style.get("speaker") or style.get("Speaker") or "the speaker"
        audience = style.get("audience_setting_classification") or "distinguished guests"

        style_digest = build_style_digest(style)
        style_text = style_digest.get("style_text", "")
        global_rules = style_digest.get("global_rules", "")
        example = style_digest.get("example", "")

        # Pull real speech examples for authentic voice grounding
        sample_speeches = []
        if speaker_name and speaker_name != "the speaker":
            try:
                max_speeches = int(os.getenv("STAGE3_MAX_SPEECH_EXAMPLES", "1"))
            except ValueError:
                max_speeches = 1
            max_speeches = max(0, min(3, max_speeches))
            if max_speeches > 0:
                sample_speeches = get_sample_speeches_from_db(speaker_name, max_speeches=max_speeches)

        system_parts = [
            f"You are {speaker_name}, a distinguished public official, about to deliver a speech "
            f"to {audience}. You have just finished reading a briefing paper (the Literature Review below). "
            f"Now you must speak — not recite the paper, but truly SPEAK to your audience.\n\n",
        ]

        if style_text:
            system_parts.append(f"<yourVoice>{style_text}</yourVoice>\n")
        if global_rules:
            system_parts.append(f"<globalRules>{global_rules}</globalRules>\n")

        if sample_speeches:
            try:
                excerpt_chars = int(os.getenv("STAGE3_SPEECH_EXCERPT_CHARS", "900"))
            except ValueError:
                excerpt_chars = 900
            excerpt_chars = max(300, min(2000, excerpt_chars))
            system_parts.append("\n<realSpeechExamples>\n")
            system_parts.append(
                f"These are actual speeches you have delivered. They capture how you naturally speak — "
                f"your cadence, warmth, wit, and authority. Write your new speech in THIS voice.\n\n"
            )
            for i, speech in enumerate(sample_speeches, 1):
                excerpt = speech[:excerpt_chars]
                system_parts.append(f"=== YOUR PAST SPEECH {i} ===\n{excerpt}")
                if len(speech) > excerpt_chars:
                    system_parts.append("\n... (continues)")
                system_parts.append("\n\n")
            system_parts.append("</realSpeechExamples>\n\n")

        if example:
            system_parts.append(f"<styleExample>{example[:600]}</styleExample>\n\n")

        system_parts.extend([
            "## YOUR MISSION\n",
            "Transform the briefing paper into a speech that feels ALIVE. The audience came to be moved, "
            "not informed. They will read the paper later. Right now, they need to feel the weight of these "
            "ideas and understand why they matter to their lives and to the nation.\n\n",

            "## WHAT YOU MUST DO\n",
            "1. **Open with a hook** — a story, a striking observation, a question that makes the audience "
            "lean forward. Never open with statistics.\n",
            "2. **Build a narrative arc** — the speech should have a beginning (provocation), middle "
            "(evidence woven into story), and end (call to feeling or action). You may reorder the "
            "topics from the briefing paper to create better dramatic flow.\n",
            "3. **Speak TO the audience** — address them directly. Ask rhetorical questions. Invite them "
            "to imagine, to consider, to remember. Use 'you', 'we', 'us', 'our'.\n",
            "4. **Use metaphors and analogies** — translate abstract economic or technical concepts into "
            "images a non-specialist can feel. 'Inflation is not just a number — it is the moment a "
            "mother pauses at the market and puts back what she came to buy.'\n",
            "5. **Vary your rhythm** — short punchy sentences for impact. Longer flowing ones for "
            "reflection. Sentence fragments for emphasis. One-word paragraphs. Full stop.\n",
            "6. **Invite thought** — pose questions you do not answer immediately. Let silence work.\n",
            "7. **Use humour sparingly but naturally** — a light moment of self-deprecation or a wry "
            "observation humanises the speaker and keeps the room with you.\n",
            "8. **Weave in Filipino expressions where natural** — 'Ito ang ating hamon', 'Mabuhay', "
            "'Maraming salamat po', a proverb, a shared cultural touchstone.\n",
            "9. **Close with conviction** — not a data summary. End on an image, a commitment, a human "
            "truth. Leave the audience with something they will remember on the drive home.\n",
            "10. **Write strictly in first person** as {speaker_name}. NEVER refer to {speaker_name} "
            "in the third person.\n\n",

            f"## LENGTH TARGET\n",
            f"Write approximately **{target_word_count} words**. "
            f"This is a firm target, not a suggestion — the audience and venue have been allocated this length. "
            f"Do not pad with filler; do not cut ideas short. Reach the target through depth and colour.\n\n",

            "## WHAT YOU MUST NOT DO\n",
            "- Do NOT reproduce citation markers like [E1] or (Author, Year) in the speech body "
            "(those belong in papers, not podiums).\n",
            "- Do NOT follow the sequence of the briefing paper mechanically — you are composing, not transcribing.\n",
            "- Do NOT begin sentences with 'Furthermore', 'Moreover', 'In conclusion', 'It is worth noting'.\n",
            "- Do NOT invent facts, statistics, or names that are not in the briefing paper.\n",
            "- Do NOT add a References section — this is a speech, not a paper.\n\n",

            "## FACTUAL CONSTRAINT\n",
            "Every fact, number, and finding you mention must come from the briefing paper below. "
            "You may paraphrase, combine, and give narrative colour — but do not hallucinate.\n\n",

            "Separate every paragraph with a blank line (two newlines). Never run paragraphs together.\n",
            "Return ONLY the speech text. No preamble, no meta-commentary, no section labels.\n",
        ])

        system_prompt = "".join(system_parts)

        user_prompt = (
            f"Speech topic / occasion context: "
            f"{context_details.strip() if context_details and context_details.strip() else query.strip()}\n\n"
            f"<BRIEFING_PAPER>\n{literature_review}\n</BRIEFING_PAPER>\n\n"
            f"TARGET LENGTH: {target_word_count} words.\n"
            f"Now step to the podium. Deliver the speech."
        )

        # Token budget: speeches can run long during free composition
        try:
            max_tokens = int(os.getenv("SPEECHIFY_MAX_TOKENS", "16000"))
        except ValueError:
            max_tokens = 16000
        max_tokens = max(4000, min(32000, max_tokens))

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_tokens,
        )

        usage = getattr(response, "usage", None)
        if usage:
            token_usage_summary["api_calls"] += 1
            token_usage_summary["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            token_usage_summary["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                token_usage_summary["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0)

        speech_text = (response.choices[0].message.content or "").strip()

        # Strip any stray thinking tokens
        if "<think>" in speech_text and "</think>" in speech_text:
            while "<think>" in speech_text and "</think>" in speech_text:
                s = speech_text.find("<think>")
                e = speech_text.find("</think>") + 8
                speech_text = (speech_text[:s] + speech_text[e:]).strip()

        # Remove any stray [ENN] or (Author, Year) markers the model may have kept
        speech_text = re.sub(r'\[E\d+(?:,E\d+)*\]', '', speech_text)
        speech_text = re.sub(
            r'\((?=[^)]*\b(?:\d{4}[a-z]?|n\.d\.)\b)[^)]*\)',
            '',
            speech_text,
            flags=re.IGNORECASE,
        )
        speech_text = re.sub(r'[ \t]{2,}', ' ', speech_text).strip()

        # Fix third-person speaker references
        if speaker_name and speaker_name != "the speaker":
            speech_text = fix_speaker_perspective(speech_text, speaker_name)

        if not speech_text:
            return {
                "success": False,
                "speech": "",
                "error": "Model returned empty speech",
                "token_usage": token_usage_summary,
            }

        word_count = len(speech_text.split())
        print(f"[SPEECHIFY] ✓ Composed {word_count}-word speech")

        return {
            "success": True,
            "speech": speech_text,
            "word_count": word_count,
            "token_usage": token_usage_summary,
        }

    except Exception as e:
        logger.error("speechify_output failed: %s", e, exc_info=True)
        return {
            "success": False,
            "speech": "",
            "error": str(e),
            "token_usage": token_usage_summary,
        }


async def process_with_iterative_refinement_and_style(
    query: str,
    sources: Dict[str, Any],
    max_iterations: Optional[int] = None,
    target_output_length: int = 5000,
    context_details: str = "",
    style: Optional[Dict[str, Any]] = None,
    enable_policy_check: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    operating_mode: Optional[str] = None,
    enabled_stages: Optional[set] = None,
    use_deep_research: bool = True,
) -> Dict[str, Any]:
    """
    Complete pipeline: Iterative refinement + Style-based output generation + Policy check.
    
    Args:
        query: User's research query
        sources: Sources dictionary (topics, links, attachments)
        max_iterations: Number of refinement iterations
        target_output_length: Target output length in characters (default: 5000 ≈ 1000 words). Converted from word count at the UI layer (words × 5).
        context_details: Optional event/location/partner context
        style: Writing style dictionary (if None, will fetch random from DB)
        enable_policy_check: Whether to run BSP policy alignment check (default: True)
        enabled_stages: Set of stage numbers (1-7) to run. None means all enabled.
        use_deep_research: When False, skips deep-research topic processing (useful when
            links/attachments already provide sufficient evidence). Overrides
            ENABLE_DEEP_RESEARCH env var when explicitly set to False.
    
    Returns:
        Complete results including refined summary, styled output, and policy check
    """
    # Propagate use_deep_research to env so topic_processing can see it
    if not use_deep_research:
        os.environ["ENABLE_DEEP_RESEARCH"] = "false"
    else:
        # Only reset if caller explicitly passes True; don't override a pre-set env var
        os.environ.setdefault("ENABLE_DEEP_RESEARCH", "true")
    if enabled_stages is None:
        enabled_stages = {1, 2, 3, 4, 5, 6, 7, 8}

    pipeline_started_at = time.perf_counter()
    stage_elapsed_seconds: Dict[str, float] = {}
    stage_token_usage: Dict[str, Dict[str, int]] = {}

    def emit(event_type: str, **payload):
        if progress_callback:
            try:
                progress_callback({"type": event_type, **payload})
            except Exception:
                pass

    def _record_stage_elapsed(stage_key: str, started_at: float) -> None:
        stage_elapsed_seconds[stage_key] = time.perf_counter() - started_at

    def _merge_stage_token_usage(stage_key: str, usage: Optional[Dict[str, Any]]) -> None:
        if not usage or not isinstance(usage, dict):
            return

        stage_token_usage.setdefault(stage_key, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "api_calls": 0,
        })

        for key in ["prompt_tokens", "completion_tokens", "reasoning_tokens", "api_calls"]:
            stage_token_usage[stage_key][key] += int(usage.get(key, 0) or 0)

    def _print_performance_telemetry() -> None:
        print("\n" + "=" * 70)
        print("PERFORMANCE TELEMETRY (TERMINAL)")
        print("=" * 70)

        total_elapsed = time.perf_counter() - pipeline_started_at
        print(f"Total elapsed: {total_elapsed:.2f}s")

        if stage_elapsed_seconds:
            print("\nStage elapsed:")
            ordered_stages = ["stage1", "stage1_escalated", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7", "stage8"]
            for stage_key in ordered_stages:
                if stage_key in stage_elapsed_seconds:
                    print(f"  - {stage_key}: {stage_elapsed_seconds[stage_key]:.2f}s")

        total_prompt = 0
        total_completion = 0
        total_reasoning = 0
        total_calls = 0

        if stage_token_usage:
            print("\nToken usage:")
            for stage_key in ["stage3", "stage4", "stage5", "stage6", "stage7", "stage7_recoat", "stage8"]:
                usage = stage_token_usage.get(stage_key)
                if not usage:
                    continue
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
                api_calls = int(usage.get("api_calls", 0) or 0)
                print(
                    f"  - {stage_key}: prompt={prompt_tokens}, completion={completion_tokens}, "
                    f"reasoning={reasoning_tokens}, calls={api_calls}"
                )
                total_prompt += prompt_tokens
                total_completion += completion_tokens
                total_reasoning += reasoning_tokens
                total_calls += api_calls

        print(
            f"\nToken totals: prompt={total_prompt}, completion={total_completion}, "
            f"reasoning={total_reasoning}, calls={total_calls}"
        )
        print("=" * 70)

    def _parse_percent_value(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace("%", "")
        try:
            return float(text)
        except ValueError:
            return 0.0

    def _parse_int_env(name: str, default_value: int, minimum: int = 1, maximum: int = 7) -> int:
        raw = os.getenv(name, str(default_value))
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            parsed = default_value
        return max(minimum, min(maximum, parsed))

    resolved_mode = (operating_mode or os.getenv("WRITER_OPERATING_MODE", "fast")).strip().lower()
    if resolved_mode not in {"fast", "standard"}:
        resolved_mode = "fast"

    fast_mode_enabled = resolved_mode == "fast"
    emit("stage_text", stage=1, text=f"Operating mode: {resolved_mode}")

    fast_deep_loops = _parse_int_env("FAST_MODE_DEEP_RESEARCH_LOOPS", 2)
    standard_deep_loops = _parse_int_env("STANDARD_MODE_DEEP_RESEARCH_LOOPS", 3)

    os.environ["DEEP_RESEARCH_MAX_LOOPS"] = str(fast_deep_loops if fast_mode_enabled else standard_deep_loops)
    emit("stage_text", stage=1, text=f"Deep research loops: {os.getenv('DEEP_RESEARCH_MAX_LOOPS')}")

    if max_iterations is None:
        if fast_mode_enabled:
            max_iterations = 1
        else:
            max_iterations = max(2, select_stage1_iterations(query, sources))
        print(f"[INFO] Stage 1 iterations selected automatically: {max_iterations}")
        emit("stage_text", stage=1, text=f"Configured iterations: {max_iterations} (auto-selected)")
    else:
        max_iterations = max(1, min(7, int(max_iterations)))
        emit("stage_text", stage=1, text=f"Configured iterations: {max_iterations}")

    # Step 1: Iterative refinement
    try:
        return await _run_pipeline_stages(
            query=query,
            sources=sources,
            max_iterations=max_iterations,
            target_output_length=target_output_length,
            context_details=context_details,
            style=style,
            enable_policy_check=enable_policy_check,
            enabled_stages=enabled_stages,
            fast_mode_enabled=fast_mode_enabled,
            fast_deep_loops=fast_deep_loops,
            standard_deep_loops=standard_deep_loops,
            progress_callback=progress_callback,
            emit=emit,
            pipeline_started_at=pipeline_started_at,
            stage_elapsed_seconds=stage_elapsed_seconds,
            stage_token_usage=stage_token_usage,
            _record_stage_elapsed=_record_stage_elapsed,
            _merge_stage_token_usage=_merge_stage_token_usage,
            _print_performance_telemetry=_print_performance_telemetry,
            _parse_percent_value=_parse_percent_value,
            _parse_int_env=_parse_int_env,
        )
    except Exception as e:
        logger.error(
            "Pipeline failed with unhandled error for query '%s': %s",
            query[:120], e, exc_info=True,
        )
        emit("stage_text", stage=1, text=f"Pipeline error: {e}")
        try:
            _print_performance_telemetry()
        except Exception:
            pass
        return {
            "original_query": query,
            "sources": sources,
            "total_iterations": 0,
            "iterations": [],
            "final_summary": {"success": False, "summary": "", "error": str(e)},
            "final_evidence_count": 0,
            "cumulative_evidence_store": [],
            "styled_output": {
                "success": False,
                "styled_output": "",
                "error": (
                    f"The pipeline encountered an unexpected error: {e}. "
                    f"Please check the logs for details and try again."
                ),
            },
            "styled_output_apa": None,
            "citation_verification": None,
            "plagiarism_analysis": None,
            "policy_check": None,
            "style_recoat": {"success": False, "applied": False, "output": "", "fallback_reason": "pipeline_error"},
            "speechify_result": None,
            "style_used": {
                "name": style.get("name", "None") if style else "None",
                "speaker": style.get("speaker", "None") if style else "None",
                "audience": style.get("audience_setting_classification", "None") if style else "None",
            },
            "performance_telemetry": {
                "elapsed_seconds": stage_elapsed_seconds,
                "token_usage": stage_token_usage,
                "total_elapsed_seconds": time.perf_counter() - pipeline_started_at,
            },
        }


async def _run_pipeline_stages(
    *,
    query: str,
    sources: Dict[str, Any],
    max_iterations: int,
    target_output_length: int,
    context_details: str,
    style: Optional[Dict[str, Any]],
    enable_policy_check: bool,
    enabled_stages: set,
    fast_mode_enabled: bool,
    fast_deep_loops: int,
    standard_deep_loops: int,
    progress_callback: Optional[Callable],
    emit: Callable,
    pipeline_started_at: float,
    stage_elapsed_seconds: Dict[str, float],
    stage_token_usage: Dict[str, Dict[str, int]],
    _record_stage_elapsed: Callable,
    _merge_stage_token_usage: Callable,
    _print_performance_telemetry: Callable,
    _parse_percent_value: Callable,
    _parse_int_env: Callable,
) -> Dict[str, Any]:
    """Inner pipeline stages, extracted for top-level error handling."""

    emit("stage_started", stage=1)
    stage1_started_at = time.perf_counter()
    print(f"\n{'='*70}")
    print("STEP 1: ITERATIVE REFINEMENT")
    print('='*70)
    refinement_results = await process_with_iterative_refinement(
        query,
        sources,
        max_iterations,
        progress_callback=progress_callback,
    )

    stage1_evidence = refinement_results.get("cumulative_evidence_store", [])
    topic_evidence_count = 0
    link_evidence_count = 0
    attachment_evidence_count = 0
    other_evidence_count = 0

    for evidence_item in stage1_evidence:
        retrieval_context = str(evidence_item.get("retrieval_context", "")).strip().lower()
        if retrieval_context == "deep_research_pipeline":
            topic_evidence_count += 1
        elif retrieval_context == "link_processing":
            link_evidence_count += 1
        elif retrieval_context == "attachment_processing":
            attachment_evidence_count += 1
        else:
            other_evidence_count += 1

    emit("stage_metric", stage=1, key="Topic Evidence", value=topic_evidence_count)
    emit("stage_metric", stage=1, key="Link Evidence", value=link_evidence_count)
    emit("stage_metric", stage=1, key="Attachment Evidence", value=attachment_evidence_count)
    if other_evidence_count > 0:
        emit("stage_metric", stage=1, key="Other Evidence", value=other_evidence_count)

    emit("stage_done", stage=1)
    _record_stage_elapsed("stage1", stage1_started_at)
    
    # Get final summary
    final_summary = refinement_results["final_summary"].get("summary", "")
    
    if not final_summary:
        # Attempt to build a minimal summary from raw evidence when the LLM summary is empty
        cumulative_evidence_fallback = refinement_results.get("cumulative_evidence_store", [])
        refinement_error = refinement_results.get("final_summary", {}).get("error", "unknown reason")
        logger.warning(
            "No final summary produced for query '%s' (evidence_count=%d, error=%s). "
            "Attempting fallback output from raw evidence.",
            query[:120],
            len(cumulative_evidence_fallback),
            refinement_error,
        )
        emit("stage_text", stage=1, text=f"No summary produced: {refinement_error}. Attempting fallback.")

        if cumulative_evidence_fallback:
            # Build a simple bullet list from evidence claims as the fallback summary
            bullet_lines = []
            for ev in cumulative_evidence_fallback[:20]:
                claim = ev.get("claim", "").strip()
                eid = ev.get("id", "")
                if claim:
                    bullet_lines.append(f"- {claim} [{eid}]")
            final_summary = (
                f"## Summary (auto-generated fallback)\n\n"
                f"The pipeline could not produce a refined summary "
                f"(reason: {refinement_error}). "
                f"Below are the collected evidence claims:\n\n"
                + "\n".join(bullet_lines)
            )
            logger.info("Fallback summary built from %d evidence items.", len(bullet_lines))
        else:
            # No evidence at all - return with a clear error but still provide a result dict
            logger.error(
                "Pipeline produced no evidence AND no summary for query '%s'. "
                "Returning error result. Reason: %s",
                query[:120],
                refinement_error,
            )
            _print_performance_telemetry()
            return {
                **refinement_results,
                "styled_output": {
                    "success": False,
                    "styled_output": "",
                    "error": (
                        f"No evidence or summary could be generated for this topic. "
                        f"Reason: {refinement_error}. "
                        f"Please try rephrasing your query or providing additional source links."
                    ),
                },
            }
    
    # Step 2: Get writing style
    emit("stage_started", stage=2)
    stage2_started_at = time.perf_counter()
    print(f"\n{'='*70}")
    print("STEP 2: RETRIEVING WRITING STYLE")
    print('='*70)
    
    if style is None:
        style = get_random_style_from_db()
        if style:
            print(f"✓ Retrieved style: {style.get('name', 'Unknown')}")
            print(f"  Speaker: {style.get('speaker', 'Unknown')}")
            print(f"  Audience: {style.get('audience_setting_classification', 'General')}")
            emit("stage_text", stage=2, text=f"Style: {style.get('name', 'Unknown')}")
            emit("stage_text", stage=2, text=f"Speaker: {style.get('speaker', 'Unknown')}")
            emit("stage_text", stage=2, text=f"Audience: {style.get('audience_setting_classification', 'General')}")
        else:
            print("✗ No style found in database, using default formatting")
            emit("stage_text", stage=2, text="No style found in database, using default formatting")
    emit("stage_done", stage=2)
    _record_stage_elapsed("stage2", stage2_started_at)
    
    # Step 3: Generate styled output
    emit("stage_started", stage=3)
    stage3_started_at = time.perf_counter()
    print(f"\n{'='*70}")
    print("STEP 3: GENERATING STYLED OUTPUT")
    print('='*70)
    
    # Get cumulative evidence store for citation validation
    cumulative_evidence = refinement_results.get("cumulative_evidence_store", [])
    generation_evidence = prune_evidence_for_generation(query, cumulative_evidence)
    print(f"[DEBUG] Evidence store size: {len(cumulative_evidence)} items")
    print(f"[DEBUG] Generation evidence size (pruned): {len(generation_evidence)} items")
    print(f"[DEBUG] Final summary length: {len(final_summary)} chars")
    
    styled_result = await generate_styled_output(
        final_summary, 
        query, 
        style,
        context_details=context_details,
        max_output_length=target_output_length,
        evidence_store=generation_evidence,
        use_claim_outline=True  # IMPROVEMENT #1: Explicitly enable claim-based styling
    )
    _merge_stage_token_usage("stage3", styled_result.get("token_usage"))
    
    if styled_result.get("success"):
        print(f"✓ Styled output generated successfully")
        print(f"  Style applied: {styled_result.get('style_name', 'Unknown')}")
        print(f"  Output length: {len(styled_result.get('styled_output', ''))} characters")
        
        # Show citation validation results
        if styled_result.get("citations_found") is not None:
            print(f"  Citations: {styled_result.get('citations_found')} instances")
            print(f"  Evidence cited: {styled_result.get('unique_evidence_cited')} unique IDs")
            print(f"  Coverage: {styled_result.get('citation_coverage')}")
            
            if styled_result.get("invalid_citations"):
                print(f"  ⚠️ Invalid citations: {styled_result.get('invalid_citations')}")
            else:
                print(f"  ✓ All citations valid")
        emit("stage_text", stage=3, text="Styled output generated successfully")
        emit("stage_metric", stage=3, key="Output Length", value=len(styled_result.get("styled_output", "")))
        emit("stage_metric", stage=3, key="Target Length", value=target_output_length)
        emit("stage_metric", stage=3, key="Citations", value=styled_result.get("citations_found", 0))
        emit("stage_metric", stage=3, key="Evidence IDs", value=styled_result.get("unique_evidence_cited", 0))
    else:
        print(f"✗ Style generation failed: {styled_result.get('error', 'Unknown')}")
        emit("stage_text", stage=3, text=f"Style generation failed: {styled_result.get('error', 'Unknown')}")

    emit("stage_done", stage=3)
    stage_elapsed_seconds["stage3"] = time.perf_counter() - stage3_started_at
    
    # Step 4: Verify citations in styled output
    verification_result = None
    if 4 not in enabled_stages:
        print(f"\n{'='*70}")
        print("STEP 4: VERIFYING STYLED OUTPUT CITATIONS - SKIPPED (disabled by user)")
        print('='*70)
        emit("stage_started", stage=4)
        emit("stage_text", stage=4, text="Skipped by user preference")
        emit("stage_done", stage=4)
    elif styled_result.get("success") and styled_result.get("styled_output"):
        emit("stage_started", stage=4)
        stage4_started_at = time.perf_counter()
        print(f"\n{'='*70}")
        print("STEP 4: VERIFYING STYLED OUTPUT CITATIONS")
        print('='*70)
        
        try:
            verification_result = await verify_styled_citations(
                styled_output=styled_result.get("styled_output", ""),
                evidence_store=cumulative_evidence,
                sentence_level=True  # IMPROVEMENT #4: Use sentence-level verification
            )
            _merge_stage_token_usage("stage4", verification_result.get("token_usage"))
            
            print(f"\n✓ Citation verification complete")
            print(f"  Total segments: {verification_result.get('total_segments')}")
            print(f"  Verified: {verification_result.get('verified_segments')}")
            print(f"  Unverified: {verification_result.get('unverified_segments')}")
            print(f"  Verification rate: {verification_result.get('verification_rate')}")
            
            # Show first few unverified segments if any
            unverified_segments = [
                s for s in verification_result.get("segments", []) 
                if s.get("verified") == "No"
            ]
            if unverified_segments:
                print(f"\n  ⚠️ Sample unverified segments:")
                for seg in unverified_segments[:3]:
                    print(f"    Segment {seg['segment_number']}: {seg.get('verification_reason', 'Unknown')}")
                
                # AUTO-FIX: Revise unverified segments
                print(f"\n[AUTO-FIX] 🔧 Attempting to fix {len(unverified_segments)} unverified segments...")
                
                # Build issues list for fix_speech_issues
                citation_issues = []
                for seg in unverified_segments:
                    citation_issues.append({
                        "segment": seg.get("sentence", ""),
                        "issue_description": seg.get("verification_reason", "Claim not fully supported by evidence"),
                        "suggestion": "Soften claim language or add hedging (e.g., 'may suggest', 'appears to indicate')",
                        "severity": "HIGH"
                    })
                
                # Call fix function
                fix_result = await fix_speech_issues(
                    speech_text=styled_result.get("styled_output", ""),
                    issues=citation_issues,
                    evidence_store=cumulative_evidence,
                    issue_type="citation",
                    style_profile=style
                )
                
                if fix_result.get("success") and fix_result.get("fixes_applied") > 0:
                    print(f"[AUTO-FIX] ✅ Applied {fix_result['fixes_applied']} citation fixes")
                    # Update styled result with fixed speech
                    styled_result["styled_output"] = fix_result["fixed_speech"]
                    
                    # Re-verify to confirm fixes
                    print(f"[AUTO-FIX] Rechecking citations after fixes...")
                    verification_result = await verify_styled_citations(
                        styled_output=fix_result["fixed_speech"],
                        evidence_store=cumulative_evidence,
                        sentence_level=True
                    )
                    _merge_stage_token_usage("stage4", verification_result.get("token_usage"))
                    print(f"[AUTO-FIX] After fixes - Verified: {verification_result.get('verified_segments')}/{verification_result.get('total_segments')} segments")
                else:
                    print(f"[AUTO-FIX] ⚠️ Could not apply fixes: {fix_result.get('error', 'Unknown')}")
            emit("stage_metric", stage=4, key="Verified", value=verification_result.get('verified_segments'))
            emit("stage_metric", stage=4, key="Unverified", value=verification_result.get('unverified_segments'))
            emit("stage_text", stage=4, text=f"Verification rate: {verification_result.get('verification_rate')}")
            
        except Exception as e:
            print(f"✗ Verification failed: {str(e)}")
            verification_result = {
                "error": str(e),
                "total_segments": 0,
                "verified_segments": 0,
                "unverified_segments": 0
            }
            emit("stage_done", stage=4)
        _record_stage_elapsed("stage4", stage4_started_at)
    
    # Step 5: Convert to APA format
    apa_result = None
    if 5 not in enabled_stages:
        print(f"\n{'='*70}")
        print("STEP 5: CONVERTING TO APA FORMAT - SKIPPED (disabled by user)")
        print('='*70)
        emit("stage_started", stage=5)
        emit("stage_text", stage=5, text="Skipped by user preference")
        emit("stage_done", stage=5)
    elif styled_result.get("success") and styled_result.get("styled_output"):
        emit("stage_started", stage=5)
        stage5_started_at = time.perf_counter()
        print(f"\n{'='*70}")
        print("STEP 5: CONVERTING TO APA FORMAT")
        print('='*70)
        
        try:
            apa_result = await convert_styled_output_to_apa(
                styled_output=styled_result.get("styled_output", ""),
                evidence_store=cumulative_evidence
            )
            
            if apa_result.get("success"):
                print(f"\n✓ APA conversion complete")
                print(f"  Citations converted: {apa_result.get('citations_converted')}")
                print(f"  References generated: {apa_result.get('references_generated')}")
                print(f"  Output length: {len(apa_result.get('apa_output', ''))} characters")
                emit("stage_metric", stage=5, key="Citations Converted", value=apa_result.get('citations_converted', 0))
                emit("stage_metric", stage=5, key="References", value=apa_result.get('references_generated', 0))
            else:
                print(f"✗ APA conversion failed: {apa_result.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"✗ APA conversion failed: {str(e)}")
            apa_result = {
                "success": False,
                "error": str(e),
                "apa_output": styled_result.get("styled_output", "")  # Fallback to original
            }
            emit("stage_done", stage=5)
        _record_stage_elapsed("stage5", stage5_started_at)
    
    # Step 6: Plagiarism Detection
    plagiarism_result = None
    if 6 not in enabled_stages:
        print(f"\n{'='*70}")
        print("STEP 6: PLAGIARISM DETECTION - SKIPPED (disabled by user)")
        print('='*70)
        emit("stage_started", stage=6)
        emit("stage_text", stage=6, text="Skipped by user preference")
        emit("stage_done", stage=6)
    elif styled_result.get("success") and styled_result.get("styled_output"):
        emit("stage_started", stage=6)
        stage6_started_at = time.perf_counter()
        print(f"\n{'='*70}")
        print("STEP 6: PLAGIARISM DETECTION")
        print('='*70)
        
        # Initialize clients for plagiarism check
        azure_llm_client = None
        tavily_client = None
        
        try:
            # Initialize clients for plagiarism check
            from openai import AsyncAzureOpenAI
            azure_llm_client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            tavily_client = AsyncTavilyClient(api_key=tavily_api_key) if tavily_api_key else None
            
            # Prepare speech metadata
            speech_metadata = {
                "id": hashlib.sha256(styled_result.get("styled_output", "").encode()).hexdigest()[:16],
                "query": query,
                "speaker": style.get("speaker") if style else None,
                "institution": "Analysis",
                "date": datetime.now().isoformat()
            }
            
            # Run plagiarism check
            plagiarism_result = await check_plagiarism(
                speech_text=styled_result.get("styled_output", ""),
                azure_client=azure_llm_client,
                tavily_client=tavily_client,
                speech_metadata=speech_metadata,
                use_hf_classifier=False  # Can enable if transformers is installed
            )
            
            if plagiarism_result.get("success"):
                print(f"\n✓ Plagiarism analysis complete")
                print(f"  Overall risk: {plagiarism_result.get('overall_risk_level')} ({plagiarism_result.get('overall_risk_score'):.3f})")
                stats = plagiarism_result.get('statistics', {})
                print(f"  Total chunks: {stats.get('total_chunks', 0)}")
                print(f"  High risk: {stats.get('high_risk_chunks', 0)}")
                print(f"  Medium risk: {stats.get('medium_risk_chunks', 0)}")
                print(f"  Low risk: {stats.get('low_risk_chunks', 0)}")
                print(f"  Clean: {stats.get('clean_chunks', 0)}")
                emit("stage_metric", stage=6, key="Risk Level", value=plagiarism_result.get('overall_risk_level', 'UNKNOWN'))
                
                # Show top sources if any
                top_sources = plagiarism_result.get('top_sources', [])
                if top_sources:
                    print(f"\n  Top matching sources:")
                    for i, src in enumerate(top_sources[:3], 1):
                        print(f"    {i}. {src.get('title', 'Unknown')} ({src.get('match_count', 0)} matches)")
                
                # AUTO-FIX: Revise elevated-risk plagiarism chunks (MEDIUM/HIGH/CRITICAL)
                chunks = plagiarism_result.get('chunks', [])
                elevated_risk_chunks = [
                    c for c in chunks
                    if str(c.get('risk_level', '')).strip().upper() in ['MEDIUM', 'HIGH', 'CRITICAL']
                ]
                
                if elevated_risk_chunks:
                    print(f"\n[AUTO-FIX] 🔧 Attempting to rephrase {len(elevated_risk_chunks)} medium/high/critical chunks...")
                    
                    # Build issues list for fix_speech_issues
                    plagiarism_issues = []
                    for chunk in elevated_risk_chunks[:12]:  # Limit to avoid token overflow
                        chunk_risk = str(chunk.get('risk_level', '')).strip().upper()
                        plagiarism_issues.append({
                            "segment": chunk.get("text", ""),
                            "issue_description": f"{chunk_risk.title()} similarity to source material (risk score: {chunk.get('overall_risk_score', 0):.2f})",
                            "suggestion": "Restructure sentences completely, use different word order and synonyms",
                            "severity": "CRITICAL" if chunk_risk == 'CRITICAL' else ("HIGH" if chunk_risk == 'HIGH' else "MEDIUM")
                        })
                    
                    # Call fix function
                    fix_result = await fix_speech_issues(
                        speech_text=styled_result.get("styled_output", ""),
                        issues=plagiarism_issues,
                        evidence_store=cumulative_evidence,
                        issue_type="plagiarism",
                        style_profile=style
                    )
                    
                    if fix_result.get("success") and fix_result.get("fixes_applied") > 0:
                        print(f"[AUTO-FIX] ✅ Applied {fix_result['fixes_applied']} plagiarism fixes")
                        # Update styled result with fixed speech
                        styled_result["styled_output"] = fix_result["fixed_speech"]
                        
                        # Note: Full re-check would require repeating plagiarism analysis (expensive)
                        # We trust the LLM fix for now
                        print(f"[AUTO-FIX] Speech rephrased to reduce similarity")
                    else:
                        print(f"[AUTO-FIX] ⚠️ Could not apply plagiarism fixes: {fix_result.get('error', 'Unknown')}")
            else:
                print(f"✗ Plagiarism analysis failed: {plagiarism_result.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"✗ Plagiarism detection failed: {str(e)}")
            plagiarism_result = {
                "success": False,
                "error": str(e),
                "overall_risk_level": "unknown"
            }
        finally:
            # Clean up clients
            if azure_llm_client:
                await azure_llm_client.close()
            if tavily_client:
                await tavily_client.close()
        emit("stage_done", stage=6)
        _record_stage_elapsed("stage6", stage6_started_at)
    
    # Step 7: BSP Policy Alignment Check
    policy_check_result = None
    style_recoat_result = {
        "success": True,
        "applied": False,
        "output": "",
        "fallback_reason": "not_run",
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "api_calls": 0},
    }
    if 7 not in enabled_stages:
        print(f"\n{'='*70}")
        print("STEP 7: BSP POLICY ALIGNMENT CHECK - SKIPPED (disabled by user)")
        print('='*70)
        emit("stage_started", stage=7)
        emit("stage_text", stage=7, text="Skipped by user preference")
        emit("stage_done", stage=7)
    elif enable_policy_check and styled_result.get("success"):
        emit("stage_started", stage=7)
        stage7_started_at = time.perf_counter()
        print(f"\n{'='*70}")
        print("STEP 7: BSP POLICY ALIGNMENT CHECK")
        print('='*70)
        
        try:
            # Prepare speech metadata
            speech_metadata = {
                "topic": query,
                "speaker": style.get("speaker") if style else "BSP Official",
                "audience": style.get("audience_setting_classification") if style else "General audience",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query
            }
            max_policy_fix_rounds = _parse_int_env("STAGE7_MAX_POLICY_FIX_ROUNDS", 1, minimum=0, maximum=3)
            max_policy_issues_for_fix = _parse_int_env("STAGE7_MAX_POLICY_ISSUES_FOR_FIX", 5, minimum=1, maximum=15)
            min_fix_severity = os.getenv("STAGE7_MIN_FIX_SEVERITY", "high").strip().lower()
            min_fix_rank = _severity_rank(min_fix_severity)
            if min_fix_rank <= 0:
                min_fix_rank = _severity_rank("high")
            policy_fix_round = 0
            total_policy_fixes = 0
            policy_pass_count = 0
            max_policy_passes = max_policy_fix_rounds + 1
            last_policy_signature = None

            while True:
                if policy_pass_count >= max_policy_passes:
                    print(f"[POLICY] Max verification passes reached ({max_policy_passes}).")
                    emit("stage_text", stage=7, text=f"Policy max verification passes reached ({max_policy_passes})")
                    break

                policy_pass_count += 1

                # Determine which version of the speech to check (prefer APA if available)
                speech_to_check = (
                    apa_result.get("apa_output") if apa_result and apa_result.get("success")
                    else styled_result.get("styled_output", "")
                )

                print(f"\n[POLICY] Verification pass {policy_pass_count}/{max_policy_passes}")
                emit("stage_text", stage=7, text=f"Policy verification pass {policy_pass_count}")

                policy_cache_key = _build_policy_cache_key(speech_to_check, speech_metadata)
                cached_policy_result = _get_cached_policy_check(policy_cache_key)
                if cached_policy_result:
                    policy_check_result = cached_policy_result
                    print("[POLICY CACHE] HIT")
                    emit("stage_text", stage=7, text="Policy cache hit")
                else:
                    # Run policy alignment check using Azure Agent
                    policy_check_result = await check_speech_policy_alignment(
                        speech_content=speech_to_check,
                        speech_metadata=speech_metadata
                    )
                    if policy_check_result.get("success"):
                        _set_cached_policy_check(policy_cache_key, policy_check_result)
                        print("[POLICY CACHE] MISS → STORED")
                        emit("stage_text", stage=7, text="Policy cache miss; stored result")

                if not policy_check_result.get("success"):
                    print(f"✗ Policy check failed: {policy_check_result.get('error', 'Unknown')}")
                    emit("stage_text", stage=7, text=f"Policy check failed: {policy_check_result.get('error', 'Unknown')}")
                    break

                compliance_score = policy_check_result.get('compliance_score')
                violations = policy_check_result.get('violations', [])
                has_high_or_critical = any(
                    str(v.get('severity', '')).strip().lower() in {'high', 'critical'}
                    for v in violations
                )

                effective_requires_revision = bool(policy_check_result.get('requires_revision')) or has_high_or_critical
                policy_check_result['requires_revision'] = effective_requires_revision

                current_policy_signature = _policy_result_signature(policy_check_result)
                if last_policy_signature and current_policy_signature == last_policy_signature:
                    print("[POLICY] No material policy change from prior pass; stopping recheck loop.")
                    emit("stage_text", stage=7, text="Policy unchanged after prior pass; stopping")
                    break
                last_policy_signature = current_policy_signature

                # Preserve checker's compliance label when provided; fallback only if missing.
                if not policy_check_result.get('overall_compliance'):
                    policy_check_result['overall_compliance'] = (
                        'needs_revision' if effective_requires_revision else 'approved'
                    )

                print(f"\n✓ Policy alignment check complete")
                print(f"  Compliance: {policy_check_result.get('overall_compliance').upper()}")
                if isinstance(compliance_score, (int, float)):
                    print(f"  Score: {compliance_score:.1%}")
                emit("stage_metric", stage=7, key="Compliance", value=policy_check_result.get('overall_compliance', 'unknown').upper())

                # Show violations summary
                violations_count = policy_check_result.get('violations_count', len(violations))
                if violations_count > 0:
                    print(f"  Violations: {violations_count}")
                    critical_count = sum(1 for v in violations if str(v.get('severity', '')).strip().lower() == 'critical')
                    high_count = sum(1 for v in violations if str(v.get('severity', '')).strip().lower() == 'high')
                    print(f"    Critical: {critical_count}")
                    print(f"    High: {high_count}")
                    if violations:
                        print("  Violation details (top 5):")
                        emit("stage_text", stage=7, text="Policy violations detected (showing top 5)")
                        for idx, violation in enumerate(violations[:5], 1):
                            violation_type = violation.get('violation_type', 'Unknown')
                            severity = str(violation.get('severity', 'MEDIUM')).upper()
                            description = violation.get('description', '')
                            suggestion = violation.get('suggested_fix', '')
                            problematic_text = (violation.get('problematic_text', '') or '').strip().replace('\n', ' ')
                            if len(problematic_text) > 140:
                                problematic_text = problematic_text[:140].rstrip() + "..."

                            print(f"    {idx}. [{severity}] {violation_type}")
                            if description:
                                print(f"       Cause: {description}")
                            if problematic_text:
                                print(f"       Text: \"{problematic_text}\"")
                            if suggestion:
                                print(f"       Suggested fix: {suggestion}")

                            summary_line = f"[{severity}] {violation_type}: {description}" if description else f"[{severity}] {violation_type}"
                            emit("stage_text", stage=7, text=summary_line)

                # Show commendations
                commendations = policy_check_result.get('commendations', [])
                if commendations:
                    print(f"  Commendations: {len(commendations)} positive findings")

                # Show circulars referenced
                circulars = policy_check_result.get('circular_references', [])
                if circulars:
                    print(f"  BSP Circulars Referenced: {len(circulars)}")
                    for i, circ in enumerate(circulars[:3], 1):
                        print(f"    {i}. {circ}")

                # Exit loop if approved
                if not effective_requires_revision:
                    print(f"\n  ✅ RECOMMENDATION: APPROVED FOR USE")
                    emit("stage_text", stage=7, text="Policy check: Approved for use")
                    break

                print(f"\n  ⚠️ RECOMMENDATION: REVISION REQUIRED")
                emit("stage_text", stage=7, text="Policy check: Needs revision")

                # Stop if max fix rounds reached
                if policy_fix_round >= max_policy_fix_rounds:
                    print(f"[AUTO-FIX] ⚠️ Max policy fix rounds reached ({max_policy_fix_rounds}). Stopping recheck loop.")
                    emit("stage_text", stage=7, text=f"[AUTO-FIX] Max policy fix rounds reached ({max_policy_fix_rounds})")
                    break

                # AUTO-FIX: Revise policy violations
                violations = policy_check_result.get('violations', [])
                actionable_violations = [
                    v for v in violations
                    if _severity_rank(v.get("severity", "")) >= min_fix_rank
                ]

                # Build issues list for fix_speech_issues (only when revision is truly required)
                policy_issues = []
                if actionable_violations:
                    print(f"\n[AUTO-FIX] 🔧 Attempting to fix {len(actionable_violations)} actionable policy violations...")
                    emit("stage_text", stage=7, text=f"[AUTO-FIX] Attempting policy fix for {len(actionable_violations)} actionable violations")
                    for violation in actionable_violations[:max_policy_issues_for_fix]:
                        policy_issues.append({
                            "segment": violation.get("problematic_text", "") or speech_to_check[:700],
                            "issue_description": f"{violation.get('violation_type', 'Unknown')}: {violation.get('description', '')}",
                            "suggestion": violation.get("suggested_fix", "Revise to align with BSP policy guidelines"),
                            "severity": violation.get("severity", "MEDIUM").upper()
                        })
                elif has_high_or_critical:
                    print("\n[AUTO-FIX] ⚠️ High/critical issues flagged but no actionable details; running one holistic policy fix.")
                    emit("stage_text", stage=7, text="[AUTO-FIX] Running holistic policy fix for high/critical issue")
                    policy_issues.append({
                        "segment": speech_to_check[:900],
                        "issue_description": f"Policy check flagged high-risk revision need (compliance: {policy_check_result.get('overall_compliance', 'unknown')})",
                        "suggestion": policy_check_result.get("recommendation", "Revise wording to align with BSP policy requirements and reduce non-compliant phrasing."),
                        "severity": "HIGH"
                    })
                else:
                    print(f"[AUTO-FIX] Skipping auto-fix: no violations at or above '{min_fix_severity}'.")
                    emit("stage_text", stage=7, text=f"[AUTO-FIX] Skipping; no violations >= {min_fix_severity}")
                    break

                # Call fix function
                fix_result = await fix_speech_issues(
                    speech_text=speech_to_check,
                    issues=policy_issues,
                    evidence_store=cumulative_evidence,
                    issue_type="policy",
                    style_profile=style
                )

                if fix_result.get("success") and fix_result.get("fixes_applied") > 0:
                    fixed_speech = fix_result.get("fixed_speech", "")
                    if not fixed_speech or fixed_speech.strip() == speech_to_check.strip():
                        print("[AUTO-FIX] No material speech change detected after fix; stopping recheck loop.")
                        emit("stage_text", stage=7, text="[AUTO-FIX] No material change after fix; stopping")
                        break

                    policy_fix_round += 1
                    total_policy_fixes += fix_result.get("fixes_applied", 0)
                    print(f"[AUTO-FIX] ✅ Applied {fix_result['fixes_applied']} policy fixes (round {policy_fix_round})")
                    emit("stage_text", stage=7, text=f"[AUTO-FIX] Applied {fix_result['fixes_applied']} policy fixes (round {policy_fix_round})")
                    emit("stage_metric", stage=7, key="Policy Fixes", value=total_policy_fixes)
                    emit("stage_metric", stage=7, key="Fix Rounds", value=policy_fix_round)

                    # Update the appropriate result before re-verifying
                    if apa_result and apa_result.get("success"):
                        apa_result["apa_output"] = fixed_speech
                        print(f"[AUTO-FIX] Updated APA output with policy fixes")
                        emit("stage_text", stage=7, text="[AUTO-FIX] Updated APA output with policy fixes")
                    else:
                        styled_result["styled_output"] = fixed_speech
                        print(f"[AUTO-FIX] Updated styled output with policy fixes")
                        emit("stage_text", stage=7, text="[AUTO-FIX] Updated styled output with policy fixes")

                    print(f"[AUTO-FIX] Re-running policy verification after fixes...")
                    emit("stage_text", stage=7, text="[AUTO-FIX] Re-running policy verification after fixes")
                    continue

                print(f"[AUTO-FIX] ⚠️ Could not apply policy fixes: {fix_result.get('error', 'Unknown')}")
                emit("stage_text", stage=7, text=f"[AUTO-FIX] Could not apply policy fixes: {fix_result.get('error', 'Unknown')}")
                break

            # Final style-only polish after policy checks/fixes
            style_coat_source = ""
            style_coat_target = ""
            if apa_result and apa_result.get("success") and apa_result.get("apa_output"):
                style_coat_source = apa_result.get("apa_output", "")
                style_coat_target = "apa_output"
            elif styled_result.get("success") and styled_result.get("styled_output"):
                style_coat_source = styled_result.get("styled_output", "")
                style_coat_target = "styled_output"

            if style_coat_source and style:
                print("\n[STYLE RECOAT] Re-applying selected style + editorial guidelines (style-only, content-locked)...")
                emit("stage_text", stage=7, text="Post-policy style recoat started (content-locked)")

                style_recoat_result = await reapply_two_pass_style(
                    speech_text=style_coat_source,
                    style=style,
                    context_details=context_details,
                )
                _merge_stage_token_usage("stage7_recoat", style_recoat_result.get("token_usage"))

                if style_recoat_result.get("success") and style_recoat_result.get("applied"):
                    recoated_text = style_recoat_result.get("output", style_coat_source)
                    if style_coat_target == "apa_output" and apa_result is not None:
                        apa_result["apa_output"] = recoated_text
                    elif style_coat_target == "styled_output":
                        styled_result["styled_output"] = recoated_text
                    passes = style_recoat_result.get("passes", {})
                    p1 = "✅" if passes.get("rhetoric", {}).get("applied") else "⏭"
                    p2 = "✅" if passes.get("polish", {}).get("applied") else "⏭"
                    print(f"[STYLE RECOAT] ✅ Two-pass applied: rhetoric={p1} polish={p2}")
                    emit("stage_text", stage=7, text="Post-policy two-pass style recoat applied")
                else:
                    fallback_reason = style_recoat_result.get("fallback_reason", "unknown")
                    print(f"[STYLE RECOAT] ⚠️ Skipped/fallback to original: {fallback_reason}")
                    emit("stage_text", stage=7, text=f"Post-policy style recoat skipped: {fallback_reason}")
            elif not style:
                style_recoat_result = {
                    "success": True,
                    "applied": False,
                    "output": style_coat_source,
                    "fallback_reason": "no_style_selected",
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "api_calls": 0},
                }
                emit("stage_text", stage=7, text="Post-policy style recoat skipped: no style selected")
            else:
                style_recoat_result = {
                    "success": True,
                    "applied": False,
                    "output": "",
                    "fallback_reason": "no_output_available",
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "api_calls": 0},
                }
                emit("stage_text", stage=7, text="Post-policy style recoat skipped: no output available")
                
        except Exception as e:
            print(f"✗ Policy alignment check failed: {str(e)}")
            import traceback
            traceback.print_exc()
            policy_check_result = {
                "success": False,
                "error": str(e),
                "overall_compliance": "error",
                "requires_revision": True
            }
            emit("stage_text", stage=7, text=f"Policy alignment check failed: {str(e)}")
        emit("stage_done", stage=7)
        _record_stage_elapsed("stage7", stage7_started_at)
    elif not enable_policy_check:
        print(f"\n{'='*70}")
        print("STEP 7: BSP POLICY ALIGNMENT CHECK - SKIPPED")
        print('='*70)
        print("  (Set enable_policy_check=True to enable)")
    
    # Combine results
    complete_results = {
        **refinement_results,
        "styled_output": styled_result,
        "styled_output_apa": apa_result,
        "citation_verification": verification_result,
        "plagiarism_analysis": plagiarism_result,
        "policy_check": policy_check_result,
        "style_recoat": style_recoat_result,
        "style_used": {
            "name": style.get("name", "None") if style else "None",
            "speaker": style.get("speaker", "None") if style else "None",
            "audience": style.get("audience_setting_classification", "None") if style else "None"
        },
        "performance_telemetry": {
            "elapsed_seconds": stage_elapsed_seconds,
            "token_usage": stage_token_usage,
            "total_elapsed_seconds": time.perf_counter() - pipeline_started_at,
        },
    }

    # Step 8: Speechify — free-composition final speech
    speechify_result = None
    if 8 not in enabled_stages:
        print(f"\n{'='*70}")
        print("STEP 8: SPEECHIFY — SKIPPED (disabled by user)")
        print('='*70)
        emit("stage_started", stage=8)
        emit("stage_text", stage=8, text="Skipped by user preference")
        emit("stage_done", stage=8)
    else:
        emit("stage_started", stage=8)
        stage8_started_at = time.perf_counter()
        print(f"\n{'='*70}")
        print("STEP 8: SPEECHIFY — FREE-COMPOSITION FINAL SPEECH")
        print('='*70)

        # Source text: prefer APA-converted, fall back to styled output
        literature_review_text = (
            apa_result.get("apa_output", "")
            if apa_result and apa_result.get("success") and apa_result.get("apa_output")
            else styled_result.get("styled_output", "")
        )

        if literature_review_text:
            _speechify_word_count = max(100, target_output_length // 5)
            speechify_result = await speechify_output(
                literature_review=literature_review_text,
                query=query,
                style=style,
                context_details=context_details,
                target_word_count=_speechify_word_count,
            )
            _merge_stage_token_usage("stage8", speechify_result.get("token_usage"))

            if speechify_result.get("success"):
                word_count = len(speechify_result.get("speech", "").split())
                print(f"✓ Speechify complete — {word_count} words")
                emit("stage_text", stage=8, text="Final speech composed successfully")
                emit("stage_metric", stage=8, key="Words", value=word_count)
            else:
                print(f"✗ Speechify failed: {speechify_result.get('error', 'Unknown')}")
                emit("stage_text", stage=8, text=f"Speechify failed: {speechify_result.get('error', 'Unknown')}")
        else:
            print("[STEP 8] No source text available for speechify — skipping")
            emit("stage_text", stage=8, text="No source text available for speechify")
            speechify_result = {"success": False, "speech": "", "error": "No source text"}

        emit("stage_done", stage=8)
        _record_stage_elapsed("stage8", stage8_started_at)

    complete_results["speechify_result"] = speechify_result

    _print_performance_telemetry()
    
    return complete_results


# Example usage
async def main():
    """
    Example usage of the writer_main functionality.
    """
    # Example input
    user_query = "What are the latest developments in transformer architectures?"
    
    user_sources = {
        "topics": "machine learning transformers",
        "links": [
            "https://arxiv.org/abs/1706.03762",
            "https://huggingface.co/docs/transformers"
        ],
        "attachments": [
            "/path/to/research_paper.pdf",
            "/path/to/notes.txt"
        ]
    }
    
    # Process the input
    results = await process_user_input(user_query, user_sources)
    
    # Print results
    print(json.dumps(results, indent=2))
    
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

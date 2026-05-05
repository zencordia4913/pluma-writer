# deep_research/pipeline.py
import re
import os, json, asyncio
from typing import Callable, Awaitable, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import AsyncTavilyClient

# langchain_azure_ai has shipped multiple Azure chat model wrappers over time.
# Keep this import compatible across versions.
try:
    # Older versions (name used in this repo historically)
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel as AzureChatModel  # type: ignore
    _AZURE_CHAT_MODEL_FLAVOR = "azure_openai_api"
except ImportError:  # pragma: no cover
    try:
        # Current versions in this environment
        from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel as AzureChatModel  # type: ignore
        _AZURE_CHAT_MODEL_FLAVOR = "azure_ai_chat_completions"
    except ImportError:  # pragma: no cover
        # Last-resort fallback
        from langchain_azure_ai.chat_models import AzureChatOpenAI as AzureChatModel  # type: ignore
        _AZURE_CHAT_MODEL_FLAVOR = "azure_chat_openai"

# --- keep your existing helpers imports ---
from .prompts import (query_writer_instructions, summarizer_instructions,
                      reflection_instructions, get_current_date)
from .formatting import deduplicate_and_format_sources, format_sources
from .states import SummaryState, SummaryStateInput, SummaryStateOutput

# Model initialization - lazy loaded to avoid import-time connection
# Cache is loop-scoped so async clients are not reused across different event loops.
_deep_research_models: Dict[str, Any] = {}


def _current_loop_cache_scope() -> str:
    try:
        loop = asyncio.get_running_loop()
        return f"loop:{id(loop)}"
    except RuntimeError:
        return "loop:none"


def _get_max_research_loops() -> int:
    """Get maximum deep-research web loops from env, with safe defaults."""
    raw_value = os.getenv("DEEP_RESEARCH_MAX_LOOPS", "3")
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = 3
    return max(1, min(7, parsed))

def _get_model(task_tier: str = "light"):
    """
    Lazy initialization of deep-research model.
    Validates environment variables and initializes model on first use.
    
    Returns:
        Initialized Azure chat model instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    model_map = {
        "light": os.getenv("DEEP_RESEARCH_LIGHT_DEPLOYMENT") or os.getenv("AZURE_DR_DEPLOYMENT"),
        "summary": os.getenv("DEEP_RESEARCH_SUMMARY_DEPLOYMENT") or os.getenv("DEEP_RESEARCH_LIGHT_DEPLOYMENT") or os.getenv("AZURE_DR_DEPLOYMENT"),
        "default": os.getenv("AZURE_DR_DEPLOYMENT"),
    }
    _model_name = model_map.get(task_tier, model_map["light"])

    model_key = f"{_model_name}:{_current_loop_cache_scope()}"

    if model_key in _deep_research_models:
        return _deep_research_models[model_key]
    
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
            f"Missing required environment variables for deep research: {', '.join(missing)}. "
            "Please set these in your .env file to use deep research mode."
        )
    
    # Initialize model
    # Note: Azure AI Inference uses endpoint+credential; older wrappers used base_url+api_key.
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
        # AzureChatOpenAI supports both OpenAI-style and Azure-style params. We prefer base_url here
        # because this repo uses AZURE_INFERENCE_ENDPOINT (not the standard azure_endpoint).
        model_instance = AzureChatModel(
            base_url=_endpoint,
            api_key=_key,
            model=_model_name,
        )

    _deep_research_models[model_key] = model_instance
    return model_instance

# notifier: async callback(event_name:str, payload:dict) -> None
Notifier = Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]]


_CODE_BLOCK_RE = re.compile(r"(```.*?```|`.*?`)", re.DOTALL)

def _normalize_latex(text: str) -> str:
    """
    Replace \\[...\\] -> $$...$$ and \\(...\\) -> $...$
    Skips code blocks/backticks to avoid mangling code.
    """
    parts = _CODE_BLOCK_RE.split(text)
    for i in range(0, len(parts), 2):  # only non-code segments
        seg = parts[i]
        # \[ ... \]  -> $$ ... $$
        seg = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", seg, flags=re.DOTALL)
        # \( ... \)  -> $ ... $
        seg = re.sub(r"\\\((.*?)\\\)", r"$\1$", seg, flags=re.DOTALL)
        parts[i] = seg
    return "".join(parts)

def _strip_thinking_tokens(text: str):
    thoughts = ""
    while "<think>" in text and "</think>" in text:
        s = text.find("<think>"); e = text.find("</think>")
        thoughts += text[s+7:e].strip() + "\n\n"
        text = text[:s] + text[e+8:]
    return thoughts.strip(), text.strip()

async def generate_query(state: SummaryState, notify: Notifier = None):
    current_date = get_current_date()
    prompt = query_writer_instructions.format(
        current_date=current_date, research_topic=state.research_topic
    )
    messages = [SystemMessage(content=prompt), HumanMessage(content="Generate a query for web search:")]
    model = _get_model("light")
    result = await model.ainvoke(messages)
    thoughts, text = _strip_thinking_tokens(result.content)
    text = _normalize_latex(text)
    if notify: await notify("thinking", {"thoughts": thoughts})

    query = json.loads(text)
    rq = query["query"]; rationale = query["rationale"]
    if notify: await notify("generate_query", {"query": rq, "rationale": rationale, "thoughts": thoughts})
    return {"search_query": rq, "rationale": rationale}

async def web_research(state: SummaryState, notify: Notifier = None):
    import logging as _logging
    _logger = _logging.getLogger("pluma.deep_research")
    tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    try:
        # Tavily rejects queries longer than 400 characters
        search_query = state.search_query[:400] if state.search_query else state.research_topic[:400]
        res = await tavily_client.search(
            search_query, max_results=3,
            max_tokens_per_source=1000, include_raw_content=False, include_images=True
        )

        # move images into state instead of global
        images = res.get("images", []) or []
        search_str = deduplicate_and_format_sources(res, max_tokens_per_source=1000)
        if notify:
            await notify("web_research", {"sources": res.get("results", []), "images": images})

        return {
            "sources_gathered": [format_sources(res)],
            "research_loop_count": state.research_loop_count + 1,
            "web_research_results": [search_str],
            "images": (state.images or []) + images,
        }
    except Exception as tavily_exc:
        # Tavily is unreachable (403/ForbiddenError from cloud IP blocks, quota exhausted, etc.)
        # Return a graceful empty result so the graph can still run summarize/finalize
        # using whatever context was already gathered.
        _logger.warning(
            "Tavily web search failed (query='%s'): %s — continuing without web results.",
            (state.search_query or state.research_topic)[:80],
            tavily_exc,
        )
        if notify:
            await notify("web_research", {"sources": [], "images": [], "error": str(tavily_exc)})
        return {
            "sources_gathered": [],
            "research_loop_count": state.research_loop_count + 1,
            "web_research_results": [f"[Web search unavailable: {tavily_exc}]"],
            "images": state.images or [],
        }
    finally:
        # Close the client to avoid unclosed session warnings
        try:
            await tavily_client.close()
        except Exception:
            pass

async def summarize_sources(state: SummaryState, notify: Notifier = None):
    existing = state.running_summary
    new_ctx = state.web_research_results[-1]
    human = (f"<Existing Summary>\n{existing}\n</Existing Summary>\n\n"
             f"<New Context>\n{new_ctx}\n</New Context>"
             f"Update the Existing Summary with the New Context on this topic:\n<User Input>\n{state.research_topic}\n</User Input>\n\n") if existing else (
            f"<Context>\n{new_ctx}\n</Context>"
            f"Create a Summary using the Context on this topic:\n<User Input>\n{state.research_topic}\n</User Input>\n\n")

    messages = [SystemMessage(content=summarizer_instructions), HumanMessage(content=human)]
    model = _get_model("summary")
    result = await model.ainvoke(messages)
    thoughts, text = _strip_thinking_tokens(result.content)
    text = _normalize_latex(text)
    if notify: await notify("thinking", {"thoughts": thoughts})
    if notify: await notify("summarize", {"summary": text})
    return {"running_summary": text}

async def reflect_on_summary(state: SummaryState, notify: Notifier = None):
    model = _get_model("light")
    result = await model.ainvoke([
        SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Reflect on our existing knowledge:\n===\n{state.running_summary}\n===\nAnd now identify a knowledge gap and generate a follow-up web search query:")
    ])
    thoughts, text = _strip_thinking_tokens(result.content)
    text = _normalize_latex(text)
    if notify: await notify("thinking", {"thoughts": thoughts})

    try:
        content = json.loads(text)
        query = content.get("follow_up_query"); gap = content.get("knowledge_gap", "")
        if notify: await notify("reflection", {"query": query or "", "knowledge_gap": gap})
        return {"search_query": query or f"Tell me more about {state.research_topic}", "knowledge_gap": gap}
    except Exception:
        fallback = f"Tell me more about {state.research_topic}"
        if notify: await notify("reflection", {"query": fallback, "knowledge_gap": "Unable to identify specific knowledge gap"})
        return {"search_query": fallback}

async def finalize_summary(state: SummaryState, notify: Notifier = None):
    imgs = state.images or []
    if len(imgs) >= 2:
        image_section = f"""
<div class="flex flex-col md:flex-row gap-4 mb-6">
  <div class="w-full md:w-1/2"><img src="{imgs[0]}" class="w-full h-auto rounded-lg shadow-md"></div>
  <div class="w-full md:w-1/2"><img src="{imgs[1]}" class="w-full h-auto rounded-lg shadow-md"></div>
</div>"""
    elif len(imgs) == 1:
        image_section = f"""
<div class="flex justify-center mb-6">
  <div class="w-full max-w-lg"><img src="{imgs[0]}" class="w-full h-auto rounded-lg shadow-md"></div>
</div>"""
    else:
        image_section = ""

    # final = f"{image_section}## Summary\n{state.running_summary}\n\n### Sources:\n"
    # for src in state.sources_gathered or []:
    #     final += f"{src}\n"
    # if notify: await notify("finalize", {"summary": final})
    # return {"running_summary": final}

    final = f"## Summary\n{state.running_summary}\n\n### Sources:\n"
    for src in state.sources_gathered or []:
        final += f"{src}\n"

    if notify:
        await notify("finalize", {
            "summary": final,
            "images": state.images[:2]  # send top 1–2 images
        })
    return {"running_summary": final}

def setup_graph(notify: Notifier = None):        # ✅ accept notify here
    builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput)

    # ✅ async wrappers that AWAIT and close over 'notify' from outer scope
    async def gen_node(state: SummaryState):
        return await generate_query(state, notify)

    async def web_node(state: SummaryState):
        return await web_research(state, notify)

    async def sum_node(state: SummaryState):
        return await summarize_sources(state, notify)

    async def refl_node(state: SummaryState):
        return await reflect_on_summary(state, notify)

    async def fin_node(state: SummaryState):
        return await finalize_summary(state, notify)

    builder.add_node("generate_query", gen_node)
    builder.add_node("web_research", web_node)
    builder.add_node("summarize_sources", sum_node)
    builder.add_node("reflect_on_summary", refl_node)
    builder.add_node("finalize_summary", fin_node)

    builder.add_edge(START, "generate_query")
    builder.add_edge("generate_query", "web_research")
    builder.add_edge("web_research", "summarize_sources")
    builder.add_edge("summarize_sources", "reflect_on_summary")

    # ❗ no notify param here
    async def route_research(state: SummaryState):
        max_loops = _get_max_research_loops()
        if state.research_loop_count < max_loops:
            if notify:
                await notify("routing", {"decision": "continue", "loop_count": state.research_loop_count})
            return "web_research"
        if notify:
            await notify("routing", {
                "decision": "finalize",
                "loop_count": state.research_loop_count,
                "max_loops": max_loops,
            })
        return "finalize_summary"

    builder.add_conditional_edges("reflect_on_summary", route_research)
    builder.add_edge("finalize_summary", END)
    return builder.compile()

async def run_deep_research(topic: str, notify: Notifier = None, return_state: bool = False):
    """Run the deep research graph.

    Args:
        topic: Research topic string.
        notify: Optional async progress callback.
        return_state: When True, return a dict with full graph state instead of
            just the summary string.  The dict has keys:
                summary (str), sources_gathered (list[str]),
                research_loop_count (int), final_knowledge_gap (str).

    Returns:
        str when return_state=False (default), dict when return_state=True.
    """
    import logging
    _logger = logging.getLogger("pluma.deep_research")
    try:
        graph = setup_graph(notify=notify)    # ✅ pass notify into setup_graph

        result = await graph.ainvoke({"research_topic": topic, "images": []})
        summary = result.get("running_summary", "")
        if not summary or not summary.strip():
            _logger.warning("Deep research returned empty summary for topic: %s", topic[:120])
        if return_state:
            return {
                "summary": summary,
                "sources_gathered": result.get("sources_gathered", []),
                "research_loop_count": result.get("research_loop_count", 0),
                "final_knowledge_gap": result.get("knowledge_gap", ""),
            }
        return summary
    except Exception as e:
        _logger.error("Deep research pipeline failed for topic '%s': %s", topic[:120], e, exc_info=True)
        if return_state:
            return {"summary": "", "sources_gathered": [], "research_loop_count": 0, "final_knowledge_gap": ""}
        return ""


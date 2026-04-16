import streamlit as st
import app.pages as pages
import app.utils as utils
import app.prompts as prompts
import json
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from io import BytesIO

# --- NEW imports ---
from datetime import datetime
from io import BytesIO

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# --- UI helper: centered "OR" header with lines ---
def or_header(text: str):
    st.markdown(
        """
        <style>
        .or-header{display:flex;align-items:center;gap:.75rem;margin:.5rem 0 1rem;}
        .or-header:before,.or-header:after{content:"";flex:1;border-top:1px solid rgba(128,128,128,.35);}
        .or-header .or-text{font-weight:600;opacity:.85;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='or-header'><span class='or-text'>{text}</span></div>", unsafe_allow_html=True)

# App title
pages.show_home()
pages.show_sidebar()

# Home page
st.header("🔧 Speech Refiner")

# # Content input
# st.session_state.content = st.text_area(
#     ":blue[**Content Input for BSP Speech:**]", st.session_state.content, 200
# )

# uploaded_files = st.file_uploader(
#     ":blue[**Upload Content Files:**]", 
#     type=["pdf", "docx", "pptx"], 
#     accept_multiple_files=True,
#     help="Upload PDF, Word, or PowerPoint files"
# )
or_header("Input the Contents or Upload the File for BSP Speech Refining")

# --- two-column layout (Col 1 / Col 2) ---
col1, col2 = st.columns([3, 2], gap="small")

with col1:
    
    st.session_state.content = st.text_area(
        ":blue[**Input Content:**]",
        st.session_state.content,
        height=160,
        key="content_input_refiner",
    )

    st.text_area(
        ":blue[**Additional Prompt Instructions:**]",
        st.session_state.get("context_details_refiner", ""),
        height=90,
        help="Optional additional instructions to guide the speech writing process.",
        key="context_details_refiner",
    )

    MIN_WORDS, MAX_WORDS, DEFAULT_WORDS = 4, 15_000, 500

    # One source of truth (word count)
    st.session_state.setdefault("max_words_refiner", DEFAULT_WORDS)
    st.session_state.setdefault("last_updated_refiner", None)

    def _update_from_slider():
        st.session_state.last_updated_refiner = "slider"
        st.session_state.max_words_refiner = st.session_state.max_words_slider_refiner

    def _update_from_input():
        st.session_state.last_updated_refiner = "input"
        v = st.session_state.max_words_input_refiner
        # Coerce + clamp
        try:
            v = int(v)
        except Exception:
            v = MIN_WORDS
        st.session_state.max_words_refiner = max(MIN_WORDS, min(MAX_WORDS, v))

    # Keep both widgets synced to the source of truth (avoid ping-pong)
    if st.session_state.last_updated_refiner != "slider":
        st.session_state["max_words_slider_refiner"] = st.session_state.max_words_refiner
    if st.session_state.last_updated_refiner != "input":
        st.session_state["max_words_input_refiner"] = st.session_state.max_words_refiner

    col_slider, col_input = st.columns([3, 1])

    with col_slider:
        st.slider(
            f":blue[**Target Word Count:**]",
            min_value=MIN_WORDS,
            max_value=MAX_WORDS,
            key="max_words_slider_refiner",
            on_change=_update_from_slider,
            disabled=False,
        )

    with col_input:
        st.number_input(
            "No. of Words",
            min_value=MIN_WORDS,
            max_value=MAX_WORDS,
            step=1,
            key="max_words_input_refiner",
            on_change=_update_from_input,
        )

    # Convert word count to approximate character count for the pipeline (avg ~5 chars/word)
    target_output_length = st.session_state.max_words_refiner * 5

with col2:
    
    uploaded_files = st.file_uploader(
        ":blue[**Upload Files:**]",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        help="Upload PDF, Word, or PowerPoint files",
        key="content_upload_refiner",
    )

# Extract text from uploaded files
extracted_text = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"
                
        elif file_type == 'docx':
            doc = Document(BytesIO(uploaded_file.read()))
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    extracted_text += paragraph.text + "\n"
                    
        elif file_type == 'pptx':
            prs = Presentation(BytesIO(uploaded_file.read()))
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        if shape.text.strip():
                            extracted_text += shape.text + "\n"


# ----------------------------
# PICK EXACTLY ONE CONTENT SOURCE + PREVIEW
# ----------------------------
has_input = bool(st.session_state.content.strip())
has_uploads = bool(extracted_text.strip())

# Let the user choose if both are present; otherwise auto-pick
if has_input and has_uploads:
    source = st.radio(
        ":blue[**Choose content source**]",
        ["Uploaded files", "Manual input"],
        horizontal=True,
        index=0,
        key="source_refiner",
        help="Use either the text you typed OR the text extracted from the uploaded files."
    )
elif has_uploads:
    source = "Uploaded files"
elif has_input:
    source = "Manual input"
else:
    source = None

# Decide content_all and show a preview when using uploads
if source == "Uploaded files":
    content_all = extracted_text  # keep full unicode; your PDF builder handles fonts
    with st.expander("📄 Preview: Extracted text from uploaded files", expanded=True):
        st.text_area(
            "Extracted Text",
            content_all,
            height=240,
            key="uploaded_text_preview_refiner",
        )
elif source == "Manual input":
    content_all = st.session_state.content
else:
    content_all = ""
    st.markdown(
    """
    <div class="bsp-alert-red" role="alert">
      <strong>Heads up:</strong> Provide content — either type in the left box or upload a file on the right.
    </div>
    <style>
      .bsp-alert-red{
        padding:12px 14px;
        margin: 4px 0 10px;
        border-radius:10px;
        border:1px solid rgba(220,53,69,.35);
        background: rgba(220,53,69,.08); /* light red */
        font-size: 0.95rem;
      }
      .bsp-alert-red strong{
        color:#b02a37; /* dark red for emphasis */
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Combine text area and extracted content
#content_all = st.session_state.content + "\n" + extracted_text.encode("ascii", errors="ignore").decode("ascii")

# Extracting the styles and creating combined display options
styles_data = utils.get_styles()
def _to_kv_rows(data):
    if not data:
        return []
    if isinstance(data, str):
        return [{"Key": "text", "Value": data}]
    rows = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            rows.append({"Key": str(key), "Value": value})
    return rows


def _normalize_style_rules(style_rules):
    if not style_rules:
        return {}
    if isinstance(style_rules, str):
        try:
            return json.loads(style_rules)
        except Exception:
            return {"style_rules": style_rules}
    if isinstance(style_rules, dict):
        return style_rules
    return {"style_rules": str(style_rules)}


def _schema_to_values(schema):
    if not isinstance(schema, dict):
        return schema
    if "enum" in schema and isinstance(schema["enum"], list):
        return schema["enum"][0] if schema["enum"] else None
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        items = schema["items"]
        if "enum" in items and isinstance(items["enum"], list):
            return items["enum"]
        return []
    if "properties" in schema and isinstance(schema["properties"], dict):
        return {k: _schema_to_values(v) for k, v in schema["properties"].items()}
    if "items" in schema and isinstance(schema["items"], dict):
        return _schema_to_values(schema["items"])
    return schema


def _extract_style_rules(item):
    rules = item.get("style") or item.get("style_instructions")
    rules = _normalize_style_rules(rules)
    if rules:
        return rules
    # Check nested properties payload
    props = item.get("properties") or {}
    if isinstance(props, str):
        try:
            props = json.loads(props)
        except Exception:
            props = {}
    if isinstance(props, dict):
        nested = props.get("style_instructions") or props.get("style")
        if isinstance(nested, dict):
            nested_values = _schema_to_values(nested)
            if nested_values:
                return nested_values
        nested = _normalize_style_rules(nested)
        if nested:
            return nested
    # Fallback: build from known top-level fields
    fallback_keys = [
        "register_profile",
        "stylistics",
        "pragmatics",
        "semantics_frames",
        "discourse_structure",
        "signature_moves",
        "evidence_spans",
    ]
    compiled = {}
    for key in fallback_keys:
        if key in item and item[key]:
            compiled[key] = item[key]
    if not compiled and isinstance(props, dict):
        for key in fallback_keys:
            if key in props and props[key]:
                compiled[key] = _schema_to_values(props[key])
    return compiled


def _render_rule_section(title, value):
    if value is None or value == "":
        st.caption("No data.")
        return
    if isinstance(value, list):
        items = [str(v) for v in value if v is not None and str(v).strip() != ""]
        if items:
            st.markdown(
                """
                <style>
                .rule-list{margin:0.25rem 0 0.5rem 1.1rem;}
                .rule-list li{margin:0.2rem 0;}
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                "<ul class='rule-list'>" + "".join([f"<li>{i}</li>" for i in items]) + "</ul>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No data.")
        return
    if isinstance(value, dict):
        rows = []
        complex_keys = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                complex_keys.append((k, v))
            else:
                rows.append({"Key": str(k), "Value": v})
        if rows:
            st.markdown(
                """
                <style>
                .rule-kv{display:flex;flex-direction:column;gap:0.35rem;margin:0.35rem 0 0.5rem;}
                .rule-kv .rule-item{display:flex;flex-wrap:wrap;gap:0.35rem;}
                .rule-kv .rule-key{font-weight:600;}
                </style>
                """,
                unsafe_allow_html=True,
            )
            items_html = "".join(
                [
                    "<div class='rule-item'><span class='rule-key'>"
                    + str(r["Key"])
                    + ":</span><span class='rule-val'>"
                    + str(r["Value"])
                    + "</span></div>"
                    for r in rows
                ]
            )
            st.markdown(f"<div class='rule-kv'>{items_html}</div>", unsafe_allow_html=True)
        for k, v in complex_keys:
            with st.expander(str(k), expanded=False):
                _render_rule_section(str(k), v)
        return
    st.markdown(
        """
        <style>
        .rule-single{margin:0.35rem 0 0.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='rule-single'>{value}</div>", unsafe_allow_html=True)


def _format_section_title(text: str) -> str:
    return str(text).replace("_", " ").replace("-", " ").strip()


def _filter_global_rulebook(rulebook: dict) -> dict:
    if not isinstance(rulebook, dict):
        return {}
    excluded = {
        "id",
        "_rid",
        "_self",
        "_etag",
        "_attachments",
        "_ts",
        "PartitionKey",
        "doc_kind",
        "scope",
        "version",
        "created_at",
        "coverage_metrics",
        "confidence",
        "evidence_spans",
        "speech_ids",
        "speech_count",
    }
    return {k: v for k, v in rulebook.items() if k not in excluded}


def _coerce_global_rulebook():
    rulebook = st.session_state.get("global_rulebook")
    if rulebook:
        return rulebook
    global_rules = st.session_state.get("global_rules")
    if isinstance(global_rules, dict):
        return global_rules
    if isinstance(global_rules, str) and global_rules.strip():
        try:
            return json.loads(global_rules)
        except Exception:
            return {"global_rules": global_rules}
    return {}


def _split_style_name(name):
    if "_" in name:
        speaker, audience = name.split("_", 1)
        return speaker.strip(), audience.strip()
    return name.strip(), "General"

speakers = []
audience_by_speaker = {}
for item in styles_data:
    name = item.get("name") or item.get("container_key") or item.get("id")
    if not name:
        speaker = item.get("speaker")
        audience = item.get("audience_setting_classification")
        if speaker and audience:
            name = f"{speaker}_{audience}"
        else:
            continue
    speaker, audience = _split_style_name(str(name))
    if speaker not in audience_by_speaker:
        audience_by_speaker[speaker] = []
        speakers.append(speaker)
    if audience not in audience_by_speaker[speaker]:
        audience_by_speaker[speaker].append(audience)

selected_speaker = st.selectbox(":blue[**Select Speaker:**]", options=speakers, index=None, key="speaker_refiner")
selected_audience = st.selectbox(
    ":blue[**Select Audience/Setting:**]",
    options=audience_by_speaker.get(selected_speaker, []) if selected_speaker else [],
    index=None,
    key="audience_refiner",
)

if not speakers:
    st.warning("No speakers available. Check the styles container data.")

# Assigning the selected style to the session state
if selected_speaker and selected_audience:
    selected_style_name = f"{selected_speaker}_{selected_audience}"
    filtered = utils.get_style_fingerprint(selected_speaker, selected_audience)
    if not filtered:
        filtered = next(
            (
                item for item in styles_data
                if str(item.get("name") or item.get("container_key") or item.get("id") or "")
                == selected_style_name
                or (
                    item.get("speaker") == selected_speaker
                    and item.get("audience_setting_classification") == selected_audience
                )
            ),
            None
        )

    st.session_state.selected_style_doc = filtered or {}

    if filtered:
        style_rules = _extract_style_rules(filtered)
        st.session_state.style_rules = style_rules
        st.session_state.style = json.dumps(style_rules, ensure_ascii=False, indent=2) if style_rules else ""
        example = filtered.get("example")
        if not example:
            evidence = filtered.get("evidence_spans") or []
            example = "\n".join([e for e in evidence if e]) if evidence else ""
        st.session_state.example = example
        st.session_state.styleId = selected_style_name
        rulebook_id = filtered.get("global_rulebook_id")
        rulebook = utils.get_rulebook(rulebook_id)
        st.session_state.global_rulebook = rulebook or {}
        st.session_state.global_rules = utils.extract_rulebook_text(rulebook)

show_loaded_rules = False

if show_loaded_rules:
    with st.container(border=True):
        st.subheader("Loaded Style & Global Rules")
        if not (selected_speaker and selected_audience):
            st.info("Select a speaker and audience/setting to view the loaded rules.")
        else:
            selected_doc = st.session_state.get("selected_style_doc") or {}
            style_source = st.session_state.get("style_rules") or _extract_style_rules(selected_doc) or st.session_state.get("style") or {}
            global_rulebook = _coerce_global_rulebook()

            st.markdown("**Style Rules**")
            if style_source:
                if isinstance(style_source, dict):
                    rule_items = list(style_source.items())
                    mid_point = (len(rule_items) + 1) // 2
                    col1, col2 = st.columns(2)
                    with col1:
                        for section, value in rule_items[:mid_point]:
                            section_title = _format_section_title(section)
                            with st.expander(section_title, expanded=False):
                                _render_rule_section(section_title, value)
                    with col2:
                        for section, value in rule_items[mid_point:]:
                            section_title = _format_section_title(section)
                            with st.expander(section_title, expanded=False):
                                _render_rule_section(section_title, value)
                else:
                    _render_rule_section("Style Rules", style_source)
            else:
                st.caption("No style rules loaded.")

            st.markdown("**Global Rules**")
            if global_rulebook:
                if isinstance(global_rulebook, dict):
                    filtered_rulebook = _filter_global_rulebook(global_rulebook)
                    rule_items = list(filtered_rulebook.items())
                    if not rule_items:
                        st.caption("No global rules loaded.")
                        rule_items = []
                    mid_point = (len(rule_items) + 1) // 2
                    col1, col2 = st.columns(2)
                    with col1:
                        for section, value in rule_items[:mid_point]:
                            section_title = _format_section_title(section)
                            with st.expander(section_title, expanded=False):
                                _render_rule_section(section_title, value)
                    with col2:
                        for section, value in rule_items[mid_point:]:
                            section_title = _format_section_title(section)
                            with st.expander(section_title, expanded=False):
                                _render_rule_section(section_title, value)
                else:
                    _render_rule_section("Global Rules", global_rulebook)
            else:
                st.caption("No global rules loaded.")

            with st.expander("Style Debug", expanded=False):
                st.write({
                    "selected_speaker": selected_speaker,
                    "selected_audience": selected_audience,
                    "style_doc_loaded": bool(selected_doc),
                    "style_doc_keys": list(selected_doc.keys())[:25],
                    "has_style_instructions": "style_instructions" in selected_doc,
                    "has_register_profile": "register_profile" in selected_doc,
                    "style_rules_keys": list(style_source.keys())[:25] if isinstance(style_source, dict) else [],
                })
        
# st.session_state.style = st.text_area(":blue[**Style:**]", st.session_state.style)

# Show the example style
guidelines = st.session_state.locals.get("relevant_guidelines", {})
guidelines_summary = st.session_state.locals.get("guideline_summaries", {}) 
selected_guidelines = []
selected_guideline_names = []

# All editorial style guidelines are selected automatically (UI hidden)
selected_guidelines = []
selected_guideline_names = []

if guidelines:
    for section_name, content in guidelines.items():
        selected_guidelines.append(content)
        selected_guideline_names.append(section_name)
else:
    pass  # No guidelines available

# Join all selected guidelines with newlines and store in session state
st.session_state.guidelines = "\n".join(selected_guidelines)

# Show the combined guidelines in a text area
# st.text_area(":blue[**Relevant Guidelines:**]", st.session_state.guidelines, height=200)


# --- NEW helpers: make DOCX/PDF from text ---

def make_docx_bytes(text: str, title: str | None = None) -> bytes:
    """Return a .docx file (bytes) with a title and body paragraphs."""
    doc = Document()
    if title:
        doc.add_heading(title, level=1)
    # Split into paragraphs on blank lines while preserving line breaks
    for block in text.replace("\r\n", "\n").split("\n\n"):
        p = doc.add_paragraph()
        for line in block.split("\n"):
            if line.strip():
                p.add_run(line)
            p.add_run("\n")
    bio = BytesIO()
    doc.save(bio)
    return bio.getvalue()


def _register_pdf_font_if_available():
    """Optionally register DejaVuSans for better Unicode PDF rendering."""
    try:
        font_path = os.path.join("assets", "DejaVuSans.ttf")
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
            return "DejaVuSans"
    except Exception:
        pass
    # Fallback to built-in Helvetica (ASCII/Latin-1 safe)
    return "Helvetica"


def make_pdf_bytes(text: str, title: str | None = None) -> bytes:
    """Return a PDF (bytes) using ReportLab."""
    font_name = _register_pdf_font_if_available()

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=title or "Rewrite",
        author="Speech Writer",
    )

    styles = getSampleStyleSheet()
    base = styles["BodyText"]
    base.fontName = font_name
    base.fontSize = 11
    base.leading = 14

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontName=font_name,
        spaceAfter=12,
    )

    story = []
    if title:
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 8))

    # Turn double newlines into paragraph breaks; single newlines stay inside a paragraph
    for block in text.replace("\r\n", "\n").split("\n\n"):
        # Escape simple HTML-sensitive chars for Paragraph
        block = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        block = block.replace("\n", "<br/>")
        story.append(Paragraph(block, base))
        story.append(Spacer(1, 6))

    doc.build(story)
    return buf.getvalue()


if st.button(
    ":blue[**Rewrite Content**]",
    key="extract_refiner",
   # disabled=content_all == ""
    disabled=(
    content_all.strip() == ""
    or st.session_state.style == ""
)
    or st.session_state.style == "",
):
    with st.container(border=True):
        with st.spinner("Processing..."):
            context_details = st.session_state.get("context_details_refiner", "").strip()
            output = prompts.rewrite_content(
                content_all,
                target_output_length,
                False,
                context_details,
            )
            utils.save_output(output, content_all)

            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            style_id = (st.session_state.get("styleId") or "Style").replace(" ", "_")
            base_name = f"rewrite_{style_id}_{ts}"
            title_text = f"Rewrite • {st.session_state.get('styleId') or 'Selected Style'}"
            docx_bytes = make_docx_bytes(output, title=title_text)
            pdf_bytes = make_pdf_bytes(output, title=title_text)

            # Upload to Azure Blob Storage
            _pdf_url = utils.upload_bytes_to_blob(
                pdf_bytes,
                f"refiner/{base_name}.pdf",
                content_type="application/pdf",
            )
            _docx_url = utils.upload_bytes_to_blob(
                docx_bytes,
                f"refiner/{base_name}.docx",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

            # Save to Cosmos DB with blob URLs
            utils.save_style_refiner_output(
                output, content_all,
                pdf_url=_pdf_url, docx_url=_docx_url,
                additional_instructions=context_details,
                editorial_style_guides=selected_guideline_names,
                target_word_count=st.session_state.get("max_words_refiner", 0),
            )

            # Cache in session state for persistent display
            st.session_state["refiner_output_cache"] = {
                "text": output,
                "docx_bytes": docx_bytes,
                "pdf_bytes": pdf_bytes,
                "base_name": base_name,
            }
            st.rerun()

# --- Persistent output display (survives download-button reruns) ---
if "refiner_output_cache" in st.session_state:
    _rc = st.session_state["refiner_output_cache"]
    with st.container(border=True):
        st.markdown("### ✨ Rewritten Output")
        st.text_area("Result", _rc["text"], height=420, key="refiner_output_display")

        _c1, _c2 = st.columns(2)
        with _c1:
            st.download_button(
                "⬇️ Download as DOCX",
                data=_rc["docx_bytes"],
                file_name=f"{_rc['base_name']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="dl_refiner_docx",
            )
        with _c2:
            st.download_button(
                "⬇️ Download as PDF",
                data=_rc["pdf_bytes"],
                file_name=f"{_rc['base_name']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_refiner_pdf",
            )

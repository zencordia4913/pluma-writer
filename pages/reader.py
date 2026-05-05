import json
import streamlit as st
import app.pages as pages
import app.utils as utils


def _split_style_name(name: str):
    if "_" in name:
        speaker, audience = name.split("_", 1)
        return speaker.strip(), audience.strip()
    return name.strip(), "General"


def _format_section_title(text: str) -> str:
    return str(text).replace("_", " ").replace("-", " ").strip()


def _value_to_editor_text(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    if isinstance(value, list):
        return "\n".join([str(v) for v in value])
    return "" if value is None else str(value)


def _parse_editor_text(value, original_type):
    if original_type == "list":
        return [v.strip() for v in value.splitlines() if v.strip()]
    if original_type == "dict":
        return json.loads(value) if value.strip() else {}
    return value


def _unwrap_schema_properties(props: dict):
    if not isinstance(props, dict):
        return props
    if "properties" in props and isinstance(props["properties"], dict):
        if props.get("type") == "object" or "required" in props:
            return props["properties"]
    return props


def _find_nested_schema(props: dict, key: str):
    if not isinstance(props, dict):
        return None
    if key in props and isinstance(props[key], dict):
        return props[key]
    for value in props.values():
        if isinstance(value, dict):
            found = _find_nested_schema(value, key)
            if found:
                return found
    return None


def _schema_debug_info(doc: dict) -> dict:
    if not isinstance(doc, dict):
        return {"doc": "invalid"}
    props = doc.get("properties")
    props_type = type(props).__name__
    if isinstance(props, str):
        try:
            props = json.loads(props)
        except Exception:
            props = None
    info = {
        "doc_keys": list(doc.keys())[:30],
        "properties_type": props_type,
    }
    if isinstance(props, dict):
        info["properties_keys"] = list(props.keys())[:30]
        nested = props.get("properties") if isinstance(props.get("properties"), dict) else {}
        info["nested_properties_keys"] = list(nested.keys())[:30]
        info["has_style_instructions"] = "style_instructions" in props or "style_instructions" in nested
        info["has_style"] = "style" in props or "style" in nested
        info["has_style_sections"] = any(
            k in props
            for k in [
                "register_profile",
                "stylistics",
                "pragmatics",
                "semantics_frames",
                "discourse_structure",
                "signature_moves",
                "evidence_spans",
            ]
        )
    return info


def _coerce_style_schema(doc: dict) -> dict:
    if not doc:
        return {}
    props = doc.get("properties") or {}
    if isinstance(props, str):
        try:
            props = json.loads(props)
        except Exception:
            props = {}
    props = _unwrap_schema_properties(props)
    if isinstance(props, dict):
        nested = props.get("style_instructions") or props.get("style")
        if isinstance(nested, dict):
            return nested
        nested = _find_nested_schema(props, "style_instructions") or _find_nested_schema(props, "style")
        if isinstance(nested, dict):
            return nested
        # Some schemas store editable sections directly under properties
        if props:
            style_keys = {
                "register_profile",
                "stylistics",
                "pragmatics",
                "semantics_frames",
                "discourse_structure",
                "signature_moves",
                "evidence_spans",
                "style_instructions",
                "style",
            }
            filtered = {k: v for k, v in props.items() if k in style_keys}
            return {"properties": filtered or props}
    candidate = doc.get("style_instructions") or doc.get("style")
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except Exception:
            candidate = {}
    if isinstance(candidate, dict):
        return candidate
    return {}


def _schema_node_type(node: dict) -> str:
    if not isinstance(node, dict):
        return "string"
    if node.get("type"):
        return node.get("type")
    if "properties" in node:
        return "object"
    if "items" in node:
        return "array"
    return "string"


def _schema_value_from_enum(node: dict):
    node_type = _schema_node_type(node)
    if node_type == "array":
        items = node.get("items") or {}
        return list(items.get("enum") or [])
    if node_type == "object":
        return {}
    return (node.get("enum") or [""])[0]


def _state_key(path: list[str]) -> str:
    return "edit_schema_" + "__".join(path)


def _get_value_from_data(data, path: list[str]):
    if not data or not isinstance(data, dict):
        return None
    if path:
        path = path[1:]
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def _editor_default_value(node_type: str, value) -> str:
    if value is None:
        return ""
    if node_type == "array":
        if isinstance(value, list):
            return "\n".join([str(v) for v in value])
        return _value_to_editor_text(value)
    if isinstance(value, (dict, list)):
        return _value_to_editor_text(value)
    return str(value)


def _render_schema_fields(node: dict, path: list[str], data_root=None):
    node_type = _schema_node_type(node)
    if node_type == "object":
        for key, child in (node.get("properties") or {}).items():
            child_type = _schema_node_type(child)
            label = _format_section_title(key)
            if child_type == "object":
                st.markdown(f"**{label}**")
                _render_schema_fields(child, path + [key], data_root)
            else:
                _render_schema_fields(child, path + [key], data_root)
        return

    state_key = _state_key(path)
    if state_key not in st.session_state:
        value = _schema_value_from_enum(node)
        data_value = _get_value_from_data(data_root, path)
        if data_value is not None:
            value = data_value
        st.session_state[state_key] = _editor_default_value(node_type, value)

    label = _format_section_title(path[-1])
    if node_type == "array":
        st.text_area(label, key=state_key, height=140, help="One item per line")
    elif node_type in ["integer", "number"]:
        try:
            current_val = float(st.session_state[state_key])
        except Exception:
            current_val = 0.0
        st.number_input(label, value=current_val, step=1.0 if node_type == "integer" else 0.1)
    else:
        st.text_area(label, key=state_key, height=90)


def _update_schema_from_state(node: dict, path: list[str]):
    node_type = _schema_node_type(node)
    if node_type == "object":
        for key, child in (node.get("properties") or {}).items():
            _update_schema_from_state(child, path + [key])
        return

    state_key = _state_key(path)
    raw_value = st.session_state.get(state_key, "")
    if node_type == "array":
        items = node.get("items") or {}
        items["enum"] = [v.strip() for v in str(raw_value).splitlines() if v.strip()]
        node["items"] = items
    elif node_type == "integer":
        try:
            node["enum"] = [int(float(raw_value))]
        except Exception:
            node["enum"] = []
    elif node_type == "number":
        try:
            node["enum"] = [float(raw_value)]
        except Exception:
            node["enum"] = []
    else:
        node["enum"] = [str(raw_value)] if str(raw_value).strip() != "" else []


def _coerce_rulebook_sections(rulebook: dict) -> dict:
    if not rulebook:
        return {}
    for key in ["global_rules", "rules", "rulebook", "rulebook_text"]:
        if key in rulebook and rulebook[key]:
            candidate = rulebook[key]
            if isinstance(candidate, str):
                try:
                    candidate = json.loads(candidate)
                except Exception:
                    return {key: candidate}
            if isinstance(candidate, dict):
                return candidate
            return {key: candidate}
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
    }
    return {k: v for k, v in rulebook.items() if k not in excluded}


def _rulebook_value_type(value) -> str:
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    return "text"


def _rulebook_value_to_text(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    if isinstance(value, list):
        return "\n".join([str(v) for v in value])
    return "" if value is None else str(value)


def _rulebook_parse_text(value: str, value_type: str):
    if value_type == "list":
        return [v.strip() for v in value.splitlines() if v.strip()]
    if value_type == "dict":
        return json.loads(value) if value.strip() else {}
    return value


def _extract_rulebook_id(doc: dict):
    if not doc:
        return None
    if doc.get("global_rulebook_id"):
        return doc.get("global_rulebook_id")
    props = doc.get("properties") or {}
    if isinstance(props, dict) and props.get("global_rulebook_id"):
        gr = props.get("global_rulebook_id")
        if isinstance(gr, dict):
            enum_vals = gr.get("enum") or []
            return enum_vals[0] if enum_vals else None
        return gr
    return None


pages.show_home()
pages.show_sidebar()

st.header("✏️Style Rules")

st.markdown(
    """
    <style>
    .stButton{width:100%;}
    .stButton > button{
        width:100%;
        border-radius:12px;
        padding:0.95rem 1.1rem;
        text-align:center;
        min-height:56px;
        height:56px;
        display:flex;
        align-items:center;
        justify-content:center;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
        border:1px solid rgba(0,0,0,.12);
        background:linear-gradient(180deg,#ffffff 0%,#f7f8fb 100%);
        box-shadow:0 6px 14px rgba(15,23,42,.08), 0 1px 2px rgba(15,23,42,.06);
        font-weight:600;
        transition:transform .12s ease, box-shadow .12s ease, border-color .12s ease;
    }
    .stButton > button:hover{
        border-color:rgba(30,64,175,.35);
        box-shadow:0 10px 20px rgba(15,23,42,.12), 0 2px 6px rgba(15,23,42,.08);
        transform:translateY(-1px);
    }
    .stButton > button:active{
        transform:translateY(0);
        box-shadow:0 4px 10px rgba(15,23,42,.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

styles = utils.get_styles()
if not styles:
    st.info("No styles found.")
    st.stop()

speaker_map = {}
global_rulebooks = {}
for item in utils.get_rulebooks():
    rulebook_id = item.get("id") or item.get("container_key")
    if rulebook_id:
        global_rulebooks[rulebook_id] = item
for item in styles:
    if item.get("doc_kind") and item.get("doc_kind") != "style_fingerprint":
        continue
    name = item.get("name") or item.get("container_key") or item.get("id") or ""
    speaker = item.get("speaker")
    audience = item.get("audience_setting_classification")
    if not speaker or not audience:
        speaker, audience = _split_style_name(str(name))
    speaker_map.setdefault(speaker, {})[audience] = item

st.session_state.setdefault("editor_selected_speaker", None)
st.session_state.setdefault("editor_selected_audience", None)
st.session_state.setdefault("selected_global_rulebook_id", None)

st.subheader("Select Speaker")
speaker_cols = st.columns(2)
for idx, speaker in enumerate(sorted(speaker_map.keys())):
    with speaker_cols[idx % 2]:
        is_selected = st.session_state.editor_selected_speaker == speaker
        with st.expander(speaker, expanded=is_selected):
            audiences = sorted(speaker_map[speaker].keys())
            audience_cols = st.columns(2)
            for a_idx, audience in enumerate(audiences):
                with audience_cols[a_idx % 2]:
                    if st.button(audience, key=f"aud_{speaker}_{audience}", use_container_width=True):
                        st.session_state.editor_selected_speaker = speaker
                        st.session_state.editor_selected_audience = audience

selected_speaker = st.session_state.editor_selected_speaker
selected_audience = st.session_state.editor_selected_audience
if selected_speaker and selected_audience:
    selected_doc = utils.get_style_fingerprint(selected_speaker, selected_audience) or speaker_map[selected_speaker][selected_audience]
    doc_id = selected_doc.get("id") or selected_doc.get("container_key") or selected_doc.get("name")

    header_col, spacer_col, action_col = st.columns([3, 0.4, 0.9])
    with header_col:
        audience_label = _format_section_title(selected_audience)
        st.subheader("Style Settings")
        st.markdown(
            f"<div style='color:#6b7280;font-size:0.9rem;margin-top:-0.5rem;margin-bottom:0.75rem;'>"
            f"{selected_speaker} / {audience_label}"
            "</div>",
            unsafe_allow_html=True,
        )
    with spacer_col:
        st.write("")
    with action_col:
        st.write("")
    style_schema = _coerce_style_schema(selected_doc)
    if not isinstance(style_schema, dict) or not style_schema:
        st.warning("This style has no editable style schema.")
        with st.expander("Schema Debug", expanded=False):
            st.write(_schema_debug_info(selected_doc))
        st.stop()

    schema_properties = style_schema.get("properties") if isinstance(style_schema, dict) else {}
    has_schema_sections = isinstance(schema_properties, dict) and bool(schema_properties)

    value_sections = {}
    value_section_types = {}

    if has_schema_sections:
        for section, section_schema in schema_properties.items():
            section_title = _format_section_title(section)
            with st.expander(section_title, expanded=False):
                data_root = selected_doc.get("style_instructions") or selected_doc.get("style") or {}
                _render_schema_fields(section_schema, [doc_id, section], data_root)
    elif isinstance(style_schema, dict) and style_schema:
        value_sections = style_schema
        for section, value in value_sections.items():
            if isinstance(value, dict):
                value_section_types[section] = "dict"
            elif isinstance(value, list):
                value_section_types[section] = "list"
            else:
                value_section_types[section] = "text"

            section_key = f"edit_{doc_id}_{section}"
            if section_key not in st.session_state:
                st.session_state[section_key] = _value_to_editor_text(value)

            section_title = _format_section_title(section)
            with st.expander(section_title, expanded=False):
                st.text_area(
                    section_title,
                    key=section_key,
                    height=160 if value_section_types[section] != "text" else 120,
                    help="Use one item per line for lists; use JSON for objects.",
                )
    else:
        st.warning("This style has no editable sections.")
        st.stop()


    if st.button("💾 Save Style Rules", key="save_style_btn", type="primary"):
        new_doc = dict(selected_doc)
        if has_schema_sections:
            new_schema = json.loads(json.dumps(style_schema))
            for section, section_schema in (new_schema.get("properties") or {}).items():
                _update_schema_from_state(section_schema, [doc_id, section])

            if "properties" in new_doc and isinstance(new_doc["properties"], dict):
                if "style_instructions" in new_doc["properties"]:
                    new_doc["properties"]["style_instructions"] = new_schema
                elif "style" in new_doc["properties"]:
                    new_doc["properties"]["style"] = new_schema
            else:
                new_doc["style_instructions"] = new_schema
        else:
            updated = {}
            for section, section_type in value_section_types.items():
                section_key = f"edit_{doc_id}_{section}"
                raw_value = st.session_state.get(section_key, "")
                updated[section] = _parse_editor_text(raw_value, section_type)
            new_doc["style_instructions"] = updated

        style_saved = utils.save_style_fingerprint(new_doc)

        if style_saved:
            st.success("Style updated and saved.")

# --- Global Settings (always visible) ---
selected_rulebook_id = st.session_state.selected_global_rulebook_id
if not selected_rulebook_id and selected_speaker and selected_audience:
    selected_rulebook_id = _extract_rulebook_id(selected_doc)

available_rulebook_ids = sorted(global_rulebooks.keys())
if available_rulebook_ids:
    if selected_rulebook_id not in available_rulebook_ids:
        selected_rulebook_id = available_rulebook_ids[0]
    selected_rulebook_id = st.selectbox(
        "Global Rulebook",
        options=available_rulebook_ids,
        index=available_rulebook_ids.index(selected_rulebook_id),
    )
    st.session_state.selected_global_rulebook_id = selected_rulebook_id

    rulebook = utils.get_rulebook(selected_rulebook_id)
    rulebook_sections = _coerce_rulebook_sections(rulebook)

    header_col, spacer_col, action_col = st.columns([3, 0.4, 0.9])
    with header_col:
        st.subheader("Global Settings")
        st.caption(f"Rulebook: {selected_rulebook_id}")
    with spacer_col:
        st.write("")
    with action_col:
        pass

    if rulebook_sections:
        for section, value in rulebook_sections.items():
            value_type = _rulebook_value_type(value)
            state_key = f"global_{selected_rulebook_id}_{section}"
            if state_key not in st.session_state:
                st.session_state[state_key] = _rulebook_value_to_text(value)
            section_title = _format_section_title(section)
            with st.expander(section_title, expanded=False):
                st.text_area(
                    section_title,
                    key=state_key,
                    height=160 if value_type != "text" else 120,
                    help="Use one item per line for lists; use JSON for objects.",
                )
    else:
        st.info("No global settings found for this rulebook.")

    if rulebook and st.button("💾 Save Global Settings", key="save_rulebook_btn", type="primary"):
        updated_sections = {}
        rulebook_saved = True
        for section, value in rulebook_sections.items():
            state_key = f"global_{selected_rulebook_id}_{section}"
            raw_value = st.session_state.get(state_key, "")
            try:
                updated_sections[section] = _rulebook_parse_text(
                    raw_value, _rulebook_value_type(value)
                )
            except Exception as exc:
                st.error(f"Global Settings: {section} - {exc}")
                rulebook_saved = False

        if rulebook_saved:
            new_rulebook = dict(rulebook)
            target_key = None
            for key in ["global_rules", "rules", "rulebook", "rulebook_text"]:
                if key in new_rulebook:
                    target_key = key
                    break
            if not target_key:
                target_key = "global_rules"
            new_rulebook[target_key] = updated_sections
            if utils.save_style_fingerprint(new_rulebook):
                st.success("Global settings saved.")
else:
    st.subheader("Global Settings")
    st.info("No global rulebooks found.")

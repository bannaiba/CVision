"""
modules/chatbot.py
==================
LLM-powered HR Assistant chatbot using Google Gemini API (new google-genai SDK).

Responsibilities
----------------
1. Build a rich system prompt from pipeline results (rankings, metrics, raw CVs).
2. Maintain conversational memory within a Streamlit session.
3. Stream responses from the Gemini API for a responsive chat experience.
4. Execute tools for retrieving details, comparing candidates, sending emails, and exporting.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Gemini Client Setup ──────────────────────────────────────────────────────

_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai not installed. "
            "Run: pip install google-genai"
        )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set in .env file. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    _gemini_client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized (google-genai SDK).")
    return _gemini_client

# ── Agent Tools ───────────────────────────────────────────────────────────────

def build_agent_tools(candidates: list, filtered: list, results_df: pd.DataFrame):
    """
    Builds the tools for Gemini to call, bound to the current session data.
    """
    lookup = {c.name: c for c in candidates}
    
    def get_all_candidates_summary() -> dict:
        """Return a summary of all candidates (name, email, cgpa, years_experience, degree). Use this when asked questions about 'who has the most X' or comparing everyone."""
        summary = []
        for c in candidates:
            summary.append({
                "name": c.name,
                "cgpa": c.cgpa if c.cgpa != -1.0 else None,
                "years_experience": c.years_exp if c.years_exp != -1.0 else None,
                "degree": c.degree
            })
        return {"candidates": summary}

    def get_candidate_details(name: str) -> dict:
        """Return full profile details and resume excerpt for one candidate by exact name."""
        record = lookup.get(name)
        if record is None:
            return {"error": f"No candidate found with name '{name}'."}
        
        fit_score = "N/A"
        try:
            row = results_df[results_df["Candidate Name"] == name]
            if not row.empty:
                val = row.iloc[0].get("Fit Score (%)", "N/A")
                if hasattr(val, "item"): val = val.item()
                fit_score = val
        except:
            pass

        def _safe_float(v):
            if v == -1.0 or v is None: return None
            return float(v)

        return {
            "name": record.name,
            "email": record.email,
            "cgpa": _safe_float(record.cgpa),
            "degree": record.degree,
            "years_experience": _safe_float(record.years_exp),
            "fit_score": fit_score,
            "resume_excerpt": record.resume_markdown[:2000],
        }

    def compare_candidates(names: list[str]) -> dict:
        """Return a side-by-side comparison of two or more named candidates."""
        found, missing = [], []
        def _safe_float(v):
            if v == -1.0 or v is None: return None
            return float(v)
            
        for n in names:
            r = lookup.get(n)
            if not r:
                missing.append(n)
            else:
                fit_score = "N/A"
                try:
                    row = results_df[results_df["Candidate Name"] == n]
                    if not row.empty:
                        val = row.iloc[0].get("Fit Score (%)", "N/A")
                        if hasattr(val, "item"): val = val.item()
                        fit_score = val
                except:
                    pass
                found.append({
                    "name": r.name, 
                    "cgpa": _safe_float(r.cgpa), 
                    "fit_score": fit_score,
                    "years_experience": _safe_float(r.years_exp), 
                    "degree": r.degree,
                    "resume_excerpt": r.resume_markdown[:2000]
                })
        return {"candidates": found, "not_found": missing}

    def search_by_skill(skill: str) -> dict:
        """Return all candidates whose resume mentions the given skill or keyword."""
        matches = [c.name for c in candidates if skill.lower() in c.resume_markdown.lower()]
        return {"skill": skill, "matches": matches, "count": len(matches)}

    def get_filtered_candidates() -> dict:
        """Return candidates who were rejected by the knockout filters, with reasons."""
        return {"filtered": [{"name": c.name, "reason": c.filter_reason} for c in filtered]}
        
    def send_decision_emails(accepted_names: list[str], rejected_names: list[str], user_confirmed: bool = False) -> dict:
        """
        Sends acceptance/rejection emails to specified candidates. 
        Requires user_confirmed=True to actually send.
        """
        if not user_confirmed:
            return {
                "status": "CONFIRMATION_REQUIRED",
                "message": "You must ask the user to confirm they want to send these emails. If they say yes, call this tool again with user_confirmed=True.",
                "staged_accepted": accepted_names,
                "staged_rejected": rejected_names
            }
            
        from modules.email_dispatch import send_email, ACCEPTANCE_TEMPLATE, REJECTION_FINAL_TEMPLATE
        
        acc_objs = [lookup[n] for n in accepted_names if n in lookup]
        rej_objs = [lookup[n] for n in rejected_names if n in lookup]
        
        sent_acc, sent_rej = 0, 0
        
        for c in acc_objs:
            if c.email:
                body = ACCEPTANCE_TEMPLATE.format(name=c.name, position=c.position_applied or "the open position", company_name="CVision")
                send_email(c.email, "Interview Invitation", body, dry_run=False)
                sent_acc += 1
                
        for c in rej_objs:
            if c.email:
                body = REJECTION_FINAL_TEMPLATE.format(name=c.name, position=c.position_applied or "the open position", company_name="CVision")
                send_email(c.email, "Application Update", body, dry_run=False)
                sent_rej += 1
        
        return {
            "status": "SUCCESS",
            "message": f"Successfully sent {sent_acc} acceptance emails and {sent_rej} rejection emails."
        }

    def export_results_to_google_sheet() -> dict:
        """
        Exports the current analysis results to the configured Google Sheet.
        """

        sheet_id = os.getenv("GOOGLE_SHEET_ID")
        creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
        if not sheet_id:
            return {"error": "GOOGLE_SHEET_ID not found in environment."}
            
        from modules.ingestion import export_results_to_sheet
        try:
            export_results_to_sheet(
                sheet_id=sheet_id,
                credentials_path=creds_path,
                results_df=results_df,
                candidates=candidates,
                selected_names=[],
                email_status={},
                tab_prefix="CVision Database"
            )
            return {"status": "SUCCESS", "message": "Successfully exported results to Google Sheet."}
        except Exception as e:
            return {"error": str(e)}

    return [get_all_candidates_summary, get_candidate_details, compare_candidates, search_by_skill, get_filtered_candidates, send_decision_emails, export_results_to_google_sheet]

# ── System Prompt Builder ─────────────────────────────────────────────────────

def build_system_prompt(results_df: pd.DataFrame, job_description: str) -> str:
    """Build a compact system prompt containing only rankings and core metrics."""
    prompt_parts = [
        "# ROLE: Expert HR Screening Assistant for CVision",
        "",
        "You are an AI HR Assistant embedded in the CVision resume screening platform.",
        "Your job is to help the recruiter analyze, compare, and make decisions about candidates.",
        "",
        "## Tool Calling Guidelines:",
        "- You are PROACTIVE. If you need information to answer the user's question, CALL THE TOOLS IMMEDIATELY. Do not ask the user for permission to look up data.",
        "- If the user asks a broad question (e.g., 'Who has the most experience?'), CALL `get_all_candidates_summary()` to check everyone at once instead of checking one by one.",
        "- Whenever you need a candidate's full profile or resume text, CALL `get_candidate_details(name)`.",
        "- Do not guess candidate details if they are not in the table below. Call the tool.",
        "- If the user asks to send emails, CALL the relevant tool with user_confirmed=False first. If they already said yes, call it with user_confirmed=True.",
        "- If the user asks to export to database, CALL the export tool immediately.",
        "",
        "---",
        "# JOB DESCRIPTION",
        "",
        job_description.strip(),
        "",
        "---",
        "# CANDIDATE RANKINGS (Summary Table)",
        "",
    ]

    display_cols = [c for c in ["Rank", "Candidate Name", "Fit Score (%)", "CGPA", "Degree", "Top Skills"] if c in results_df.columns]

    if display_cols:
        header = "| " + " | ".join(display_cols) + " |"
        separator = "| " + " | ".join(["---"] * len(display_cols)) + " |"
        prompt_parts.append(header)
        prompt_parts.append(separator)

        for _, row in results_df.iterrows():
            cells = []
            for col in display_cols:
                val = row.get(col, "N/A")
                if col == "Fit Score (%)":
                    cells.append(f"{val:.1f}%" if isinstance(val, (int, float)) else str(val))
                else:
                    cells.append(str(val))
            prompt_parts.append("| " + " | ".join(cells) + " |")
        prompt_parts.append("")

    return "\n".join(prompt_parts)

# ── Chat Function ─────────────────────────────────────────────────────────────

def chat_with_assistant(
    user_message: str,
    chat_history: list[dict],
    system_prompt: str,
    tools: list
) -> tuple[str, list]:
    from google.genai import types
    client = _get_gemini_client()

    config = types.GenerateContentConfig(
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.4,
    )

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]),
        types.Content(role="model", parts=[types.Part.from_text(text="I understand the guidelines and have access to the tools. How can I help?")]),
    ]

    for msg in chat_history:
        role = msg["role"]
        parts = []
        for p in msg.get("parts", []):
            if isinstance(p, str):
                parts.append(types.Part.from_text(text=p))
            elif isinstance(p, dict) and "function_call" in p:
                parts.append(types.Part.from_function_call(name=p["function_call"]["name"], args=p["function_call"]["args"]))
            elif isinstance(p, dict) and "function_response" in p:
                parts.append(types.Part.from_function_response(name=p["function_response"]["name"], response=p["function_response"]["response"]))
        contents.append(types.Content(role=role, parts=parts))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    tool_registry = {fn.__name__: fn for fn in tools}
    trace = []

    from google.genai.errors import APIError

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=contents, config=config)

        max_hops = 3
        hops = 0
        
        while response.function_calls and hops < max_hops:
            # The model's message containing the tool calls MUST be appended first
            contents.append(response.candidates[0].content)
            
            function_responses = []
            for call in response.function_calls:
                fn = tool_registry.get(call.name)
                
                if fn:
                    try:
                        result = fn(**call.args)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool '{call.name}'"}
                    
                trace.append({"tool": call.name, "args": dict(call.args), "result": result})
                function_responses.append(
                    types.Part.from_function_response(name=call.name, response=result)
                )

            # Append all function responses in a single turn to support parallel tool calling
            contents.append(types.Content(role="user", parts=function_responses))

            response = client.models.generate_content(model="gemini-2.5-flash", contents=contents, config=config)
            hops += 1

        # Check if response.text raises ValueError (no text parts) or is empty
        try:
            response_text = response.text
        except ValueError:
            response_text = ""

        if not response_text:
            if trace:
                return "✅ Action completed successfully.", trace
            print(f"\n\n🚨 EMPTY RESPONSE DEBUG: {response.candidates[0] if response.candidates else 'No candidates'}\n\n")
            return "⚠️ **Model Error:** The AI returned an empty response (see console logs).", trace
            
        return response_text, trace

    except APIError as e:
        if e.code == 429:
            return "⚠️ **Rate Limit Reached:** You have exceeded the free tier quota for this model. Please try again later.", trace
        if e.code == 503:
            return "⚠️ **Server Overloaded:** Google's servers for this model are currently experiencing high demand. Please wait a moment and try again.", trace
        raise e

import streamlit as st

def get_state():
    """Get the current session state"""
    return st.session_state

def get_report_ctx():
    """Get the current script run context"""
    return st.runtime.scriptrunner.script_run_context.get_script_run_ctx()

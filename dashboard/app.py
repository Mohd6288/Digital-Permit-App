#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import requests
from datetime import datetime

# Change this if your backend URL/port is different
# API_BASE = "http://localhost:8000"
API_BASE = 'https://digital-permit-app.onrender.com'


# ==============================
# Helper functions
# ==============================

def api_get(path, params=None, default=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ The server is taking too long to respond. It may be starting up. Try again in a moment.")
        return default if default is not None else []
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach backend API. Check that your Render service is running and the URL is correct.")
        return default if default is not None else []
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
        return default if default is not None else []


def api_post(path, json=None, default=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json, timeout=25)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ The server is taking too long to respond. It may be starting up. Try again in a moment.")
        return default
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach backend API. Check that your Render service is running and the URL is correct.")
        return default
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
        return default


def api_put(path, json=None, default=None):
    try:
        r = requests.put(f"{API_BASE}{path}", json=json, timeout=25)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ The server is taking too long to respond. It may be starting up. Try again in a moment.")
        return default
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach backend API. Check that your Render service is running and the URL is correct.")
        return default
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
        return default


# Work category → suggested types mapping
CATEGORY_SUGGESTIONS = {
    "General Maintenance": ["Cold Work"],
    "Hot Work / Welding": ["Hot Work"],
    "Confined Space": ["Confined Space Entry"],
    "Electrical / Panels": ["Electrical / LOTO"],
    "Working at Height": ["Work at Height"],
    "Lifting / Cranes": ["Lifting Operations"],
    "Excavation / Civil": ["Excavation"],
    "Chemical Handling": ["Chemical Handling"],
}


HIGH_RISK_TYPES = {"Hot Work", "Confined Space Entry", "Work at Height", "Electrical / LOTO"}


# ==============================
# Pages
# ==============================

def safety_header():
    st.markdown(
        """
        <div style="
            padding: 18px 20px;
            border-radius: 14px;
            background: linear-gradient(90deg, #ff6b6b, #feca57);
            color: white;
            margin-bottom: 15px;
        ">
            <h2 style="margin:0; font-size: 24px;">⚠️ Safety First – Digital Permit Issuer</h2>
            <p style="margin:4px 0 0 0; font-size: 14px;">
                One place to issue, review, and close permits – with safety as the number one priority.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview():
    safety_header()
    st.title("Dashboard")

    col_top1, col_top2 = st.columns([2, 1])

    with col_top1:
        st.markdown("### Permit Overview")

        active_count = len(api_get("/permits", params={"status": "issued"}, default=[]))
        draft_count = len(api_get("/permits", params={"status": "draft"}, default=[]))
        closed_count = len(api_get("/permits", params={"status": "closed"}, default=[]))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Active Permits", active_count)
        with c2:
            st.metric("Draft Permits", draft_count)
        with c3:
            st.metric("Closed Permits", closed_count)

    with col_top2:
        st.markdown(
            """
            <div style="
                padding: 14px 16px;
                border-radius: 12px;
                background-color: #00000f1f;
                border: 1px solid #dee2e6;
                font-size: 13px;
            ">
                <b>Quick Tip:</b><br/>
                • High-risk work (Hot, Confined Space, Height, Electrical) should always have clear hazards & controls.<br/>
                • Make sure issuer & receiver sign before work starts.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Recent Permits")

    permits = api_get("/permits", default=[])
    if not permits:
        st.info("No permits yet. Go to 'Create Permit' to add the first one.")
        return

    for p in permits:
        risk_tag = ""
        if any(pt in HIGH_RISK_TYPES for pt in p.get("permit_type", [])):
            risk_tag = " | 🔴 High Risk"
        elif p.get("permit_type"):
            risk_tag = " | 🟡 Medium Risk"
        else:
            risk_tag = " | 🟢 Low Risk"

        with st.expander(f"#{p['id']} – {p['job_description'][:60]}{risk_tag}"):
            st.write(f"**Location:** {p['location']}")
            st.write(f"**Type(s):** {', '.join(p['permit_type']) or 'N/A'}")
            st.write(f"**Status:** {p['status']}")
            st.write(f"**Contractor:** {p.get('contractor') or '-'}")
            st.write(f"**Area Owner:** {p.get('area_owner') or '-'}")
            st.write(f"**Created:** {p['created_at']}")
            issuer_info = p.get("issuer_name") or "-"
            rec_info = p.get("receiver_name") or "-"
            st.write(f"**Issuer:** {issuer_info}")
            st.write(f"**Receiver:** {rec_info}")
            if st.button("Open details", key=f"detail_{p['id']}"):
                st.session_state["selected_permit_id"] = p["id"]
                st.session_state["page"] = "Permit Details"
                st.rerun()


def page_create_permit():
    safety_header()
    st.title("Create New Permit")

    with st.container():
        st.markdown(
            """
            <div style="
                padding: 10px 14px;
                border-radius: 10px;
                background-color: #e7f5ff;
                border: 1px solid #d0ebff;
                font-size: 13px;
                margin-bottom: 10px;
            ">
                Fill in the job details, choose a work category, and let the system suggest hazards and controls for you.
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_main, col_side = st.columns([2.2, 1.3])

    with col_main:
        with st.form("permit_form"):
            job_description = st.text_area("Job Description *", height=120,
                                           placeholder="Example: Hot work on pump P-101 in main plant area...")
            location = st.text_input("Work Location *", placeholder="Example: Main Plant – Area B")
            contractor = st.text_input("Contractor Company")
            area_owner = st.text_input("Area Owner / Department")
            equipment_code = st.text_input("Equipment / Asset Code")

            st.markdown("#### Work Category")
            category = st.selectbox(
                "Select category (helps suggest permit type & risk)",
                ["-- Select --"] + list(CATEGORY_SUGGESTIONS.keys())
            )

            st.markdown("#### Type of Work (you can pick multiple)")
            col1, col2 = st.columns(2)
            types = []
            with col1:
                if st.checkbox("Hot Work"):
                    types.append("Hot Work")
                if st.checkbox("Cold Work"):
                    types.append("Cold Work")
                if st.checkbox("Confined Space Entry"):
                    types.append("Confined Space Entry")
                if st.checkbox("Work at Height"):
                    types.append("Work at Height")
            with col2:
                if st.checkbox("Electrical / LOTO"):
                    types.append("Electrical / LOTO")
                if st.checkbox("Excavation"):
                    types.append("Excavation")
                if st.checkbox("Lifting Operations"):
                    types.append("Lifting Operations")
                if st.checkbox("Chemical Handling"):
                    types.append("Chemical Handling")

            gas_test_required = st.checkbox("Gas Test Required?", value=False)

            st.markdown("#### Permit Parties")
            issuer_name = st.text_input("Issuer Name *", placeholder="Person issuing the permit")
            receiver_name = st.text_input("Receiver Name", placeholder="Supervisor / Person in charge")

            use_ai = st.checkbox("Use AI to suggest hazards and controls", value=True)

            submitted = st.form_submit_button("Create Permit")

        # Outside form: show suggestions from category
        if category != "-- Select --":
            suggested = CATEGORY_SUGGESTIONS[category]
            st.markdown("##### Suggested work type for this category")
            st.info(", ".join(suggested))

    with col_side:
        st.markdown("### Safety Snapshot")
        if category != "-- Select --":
            if any(t in HIGH_RISK_TYPES for t in CATEGORY_SUGGESTIONS[category]):
                st.markdown("**Risk Level:** 🔴 High Risk")
                st.write("Make sure emergency & rescue plans are in place.")
            else:
                st.markdown("**Risk Level:** 🟡 Medium / Low")
        else:
            st.markdown("**Risk Level:** –")

        st.markdown("---")
        st.markdown("### What AI will do")
        st.write("• Suggest hazards based on description")
        st.write("• Suggest control measures (PPE, procedures)")
        st.write("• You can still edit everything later in the details page.")

    if submitted:
        if not job_description or not location:
            st.error("Job description and location are required.")
            return
        if not issuer_name:
            st.error("Issuer name is required.")
            return

        hazards = []
        controls = []
        if use_ai:
            ai_resp = api_post("/ai/hazards", {"job_description": job_description}, default=None)
            if ai_resp:
                hazards = ai_resp.get("hazards", [])
                controls = ai_resp.get("controls", [])

        new_permit = {
            "job_description": job_description,
            "location": location,
            "permit_type": types,
            "contractor": contractor or None,
            "area_owner": area_owner or None,
            "equipment_code": equipment_code or None,
            "hazards": hazards,
            "controls": controls,
            "gas_test_required": gas_test_required,
            "status": "draft",
            "issuer_name": issuer_name,
            "receiver_name": receiver_name or None,
            "issuer_signed": False,
            "receiver_signed": False,
            "issuer_signed_at": None,
            "receiver_signed_at": None,
        }

        created = api_post("/permits", json=new_permit, default=None)
        if created:
            st.success(f"Permit #{created['id']} created.")
            st.session_state["selected_permit_id"] = created["id"]
            st.session_state["page"] = "Permit Details"
            st.rerun()


def show_gas_tests(permit_id: int):
    st.subheader("Gas Test Readings")
    tests = api_get(f"/gas-tests/{permit_id}", default=[])
    if not tests:
        st.info("No gas tests recorded.")
        return
    for g in sorted(tests, key=lambda x: x["timestamp"], reverse=True):
        st.write(
            f"- **{g['timestamp']}** | O₂ {g['o2']}% | LEL {g['lel']}% | H₂S {g['h2s']} ppm | CO {g['co']} ppm"
        )


def page_permit_details():
    safety_header()

    pid = st.session_state.get("selected_permit_id")
    if not pid:
        st.warning("No permit selected. Go to 'Overview' and choose a permit.")
        return

    permit = api_get(f"/permits/{pid}", default=None)
    if not permit:
        st.error("Could not load permit details.")
        return

    st.title(f"Permit #{permit['id']}")

    # Top info cards
    c1, c2, c3 = st.columns(3)
    with c1:
        risk = "Low"
        icon = "🟢"
        if any(pt in HIGH_RISK_TYPES for pt in permit.get("permit_type", [])):
            risk = "High"
            icon = "🔴"
        elif permit.get("permit_type"):
            risk = "Medium"
            icon = "🟡"
        st.metric("Risk Level", f"{icon} {risk}")
    with c2:
        st.metric("Status", permit["status"].capitalize())
    with c3:
        st.metric("Location", permit["location"])

    st.markdown("---")

    col_left, col_right = st.columns([1.8, 1.2])

    # LEFT: Details and safety info
    with col_left:
        st.markdown("### Job Details")
        st.write(f"**Job:** {permit['job_description']}")
        st.write(f"**Type(s):** {', '.join(permit['permit_type']) or 'N/A'}")
        st.write(f"**Contractor:** {permit.get('contractor') or '-'}")
        st.write(f"**Area Owner:** {permit.get('area_owner') or '-'}")
        st.write(f"**Equipment:** {permit.get('equipment_code') or '-'}")
        st.write(f"**Created:** {permit['created_at']}")
        st.write(f"**Updated:** {permit['updated_at']}")

        st.markdown("#### Hazards")
        if permit["hazards"]:
            for h in permit["hazards"]:
                st.markdown(f"- {h}")
        else:
            st.info("No hazards listed.")

        st.markdown("#### Control Measures")
        if permit["controls"]:
            for c in permit["controls"]:
                st.markdown(f"- {c}")
        else:
            st.info("No controls listed.")

        st.markdown("---")
        st.subheader("Status & Gas Tests")

        new_status = st.selectbox(
            "Permit status",
            options=["draft", "issued", "closed"],
            index=["draft", "issued", "closed"].index(permit["status"]),
        )
        if st.button("Save Status"):
            updated = api_put(f"/permits/{pid}", {"status": new_status}, default=None)
            if updated:
                st.success(f"Status updated to {updated['status']}")
                st.rerun()

        st.markdown("##### Gas Tests")
        show_gas_tests(pid)

        with st.form("gas_form"):
            st.markdown("Add Gas Test Reading")
            o2 = st.number_input("O₂ (%)", min_value=0.0, max_value=25.0, value=20.9)
            lel = st.number_input("LEL (%)", min_value=0.0, max_value=100.0, value=0.0)
            h2s = st.number_input("H₂S (ppm)", min_value=0.0, value=0.0)
            co = st.number_input("CO (ppm)", min_value=0.0, value=0.0)
            submitted = st.form_submit_button("Add Gas Test")

        if submitted:
            payload = {
                "permit_id": pid,
                "o2": o2,
                "lel": lel,
                "h2s": h2s,
                "co": co,
                "timestamp": datetime.utcnow().isoformat(),
            }
            added = api_post("/gas-tests", payload, default=None)
            if added:
                st.success("Gas test added.")
                st.rerun()

    # RIGHT: Signatures pane
    with col_right:
        st.markdown("### Signatures")

        issuer_name = permit.get("issuer_name") or "-"
        receiver_name = permit.get("receiver_name") or "-"

        issuer_signed = permit.get("issuer_signed", False)
        receiver_signed = permit.get("receiver_signed", False)
        issuer_signed_at = permit.get("issuer_signed_at") or "-"
        receiver_signed_at = permit.get("receiver_signed_at") or "-"

        st.markdown("#### Issuer")
        st.write(f"**Name:** {issuer_name}")
        st.write(f"**Signed:** {'✅ Yes' if issuer_signed else '❌ No'}")
        st.write(f"**Signed at:** {issuer_signed_at}")
        signer_name_input = st.text_input("Confirm issuer name to sign", key="issuer_sign_name")
        if st.button("Sign as Issuer"):
            if signer_name_input.strip() == "" or signer_name_input.strip().lower() != (issuer_name or "").strip().lower():
                st.error("Name must match the issuer name to sign.")
            else:
                payload = {
                    "issuer_signed": True,
                    "issuer_signed_at": datetime.utcnow().isoformat(),
                }
                updated = api_put(f"/permits/{pid}", payload, default=None)
                if updated:
                    st.success("Issuer signed successfully.")
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Receiver")
        st.write(f"**Name:** {receiver_name}")
        st.write(f"**Signed:** {'✅ Yes' if receiver_signed else '❌ No'}")
        st.write(f"**Signed at:** {receiver_signed_at}")
        receiver_sign_name_input = st.text_input("Confirm receiver name to sign", key="receiver_sign_name")
        if st.button("Sign as Receiver"):
            if not receiver_name:
                st.error("No receiver name set for this permit.")
            elif receiver_sign_name_input.strip() == "" or receiver_sign_name_input.strip().lower() != (receiver_name or "").strip().lower():
                st.error("Name must match the receiver name to sign.")
            else:
                payload = {
                    "receiver_signed": True,
                    "receiver_signed_at": datetime.utcnow().isoformat(),
                }
                updated = api_put(f"/permits/{pid}", payload, default=None)
                if updated:
                    st.success("Receiver signed successfully.")
                    st.rerun()


# ==============================
# Main app
# ==============================

def main():
    st.set_page_config(
        page_title="Digital Permit Issuer",
        page_icon="🦺",
        layout="wide",
    )

    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Create Permit", "Permit Details"],
        index=["Overview", "Create Permit", "Permit Details"].index(st.session_state["page"]),
    )
    st.session_state["page"] = page

    if page == "Overview":
        page_overview()
    elif page == "Create Permit":
        page_create_permit()
    elif page == "Permit Details":
        page_permit_details()


if __name__ == "__main__":
    main()

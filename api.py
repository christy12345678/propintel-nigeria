import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from catboost import CatBoostRegressor
import joblib
from google import genai
from google.genai import types
from groq import Groq  
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# =====================================================================
# SYSTEM INITIALIZATION & SECURE KEY EXTRACTION
# =====================================================================
st.set_page_config(page_title="PropIntel Nigeria", page_icon="🇳🇬", layout="wide")
load_dotenv() # Load hidden variables from the local .env file into memory

# 👇 TEMPORARY KEY DEBUGGER (Remove before pushing to GitHub) 👇
st.sidebar.write("### 🔍 Key Diagnostic Status")
st.sidebar.write(f"Gemini Key Loaded: {bool(os.environ.get('GEMINI_API_KEY'))}")
st.sidebar.write(f"Groq Key 1 Loaded: {bool(os.environ.get('GROQ_API_KEY_1'))}")
st.sidebar.write(f"Groq Key 2 Loaded: {bool(os.environ.get('GROQ_API_KEY_2'))}")


YOUR_PROJECT_NAME = "PropIntel_Core_Engine_v1"
EXCHANGE_RATE_PARALLEL = 1530.0  
USER_DB_FILE = "app_users.json"
FEEDBACK_FILE = "user_feedback.csv"
MASTER_DATA_PATH = "C:\\Users\\HI\\Documents\\nigeria_houses_data.csv"

# Ingest keys from system memory
gemini_key = os.environ.get("GEMINI_API_KEY")
groq_key_1 = os.environ.get("GROQ_API_KEY_1")
groq_key_2 = os.environ.get("GROQ_API_KEY_2")

# Ensure strip removes any invisible Windows line breaks or spaces
if gemini_key: gemini_key = gemini_key.strip()
if groq_key_1: groq_key_1 = groq_key_1.strip()
if groq_key_2: groq_key_2 = groq_key_2.strip()

# Initialize Cloud API Clients
client_gemini = genai.Client(api_key=gemini_key)
client_groq_1 = Groq(api_key=groq_key_1)
client_groq_2 = Groq(api_key=groq_key_2)

# =====================================================================
# DATA VALIDATION SCHEMAS (STRUCTURAL FIREWALL)
# =====================================================================
class PropertyQuerySchema(BaseModel):
    bedrooms: int = Field(..., ge=1, le=12)
    bathrooms: int = Field(..., ge=1, le=12)
    toilets: int = Field(..., ge=1, le=15)
    state: str = Field(..., min_length=2)
    town: str = Field(..., min_length=2)
    title: str = Field(..., min_length=2)
    is_luxury: int = Field(..., ge=0, le=1)
    is_new: int = Field(..., ge=0, le=1)
    has_bq: int = Field(..., ge=0, le=1)

class LLMAuditSchema(BaseModel):
    verdict: str  
    reasoning: str
    finalized_price: float

# =====================================================================
# MODEL LOADING ENGINE
# =====================================================================
@st.cache_resource
def load_production_models():
    tier_map = joblib.load(f"{YOUR_PROJECT_NAME}_tier_mapping.pkl")
    models = {
        "Tier_1_Premium": CatBoostRegressor().load_model(f"{YOUR_PROJECT_NAME}_Tier_1_Premium.cbm"),
        "Tier_2_Growth_Hubs": CatBoostRegressor().load_model(f"{YOUR_PROJECT_NAME}_Tier_2_Growth_Hubs.cbm"),
        "Tier_3_Baseline": CatBoostRegressor().load_model(f"{YOUR_PROJECT_NAME}_Tier_3_Baseline.cbm")
    }
    return tier_map, models

try:
    tier_mapping, models_vault = load_production_models()
except Exception as e:
    st.error(f"Live engine profile assets missing! Run main.py first.")
    st.stop()

# =====================================================================
# THE HYBRID SELF-CACHING VALUATION & CASCADING AUDIT CORE
# =====================================================================
def get_hybrid_intelligent_valuation(inputs: PropertyQuerySchema):
    state_norm = inputs.state.strip().title()
    town_norm = inputs.town.strip().title()
    title_norm = inputs.title.strip().title()
    
    # ─── PHASE 1: KNOWLEDGE CACHE LOOKUP ───
    if os.path.exists(MASTER_DATA_PATH):
        try:
            cache_df = pd.read_csv(MASTER_DATA_PATH)
            match = cache_df[
                (cache_df['state'].astype(str).str.strip().str.title() == state_norm) &
                (cache_df['town'].astype(str).str.strip().str.title() == town_norm) &
                (cache_df['bedrooms'] == inputs.bedrooms) &
                (cache_df['bathrooms'] == inputs.bathrooms) &
                (cache_df['toilets'] == inputs.toilets)
            ]
            if not match.empty:
                st.toast("⚡ Instant Database Cache Hit!", icon="💾")
                assigned_tier = tier_mapping.get(state_norm, 'Tier_3_Baseline')
                return float(match.iloc['price']), assigned_tier, "DATABASE_CACHE (Instant Retrieval)"
        except Exception:
            pass

    # ─── PHASE 2: RUN LOCAL CATBOOST BACKEND MODEL ───
    bed_bath_ratio = inputs.bedrooms / (inputs.bathrooms + 0.1)
    state_town_combo = f"{state_norm}_{town_norm}"
    
    macro_profiles = {
        'Lagos': [5, 0.88], 'Abuja': [4, 0.90], 'Rivers': [4, 0.75], 'Delta': [3, 0.72],
        'Oyo': [3, 0.65], 'Anambra': [3, 0.70], 'Kano': [3, 0.58], 'Enugu': [3, 0.66]
    }
    igr, dev = macro_profiles.get(state_norm, [2, 0.55])
    assigned_tier = tier_mapping.get(state_norm, 'Tier_3_Baseline')
    chosen_model = models_vault[assigned_tier]
    
    input_df = pd.DataFrame([{
        'bedrooms': inputs.bedrooms, 'bathrooms': inputs.bathrooms, 'toilets': inputs.toilets,
        'state': state_norm, 'town': town_norm, 'title': title_norm,
        'is_luxury_estate': inputs.is_luxury, 'is_new_build': inputs.is_new, 'has_boys_quarter': inputs.has_bq,
        'bed_bath_ratio': bed_bath_ratio, 'state_town_combo': state_town_combo,
        'state_igr_tier': igr, 'dev_index': dev
    }])
    
    # REPLACE IT WITH THIS BULLETPROOF EXTRACTION LAYER:
    prediction_raw = chosen_model.predict(input_df)
    clean_scalar_price = float(prediction_raw[0]) if hasattr(prediction_raw, '__len__') else float(prediction_raw)
    catboost_guess = max(clean_scalar_price, 5_000_000)

    # 🟢 MAKE SURE THIS PROMPT BLOCK HAS EXACTLY 4 SPACES OF INDENTATION BEFORE IT
    prompt = f"""
    You are an elite real estate value auditor in Nigeria.
    Our model predicted that a property in {state_norm} ({town_norm}) with {inputs.bedrooms} beds, 
    {inputs.bathrooms} baths, layout type '{title_norm}' has a fair value of ₦{catboost_guess:,.2f}.
    
    Task: Assess if this number matches current real-world pricing trends for this specific neighborhood.
    Return a valid JSON payload matching this structure exactly: 
    {{"verdict": "APPROVED", "reasoning": "text reason", "finalized_price": 12345.0}}
    """

    # ─── PHASE 3: PRIMARY AI AUDITOR -> GEMINI 2.5 FLASH ───
    try:
        response = client_gemini.models.generate_content(
            model='gemini-2.5-flash', contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=LLMAuditSchema, temperature=0.1),
        )
        audit_output = json.loads(response.text)
        finalized_price = max(float(audit_output['finalized_price']), 5_000_000)
        
        if os.path.exists(MASTER_DATA_PATH):
            pd.DataFrame([{'bedrooms': inputs.bedrooms, 'bathrooms': inputs.bathrooms, 'toilets': inputs.toilets, 'state': state_norm, 'town': town_norm, 'title': title_norm, 'price': finalized_price}]).to_csv(MASTER_DATA_PATH, mode='a', header=False, index=False)
        return finalized_price, assigned_tier, f"🥇 GEMINI_AUDITED: {audit_output['reasoning']}"

    except Exception as gemini_err:
        st.toast("⚠️ Gemini API Ceiling Met. Engaging Groq Channel 1 Failover...", icon="🔄")
        
        # ─── PHASE 4: SECONDARY AI AUDITOR -> GROQ LLAMA 3 (KEY 1) ───
        try:
            chat_completion = client_groq_1.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192", response_format={"type": "json_object"}, temperature=0.1
            )
            groq_1_output = json.loads(chat_completion.choices.message.content)
            finalized_price = max(float(groq_1_output['finalized_price']), 5_000_000)
            
            if os.path.exists(MASTER_DATA_PATH):
                pd.DataFrame([{'bedrooms': inputs.bedrooms, 'bathrooms': inputs.bathrooms, 'toilets': inputs.toilets, 'state': state_norm, 'town': town_norm, 'title': title_norm, 'price': finalized_price}]).to_csv(MASTER_DATA_PATH, mode='a', header=False, index=False)
            return finalized_price, assigned_tier, f"🥈 GROQ_CH_1_AUDITED: {groq_1_output['reasoning']}"
            
        except Exception as groq_1_err:
            st.toast("⚠️ Groq Key 1 Rate Limited. Engaging Groq Channel 2 Backup...", icon="🔄")
            
            # ─── PHASE 5: TERTIARY AI AUDITOR -> GROQ LLAMA 3 (KEY 2) ───
            try:
                chat_completion = client_groq_2.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192", response_format={"type": "json_object"}, temperature=0.1
                )
                groq_2_output = json.loads(chat_completion.choices.message.content)
                finalized_price = max(float(groq_2_output['finalized_price']), 5_000_000)
                
                if os.path.exists(MASTER_DATA_PATH):
                    pd.DataFrame([{'bedrooms': inputs.bedrooms, 'bathrooms': inputs.bathrooms, 'toilets': inputs.toilets, 'state': state_norm, 'town': town_norm, 'title': title_norm, 'price': finalized_price}]).to_csv(MASTER_DATA_PATH, mode='a', header=False, index=False)
                return finalized_price, assigned_tier, f"🥉 GROQ_CH_2_AUDITED: {groq_2_output['reasoning']}"
                
            except Exception as groq_2_err:
                # ─── PHASE 6: CRITICAL PIPELINE FALLBACK -> CATBOOST DIRECT ───
                st.toast("🚨 All API Gateways Exhausted. Engaging local backend logic.", icon="🛑")
                return catboost_guess, assigned_tier, f"🏅 LOCAL_CATBOOST_DIRECT (Bypass due to Cloud Errors)"
# =====================================================================
# INITIALIZE GLOBAL VARIABLE STATES FIRST (CRITICAL FIX)
# =====================================================================
# Always verify and set defaults before running conditional interface checks!
if "logged_in" not in st.session_state: 
    st.session_state["logged_in"] = False

if "username" not in st.session_state: 
    st.session_state["username"] = None

if not os.path.exists(USER_DB_FILE):
    with open(USER_DB_FILE, "w") as f: 
        json.dump({"admin": "password123"}, f)

if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["User", "Town", "State", "Predicted_Naira", "Rating", "Comments"]).to_csv(FEEDBACK_FILE, index=False)
# =====================================================================
# FRONTEND INTERFACE VIEW DESIGN (AUTHENTICATION GATEWAY)
# =====================================================================
if not st.session_state["logged_in"]:
    # 🔒 This screen only renders if the user has NOT authenticated
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🇳🇬 PropIntel Intelligence Suite</h1>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns(3)
    with col_c:
        st.markdown("---")
        user_in = st.text_input("Username / Email Profile")
        pass_in = st.text_input("Security Key Password", type="password")
        if st.button("Authenticate Dashboard", use_container_width=True):
            with open(USER_DB_FILE, "r") as f: users = json.load(f)
            if user_in in users and users[user_in] == pass_in:
                st.session_state["logged_in"] = True
                st.session_state["username"] = user_in
                st.rerun()
            else: 
                st.error("Access parameters failed.")

else:
    # 🔓 CRITICAL FIX: EVERYTHING BELOW IS NOW PERFECTLY INDENTED INSIDE THIS ELSE BLOCK
    # This guarantees the dashboard will remain hidden until a successful login happens!
    st.sidebar.markdown(f"### 🔐 User session: `{st.session_state['username']}`")
    if st.sidebar.button("Log out of Dashboard", use_container_width=True):
        st.session_state["logged_in"] = False
        st.rerun()

    tab_val, tab_feedback = st.tabs(["🏠 Hybrid AI Valuation Engine", "📩 Intelligent Feedback Audit"])
    
    with tab_val:
        st.markdown("### Real Estate Parameters Configuration Matrix")
        r1, r2, r3 = st.columns(3)
        with r1: state_box = st.selectbox("Target State", sorted(list(tier_mapping.keys())))
        with r2: town_box = st.text_input("Specific Town Name", value="Lekki Phase 1")
        with r3: title_box = st.selectbox("Property Design Layout", ["House", "Duplex", "Terrace House", "Flat"])
        
        r4, r5, r6 = st.columns(3)
        with r4: beds_slider = st.slider("Bedrooms", 1, 10, 4)
        with r5: baths_slider = st.slider("Bathrooms", 1, 10, 4)
        with r6: toilets_slider = st.slider("Toilets", 1, 10, 5)
        
        luxury_cb = st.checkbox("Gated / Serviced Estate Layout")
        new_cb = st.checkbox("Brand New Finish Build")
        bq_cb = st.checkbox("Includes Serviced BQ / Outbuilding")

        try:
            validated_query = PropertyQuerySchema(
                bedrooms=beds_slider, bathrooms=baths_slider, toilets=toilets_slider,
                state=state_box, town=town_box, title=title_box,
                is_luxury=int(luxury_cb), is_new=int(new_cb), has_bq=int(bq_cb)
            )
            
            final_price, tier_label, pipeline_trace = get_hybrid_intelligent_valuation(validated_query)
            
            st.markdown("---")
            st.markdown("### 🏷️ Intelligent Negotiation Valuation Bounds")
            
            if "🥇" in pipeline_trace: st.success(f"⚡ **Pipeline Trace:** {pipeline_trace}")
            elif "🥈" in pipeline_trace: st.warning(f"⚡ **Pipeline Trace:** {pipeline_trace}")
            elif "🥉" in pipeline_trace: st.error(f"⚡ **Pipeline Trace:** {pipeline_trace}")
            else: st.info(f"⚡ **Pipeline Trace:** {pipeline_trace}")
            
            mae_buffers = {"Tier_1_Premium": 55_415_796, "Tier_2_Growth_Hubs": 32_683_031, "Tier_3_Baseline": 11_107_550}
            current_mae = mae_buffers.get(tier_label, 25_000_000)
            v1, v2, v3 = st.columns(3)
            v1.metric("Aggressive Buyer Bid", f"₦{max(final_price - (current_mae*0.8), final_price*0.7):,.2f}")
            v2.metric("Audited Fair Market Value Anchor", f"₦{final_price:,.2f}")
            v3.metric("Seller Ceiling Limit", f"₦{final_price + (current_mae*1.1):,.2f}")
            
        except Exception as validation_error:
            st.error(f"Pydantic Structural Block Error: {validation_error}")

    with tab_feedback:
        st.markdown("### 📩 Autonomous Feedback Auditing System")
        with st.form("live_feedback_audit_form", clear_on_submit=True):
            rating_select = st.select_slider("Accuracy Rating", options=["Inaccurate", "Acceptable Guess", "Perfect"])
            feedback_comment = st.text_input("Enter exact real-world selling price or neighborhood notes:")
            
            if st.form_submit_button("Verify & Auto-Inject Entry"):
                raw_digits = ''.join(filter(str.isdigit, str(feedback_comment)))
                claimed_price = float(raw_digits) if raw_digits else final_price
                
                audit_prompt = f"Verify if a user claiming a property in {state_box} ({town_box}) sold for ₦{claimed_price:,.2f} is realistic or spam."
                try:
                    res = client_gemini.models.generate_content(
                        model='gemini-2.5-flash', contents=audit_prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=LLMAuditSchema, temperature=0.1)
                    )
                    audit_res = json.loads(res.text)
                    if audit_res['verdict'] == "REJECTED_SPAM":
                        st.error(f"⛔ Blocked by AI Firewall! Reason: {audit_res['reasoning']}")
                    else:
                        pd.DataFrame([{"User": st.session_state["username"], "Town": town_box, "State": state_box, "Predicted_Naira": audit_res['finalized_price'], "Rating": rating_select, "Comments": audit_res['reasoning']}]).to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
                        st.toast(f"✅ Auto-Injected: {audit_res['verdict']}", icon="💾")
                except Exception:
                    pd.DataFrame([{"User": st.session_state["username"], "Town": town_box, "State": state_box, "Predicted_Naira": claimed_price, "Rating": rating_select, "Comments": feedback_comment}]).to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
                    st.toast("✅ Saved successfully to standard log files.", icon="💾")

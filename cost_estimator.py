import streamlit as st
import pandas as pd
import io
import requests
from bs4 import BeautifulSoup
import re

# Page configuration
st.set_page_config(page_title="Universal AI & OCR Cost Estimator", layout="wide")

# Fetch dynamic Azure OpenAI pricing
def fetch_openai_pricing():
    url = "https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/"
    pricing = {}
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for table in soup.find_all('table'):
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            if 'Model' not in headers:
                continue
            p_idx = next((i for i,h in enumerate(headers) if 'Prompt' in h or 'Input' in h), None)
            c_idx = next((i for i,h in enumerate(headers) if 'Completion' in h or 'Output' in h), None)
            if p_idx is None or c_idx is None:
                continue
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) <= max(p_idx, c_idx):
                    continue
                name = cols[0].get_text(strip=True)
                p_txt, c_txt = cols[p_idx].get_text(), cols[c_idx].get_text()
                m1 = re.search(r"\$([\d,]+\.?\d*)", p_txt)
                m2 = re.search(r"\$([\d,]+\.?\d*)", c_txt)
                if m1 and m2:
                    pricing[name] = {
                        'prompt': float(m1.group(1).replace(',', ''))/1000,
                        'completion': float(m2.group(1).replace(',', ''))/1000
                    }
    except:
        pass
    return pricing

# Static pricing defaults for core GPT models
STATIC_PRICING = {
    'GPT-3.5 Turbo':      {'prompt':0.0015,   'completion':0.002},
    'GPT-3.5 Turbo 16K':  {'prompt':0.003,    'completion':0.004},
    'GPT-4':              {'prompt':0.03,     'completion':0.06},
    'GPT-4o':             {'prompt':0.00275,  'completion':0.011},
    'GPT-4.5 Preview':    {'prompt':0.075,    'completion':0.150},
    'o3-mini Global':     {'prompt':0.00110,  'completion':0.00440}
}
# Combine dynamic and static pricing
dynamic = fetch_openai_pricing()
PRICING = {**STATIC_PRICING, **dynamic}

# Models metadata
top_models = [
    {'Model':'GPT-3.5 Turbo','Release':'2022-11-30'},
    {'Model':'GPT-4','Release':'2023-03-14'},
    {'Model':'GPT-4o','Release':'2024-05-01'},
    {'Model':'GPT-4.5 Preview','Release':'2025-02-27'},
    {'Model':'o3-mini Global','Release':'2025-01-31'}
]

# Static equivalents for non-Azure providers
equivalents_static = {
    'GPT-3.5 Turbo': {
        'OpenAI':    {'model':'gpt-3.5-turbo','prompt':0.0015,'completion':0.002},
        'GCP':       {'model':'PaLM 2','prompt':0.002,'completion':0.003},
        'Anthropic': {'model':'Claude 2','prompt':0.0018,'completion':0.0025},
        'Mistral':   {'model':'Mistral 7B','prompt':0.00025,'completion':0.00025}
    },
    'GPT-4': {
        'OpenAI':    {'model':'gpt-4','prompt':0.03,'completion':0.06},
        'GCP':       {'model':'PaLM 2 Pro','prompt':0.04,'completion':0.08},
        'Anthropic': {'model':'Claude 3','prompt':0.025,'completion':0.05},
        'Mistral':   {'model':'Mistral 7B Instruct','prompt':0.00025,'completion':0.00025}
    },
    'GPT-4o': {
        'OpenAI':    {'model':'gpt-4o','prompt':0.00275,'completion':0.011},
        'GCP':       {'model':'PaLM 2 Vision','prompt':0.05,'completion':0.1},
        'Anthropic': {'model':'Claude 2 Vision','prompt':0.03,'completion':0.06},
        'Mistral':   {'model':'Mistral Vision','prompt':0.00025,'completion':0.00025}
    },
    'GPT-4.5 Preview': {
        'OpenAI':    {'model':'gpt-4.5-preview','prompt':0.075,'completion':0.15},
        'GCP':       {'model':'Gemini Pro','prompt':0.05,'completion':0.1},
        'Anthropic': {'model':'Claude 3 Sonnet','prompt':0.03,'completion':0.06},
        'Mistral':   {'model':'Mistral Perf 7B','prompt':0.00025,'completion':0.00025}
    },
    'o3-mini Global': {
        'OpenAI':    {'model':'gpt-3.5-turbo-mini','prompt':0.00110,'completion':0.00440},
        'GCP':       {'model':'PaLM 2 Small','prompt':0.005,'completion':0.02},
        'Anthropic': {'model':'Claude Instant','prompt':0.002,'completion':0.007},
        'Mistral':   {'model':'Mistral Mega B','prompt':0.00025,'completion':0.00025}
    }
}

# Build equivalents dict with dynamic Azure entries
equivalents = {}
for m in equivalents_static:
    azure_rates = PRICING.get(m, {'prompt': None, 'completion': None})
    equivalents[m] = {
        'Azure':     {'model': m, 'prompt': azure_rates['prompt'], 'completion': azure_rates['completion']},
        **equivalents_static[m]
    }

# Sidebar inputs
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Mode", ["AI Model Cost","OCR Cost"])

if mode == "AI Model Cost":
    st.header("AI Model Cost Estimator")
    in_tok  = st.sidebar.number_input("Prompt tokens/doc",    2000, step=100)
    out_tok = st.sidebar.number_input("Completion tokens/doc", 500,  step=100)
    docs    = st.sidebar.number_input("Docs per month",        100,  step=1)
    model   = st.sidebar.selectbox("Model", list(PRICING.keys()))

    p_rate  = PRICING[model]['prompt']
    c_rate  = PRICING[model]['completion']
    t_in    = in_tok * docs
    t_out   = out_tok * docs
    cost_in = t_in/1000 * p_rate
    cost_out= t_out/1000 * c_rate
    total   = cost_in + cost_out

    col1, col2, col3 = st.columns(3)
    col1.metric("Prompt Rate ($/1K)", f"{p_rate:.4f}")
    col2.metric("Completion Rate ($/1K)", f"{c_rate:.4f}")
    col3.metric("Monthly Cost", f"${total:,.2f}")

    # Top models table
    st.subheader("Top Model Cost Comparison")
    df_top = pd.DataFrame(top_models)
    df_top['Release'] = pd.to_datetime(df_top['Release'])
    price_df = pd.DataFrame.from_dict(PRICING, orient='index')\
                   .reset_index().rename(columns={'index':'Model','prompt':'Prompt Price','completion':'Completion Price'})
    df_comp = df_top.merge(price_df, on='Model', how='left')
    df_comp['Cost In']  = df_comp['Prompt Price']    * t_in/1000
    df_comp['Cost Out'] = df_comp['Completion Price'] * t_out/1000
    df_comp['Total']    = df_comp['Cost In'] + df_comp['Cost Out']
    st.dataframe(df_comp.style.format({
        'Release':'{:%Y-%m-%d}',
        'Prompt Price':'{:.2f}',
        'Completion Price':'{:.2f}',
        'Cost In':'${:,.2f}',
        'Cost Out':'${:,.2f}',
        'Total':'${:,.2f}'
    }), use_container_width=True)

    # Multi-cloud comparison
    st.subheader(f"Multi-Cloud Cost Comparison for {model}")
    mc = equivalents.get(model, {})
    rows = []
    for prov, info in mc.items():
        ci = t_in/1000 * info['prompt'] if info['prompt'] is not None else None
        co = t_out/1000 * info['completion'] if info['completion'] is not None else None
        rows.append({'Provider':prov, 'Model':info['model'], 'Cost Prompt ($)':ci, 'Cost Completion ($)':co, 'Total Cost ($)': (ci or 0)+(co or 0)})
    df_mc = pd.DataFrame(rows)
    st.dataframe(df_mc.style.format({
        'Cost Prompt ($)':'${:,.2f}',
        'Cost Completion ($)':'${:,.2f}',
        'Total Cost ($)':'${:,.2f}'
    }), use_container_width=True)

    # Export report
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        summary = pd.DataFrame([
            {'Parameter':'Mode','Value':mode},
            {'Parameter':'Model','Value':model},
            {'Parameter':'Prompt tokens/doc','Value':in_tok},
            {'Parameter':'Completion tokens/doc','Value':out_tok},
            {'Parameter':'Docs per month','Value':docs},
            {'Parameter':'Monthly Cost','Value':round(total,2)}
        ])
        summary.to_excel(writer, index=False, sheet_name='Report', startrow=0)
        df_comp.to_excel(writer, index=False, sheet_name='Report', startrow=len(summary)+2)
        df_mc.to_excel(writer, index=False, sheet_name='Report', startrow=len(summary)+len(df_comp)+4)
    buf.seek(0)
    st.download_button("Download Excel Report", buf, "ai_ocr_report.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.header("Document OCR Cost Comparison")
    pages = st.sidebar.number_input("Pages/doc",    1, step=1)
    docs  = st.sidebar.number_input("Docs/month",  100, step=1)
    rates = {'Azure Read API':1.50/1000,'AWS Textract':1.50/1000,'Google OCR':1.50/1000}
    ocr = []
    for prov, rate in rates.items():
        cost = pages*docs*rate
        ocr.append({'Provider':prov,'Monthly Cost ($)':cost})
    df_ocr = pd.DataFrame(ocr)
    df_ocr['Monthly Cost ($)'] = df_ocr['Monthly Cost ($)'].map(lambda x: f"${x:,.2f}")
    st.dataframe(df_ocr, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        params = pd.DataFrame([
            {'Parameter':'Mode','Value':mode},
            {'Parameter':'Pages/doc','Value':pages},
            {'Parameter':'Docs/month','Value':docs}
        ])
        params.to_excel(writer, index=False, sheet_name='Report', startrow=0)
        df_ocr.to_excel(writer, index=False, sheet_name='Report', startrow=len(params)+2)
    buf.seek(0)
    st.download_button("Download OCR Report", buf, "ocr_report.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.info("Install: pip install streamlit pandas requests beautifulsoup4 && streamlit run cost_estimator.py")
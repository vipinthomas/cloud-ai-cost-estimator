import streamlit as st
import pandas as pd
import io
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="Azure AI Model Cost Estimator & Comparison", layout="centered")

@st.cache_data(show_spinner=False)
def fetch_all_model_prices():
    """
    Scrape all pricing tables on the Azure OpenAI Service page.
    Returns dict: model_name -> {'prompt': price_per_1k, 'completion': price_per_1k}
    """
    url = "https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/"
    pricing = {}
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Iterate all tables to capture every model
        for table in soup.find_all('table'):
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            if 'Model' not in headers:
                continue
            # find prompt and completion columns
            prompt_idx = next((i for i, h in enumerate(headers) if 'Prompt' in h or 'Input' in h), None)
            comp_idx = next((i for i, h in enumerate(headers) if 'Completion' in h or 'Output' in h), None)
            if prompt_idx is None or comp_idx is None:
                continue
            # parse rows, accumulate
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) <= max(prompt_idx, comp_idx):
                    continue
                name = cols[0].get_text(strip=True)
                p_txt = cols[prompt_idx].get_text(strip=True)
                c_txt = cols[comp_idx].get_text(strip=True)
                m1 = re.search(r"\$([\d,]+\.?\d*)", p_txt)
                m2 = re.search(r"\$([\d,]+\.?\d*)", c_txt)
                if not m1 or not m2:
                    continue
                p = float(m1.group(1).replace(',', '')) / 1000
                c = float(m2.group(1).replace(',', '')) / 1000
                pricing[name] = {'prompt': p, 'completion': c}
    except Exception:
        pass
    return pricing

# Fetch dynamic pricing and merge with static fallbacks
dynamic_prices = fetch_all_model_prices()
STATIC_MAP = {
    "GPT-3.5 Turbo":                {'prompt': 0.0015,    'completion': 0.002},
    "GPT-35 Turbo 16K":             {'prompt': 0.003,     'completion': 0.004},
    "GPT-4":                        {'prompt': 0.03,      'completion': 0.06},
    "GPT-4o":                       {'prompt': 0.00275,   'completion': 0.011},
    "GPT-4.5 Preview (2025-02-27 Global)": {'prompt': 75/1000, 'completion': 150/1000},
    "o1 2024-12-17 Global":         {'prompt': 15/1000,   'completion': 60/1000},
    "o3 mini (2025-01-31 Global)":  {'prompt': 1.10/1000, 'completion': 4.40/1000},
    "Computer-Using Agent (CUA)":    {'prompt': 3/1000,    'completion': 12/1000},
    "GPT-4o Realtime Preview (2024-12-17 Global)": {'prompt': 5/1000, 'completion': 20/1000},
    "GPT-4o Mini Realtime Preview (2024-12-17 Global)": {'prompt': 0.60/1000, 'completion': 2.40/1000},
}
PRICING_MAP = {**STATIC_MAP, **dynamic_prices}

# Metadata for top 10 models
MODEL_METADATA = [
    {"Model": "GPT-3.5 Turbo", "Release Date": "2022-11-30", "Popularity": 1},
    {"Model": "GPT-4", "Release Date": "2023-03-14", "Popularity": 2},
    {"Model": "GPT-4o", "Release Date": "2024-05-01", "Popularity": 3},
    {"Model": "GPT-3.5 Turbo 16K", "Release Date": "2023-06-01", "Popularity": 4},
    {"Model": "GPT-4.5 Preview (2025-02-27 Global)", "Release Date": "2025-02-27", "Popularity": 5},
    {"Model": "o3 mini (2025-01-31 Global)", "Release Date": "2025-01-31", "Popularity": 6},
    {"Model": "o1 2024-12-17 Global", "Release Date": "2024-12-17", "Popularity": 7},
    {"Model": "o1 preview (2024-09-12 Global)", "Release Date": "2024-09-12", "Popularity": 8},
    {"Model": "GPT-4o Realtime Preview (2024-12-17 Global)", "Release Date": "2024-12-17", "Popularity": 9},
    {"Model": "GPT-4o Mini Realtime Preview (2024-12-17 Global)","Release Date":"2024-12-17","Popularity":10},
]

# User inputs
title = "Azure AI Model Cost Estimator & Comparison"
st.title(title)
prompt_tokens = st.number_input("Avg prompt tokens per doc", min_value=0, value=2000, step=100)
completion_tokens = st.number_input("Avg completion tokens per doc", min_value=0, value=500, step=100)
docs_per_month = st.number_input("Docs processed per month", min_value=0, value=100, step=1)
model = st.selectbox("Select model for estimate", sorted(PRICING_MAP.keys()))
prompt_price = PRICING_MAP[model]['prompt']
completion_price = PRICING_MAP[model]['completion']

# Single-model estimate
tot_p = prompt_tokens * docs_per_month
tot_c = completion_tokens * docs_per_month
cost_p = tot_p / 1000 * prompt_price
cost_c = tot_c / 1000 * completion_price
total = cost_p + cost_c

st.subheader("Estimated Monthly Cost for Selected Model")
st.markdown(f"- **Model:** {model}")
st.markdown(f"- Prompt tokens: {tot_p:,} → ${cost_p:,.2f}")
st.markdown(f"- Completion tokens: {tot_c:,} → ${cost_c:,.2f}")
st.markdown(f"### **Total: ${total:,.2f}**")

# Export to Excel in a single sheet with inputs, selected model, and comparison
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    sheet_name = 'Report'
    # Write Inputs at the top
    inputs_df = pd.DataFrame({
        'Parameter': ['Model', 'Avg prompt tokens/doc', 'Avg completion tokens/doc', 'Docs/month', 'Prompt price ($/1K)', 'Completion price ($/1K)'],
        'Value': [model, prompt_tokens, completion_tokens, docs_per_month, prompt_price, completion_price]
    })
    inputs_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=0)
    # Write Selected Model Cost below inputs
    cost_df = pd.DataFrame({
        'Parameter': ['Total prompt tokens', 'Total completion tokens', 'Cost for prompts ($)', 'Cost for completions ($)', 'Total estimated cost ($)'],
        'Value': [tot_p, tot_c, round(cost_p,2), round(cost_c,2), round(total,2)]
    })
    start_row = len(inputs_df) + 3  # blank row separation
    cost_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=start_row)
    # Write Comparison table below cost
    df_meta = pd.DataFrame(MODEL_METADATA)
    df_meta['Release Date'] = pd.to_datetime(df_meta['Release Date'])
    prices_df = pd.DataFrame.from_dict(PRICING_MAP, orient='index').reset_index().rename(columns={'index':'Model','prompt':'Prompt Price','completion':'Completion Price'})
    comp_df = df_meta.merge(prices_df, on='Model', how='left')
    comp_df['Cost (Prompt)'] = comp_df['Prompt Price'] * tot_p / 1000
    comp_df['Cost (Completion)'] = comp_df['Completion Price'] * tot_c / 1000
    comp_df['Total Cost'] = comp_df['Cost (Prompt)'] + comp_df['Cost (Completion)']
    comp_df = comp_df.sort_values(['Release Date','Popularity'], ascending=[False,True])
    comp_start = start_row + len(cost_df) + 3  # blank rows
    comp_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=comp_start)
buffer.seek(0)
st.download_button('Download Excel Report', data=buffer, file_name='azure_ai_cost_report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Dynamic comparison table display remains unchanged
("Top 10 Models: Cost Comparison Based on Your Inputs")
st.dataframe(
    comp_df[['Model','Release Date','Popularity','Prompt Price','Completion Price','Cost (Prompt)','Cost (Completion)','Total Cost']]
    .style
    .format({
        'Release Date': lambda x: x.strftime('%Y-%m-%d'),
        'Prompt Price': '{:.2f}',
        'Completion Price': '{:.2f}',
        'Cost (Prompt)': '${:,.2f}'.format,
        'Cost (Completion)': '${:,.2f}'.format,
        'Total Cost': '${:,.2f}'.format
    })
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold'), ('min-width', '120px'), ('text-align', 'left')]},
        {'selector': 'td', 'props': [('min-width', '120px'), ('text-align', 'right')]}    ])
    .set_properties(subset=['Model','Release Date'], **{'text-align':'left'})
    , use_container_width=True
)

st.info("Install dependencies: pip install streamlit pandas beautifulsoup4 && run with `streamlit run cost_estimator.py`. ")


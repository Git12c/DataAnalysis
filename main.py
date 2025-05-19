import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from prophet import Prophet
from data_loader import load_and_merge_data
from preprocessing import preprocess_for_prophet
from prophet_model import train_and_forecast, save_prophet_plots
from dotenv import load_dotenv
import requests

app = FastAPI()

# Enable CORS for Power BI and other clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store uploaded dataframes in memory for demo (use a database for production)
dataframes = {}

dataframes['main'] = load_and_merge_data()

# Prophet model and forecast cache
prophet_model = None
forecast_df = None

# Default user selection (update based on available data)
def get_default_selection(df):
    # Pick first available ProductID and Branch with enough records
    if 'ProductID' in df.columns:
        product_counts = df['ProductID'].value_counts()
        products = product_counts[product_counts >= 60].index.tolist()
        if products:
            product_id = products[0]
        else:
            product_id = df['ProductID'].iloc[0]
    else:
        product_id = None
    if 'Ship Branch' in df.columns:
        branch = df['Ship Branch'].dropna().unique()[0]
    elif 'Branch' in df.columns:
        branch = df['Branch'].dropna().unique()[0]
    else:
        branch = None
    return {'product_id': product_id, 'branch': branch}

# Utility: Prepare Prophet data
def prepare_prophet_data(df):
    # Use the modular preprocessing
    return preprocess_for_prophet(df, product_id=user_selection['product_id'], branch=user_selection['branch'])

# Set user_selection based on available data
df_main = dataframes['main']
user_selection = get_default_selection(df_main)

# Train Prophet model and cache forecast
@app.on_event("startup")
def train_prophet():
    global prophet_model, forecast_df
    df = dataframes['main']
    prophet_data = prepare_prophet_data(df)
    if not prophet_data.empty:
        prophet_model, forecast_df = train_and_forecast(prophet_data, periods=90)
        save_prophet_plots(prophet_model, forecast_df)
    else:
        prophet_model = None
        forecast_df = None

@app.get("/api/data", response_class=JSONResponse)
def api_data():
    df = dataframes['main']
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/api/prophet-forecast", response_class=JSONResponse)
def api_prophet_forecast():
    global forecast_df
    if forecast_df is not None:
        return JSONResponse(content=forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient="records"))
    return JSONResponse(content={"error": "No forecast available"}, status_code=404)

@app.get("/prophet/plot", response_class=HTMLResponse)
def prophet_plot(request: Request):
    # Serve the pre-generated forecast plot image if Prophet model is not available
    import os
    static_path = os.path.join("static", "prophet_forecast.png")
    if os.path.exists(static_path):
        plot_url = "/static/prophet_forecast.png"
        return templates.TemplateResponse("index.html", {"request": request, "plot_url": plot_url})
    # fallback to old logic
    global prophet_model, forecast_df
    if prophet_model is None or forecast_df is None:
        return templates.TemplateResponse("index.html", {"request": request, "upload_message": "No forecast available."})
    fig = prophet_model.plot(forecast_df)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plot_url = f"data:image/png;base64,{img_base64}"
    return templates.TemplateResponse("index.html", {"request": request, "plot_url": plot_url})

@app.get("/prophet/components", response_class=HTMLResponse)
def prophet_components(request: Request):
    # Serve the pre-generated components plot image if Prophet model is not available
    import os
    static_path = os.path.join("static", "prophet_components.png")
    if os.path.exists(static_path):
        plot_url = "/static/prophet_components.png"
        return templates.TemplateResponse("index.html", {"request": request, "plot_url": plot_url})
    # fallback to old logic
    global prophet_model, forecast_df
    if prophet_model is None or forecast_df is None:
        return templates.TemplateResponse("index.html", {"request": request, "upload_message": "No forecast available."})
    fig = prophet_model.plot_components(forecast_df)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plot_url = f"data:image/png;base64,{img_base64}"
    return templates.TemplateResponse("index.html", {"request": request, "plot_url": plot_url})

@app.get("/data-table", response_class=HTMLResponse)
def data_table(request: Request):
    df = dataframes['main']
    if df.empty:
        table_html = '<div class="alert alert-warning">No data available.</div>'
    else:
        table_html = df.head(100).to_html(classes='table table-striped', index=False)
    return templates.TemplateResponse("index.html", {"request": request, "table_html": table_html})

# Setup templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def main_ui(request: Request):
    df = dataframes['main']
    # Only show ProductIDs with at least 60 records
    product_counts = df['ProductID'].value_counts()
    products = sorted(product_counts[product_counts >= 60].index.tolist()) if 'ProductID' in df.columns else []
    branches = sorted(df['Ship Branch'].dropna().unique()) if 'Ship Branch' in df.columns else []
    return templates.TemplateResponse("index.html", {
        "request": request,
        "branches": branches,
        "products": products,
        "selected_branch": user_selection['branch'],
        "selected_product": user_selection['product_id'],
        "show_selection_form": True
    })

@app.get("/select", response_class=HTMLResponse)
def select_form(request: Request):
    df = dataframes['main']
    product_counts = df['ProductID'].value_counts()
    products = sorted(product_counts[product_counts >= 60].index.tolist()) if 'ProductID' in df.columns else []
    branches = sorted(df['Ship Branch'].dropna().unique()) if 'Ship Branch' in df.columns else []
    return templates.TemplateResponse("index.html", {
        "request": request,
        "branches": branches,
        "products": products,
        "selected_branch": user_selection['branch'],
        "selected_product": user_selection['product_id'],
        "show_selection_form": True
    })

@app.post("/select", response_class=HTMLResponse)
def update_selection(request: Request, product_id: int = Form(...), branch: str = Form(...)):
    user_selection['product_id'] = product_id
    user_selection['branch'] = branch
    # Re-run model and update plots
    df = dataframes['main']
    prophet_data = preprocess_for_prophet(df, product_id=product_id, branch=branch)
    global prophet_model, forecast_df
    if not prophet_data.empty:
        prophet_model, forecast_df = train_and_forecast(prophet_data, periods=90)
        save_prophet_plots(prophet_model, forecast_df)
        message = f"Model updated for Product ID {product_id} and Branch '{branch}'."
    else:
        prophet_model = None
        forecast_df = None
        message = "No data available for the selected Product ID and Branch."
    branches = sorted(df['Ship Branch'].dropna().unique())
    products = sorted(df['ProductID'].dropna().unique())
    return templates.TemplateResponse("index.html", {
        "request": request,
        "branches": branches,
        "products": products,
        "selected_branch": branch,
        "selected_product": product_id,
        "upload_message": message,
        "show_selection_form": True
    })

# Helper to robustly get the correct manager/head name
# Always returns a non-empty string (never blank or N/A)
def get_manager_or_head(subdf, colname, global_mode=None):
    if colname in subdf.columns and not subdf[colname].isnull().all():
        vals = subdf[colname].dropna()
        if not vals.empty:
            mode = vals.mode()
            if not mode.empty and pd.notnull(mode[0]) and str(mode[0]).strip():
                return str(mode[0]).strip()
            # fallback to first non-null
            for v in vals:
                if pd.notnull(v) and str(v).strip():
                    return str(v).strip()
    # fallback to global mode if provided
    if global_mode:
        return global_mode
    return ""

@app.get("/train/all", response_class=HTMLResponse)
def train_all_report(request: Request):
    import pandas as pd
    df = dataframes['main']
    if df.empty:
        return HTMLResponse("<div class='alert alert-warning'>No data available for consolidated training.</div>")
    results = []
    product_ids = sorted(df['ProductID'].unique())
    branches = sorted(df['Ship Branch'].unique())
    # Compute global modes for fallback
    global_mgr = get_manager_or_head(df, 'Regional Manager')
    global_head = get_manager_or_head(df, 'Sales Head')
    for pid in product_ids:
        for branch in branches:
            subdf = df[(df['ProductID'] == pid) & (df['Ship Branch'] == branch)]
            if len(subdf) < 10:
                continue
            prophet_data = preprocess_for_prophet(df, product_id=pid, branch=branch)
            if prophet_data.empty:
                continue
            model, forecast = train_and_forecast(prophet_data, periods=90)
            total_forecast = forecast['yhat'].sum()
            avg_forecast = forecast['yhat'].mean()
            # --- Robust manager/head extraction: always use most frequent non-null in subdf, fallback to first non-null, then global ---
            def robust_name(subdf, col, global_mode):
                vals = subdf[col].dropna() if col in subdf.columns else pd.Series(dtype=str)
                if not vals.empty:
                    mode = vals.mode()
                    if not mode.empty and pd.notnull(mode[0]) and str(mode[0]).strip():
                        return str(mode[0]).strip()
                    for v in vals:
                        if pd.notnull(v) and str(v).strip():
                            return str(v).strip()
                if global_mode:
                    return global_mode
                return ""
            regional_manager = robust_name(subdf, 'Regional Manager', global_mgr)
            sales_head = robust_name(subdf, 'Sales Head', global_head)
            results.append({
                'ProductID': pid,
                'Branch': branch,
                'TotalForecast': total_forecast,
                'AvgForecast': avg_forecast,
                'RegionalManager': regional_manager,
                'SalesHead': sales_head
            })
    if not results:
        return HTMLResponse("<div class='alert alert-warning'>No sufficient data for consolidated report.</div>")
    results_df = pd.DataFrame(results)
    # Pivot table with units in column heading and full border
    pivot = results_df.pivot(index='ProductID', columns='Branch', values='TotalForecast').fillna(0)
    pivot.columns = [f"{col} (units)" for col in pivot.columns]
    pivot_html = pivot.to_html(classes='table table-bordered table-sm', border=2, justify='center')
    # Overall summary
    overall_total = results_df['TotalForecast'].sum()
    overall_avg = results_df['AvgForecast'].mean()
    # Only consider rows with manager and head names for best/worst
    valid_results = results_df[(results_df['RegionalManager'] != '') & (results_df['SalesHead'] != '')]
    if not valid_results.empty:
        best_row = valid_results.loc[valid_results['TotalForecast'].idxmax()]
        worst_row = valid_results.loc[valid_results['TotalForecast'].idxmin()]
    else:
        best_row = results_df.loc[results_df['TotalForecast'].idxmax()]
        worst_row = results_df.loc[results_df['TotalForecast'].idxmin()]
    improvement_needed = best_row['TotalForecast'] - worst_row['TotalForecast']
    improvement_pct = (improvement_needed / best_row['TotalForecast']) * 100 if best_row['TotalForecast'] else 0
    insight = f"""
    <b>Overall Insight:</b> The total forecasted sales across all products and branches is <b>{overall_total:,.2f} units</b>.<br>
    The average forecast per product-branch pair is <b>{overall_avg:,.2f} units</b>.<br>
    <b>Best performing:</b> ProductID <b>{best_row['ProductID']}</b> in <b>{best_row['Branch']}</b>{f" (Regional Manager: <b>{best_row['RegionalManager']}</b>, Sales Head: <b>{best_row['SalesHead']}</b>)" if best_row['RegionalManager'] or best_row['SalesHead'] else ''} with <b>{best_row['TotalForecast']:,.2f} units</b>.<br>
    <b>Lowest performing:</b> ProductID <b>{worst_row['ProductID']}</b> in <b>{worst_row['Branch']}</b>{f" (Regional Manager: <b>{worst_row['RegionalManager']}</b>, Sales Head: <b>{worst_row['SalesHead']}</b>)" if worst_row['RegionalManager'] or worst_row['SalesHead'] else ''} with <b>{worst_row['TotalForecast']:,.2f} units</b>.<br>
    <b>Improvement Opportunity:</b> To bring the lowest performing cell up to the best, an increase of <b>{improvement_needed:,.2f} units</b> ({improvement_pct:.1f}%) is needed.<br>
    <b>Actionable Recommendation:</b> Focus on <b>{worst_row['Branch']}</b>{f" (Regional Manager: <b>{worst_row['RegionalManager']}</b>, Sales Head: <b>{worst_row['SalesHead']}</b>)" if worst_row['RegionalManager'] or worst_row['SalesHead'] else ''} for targeted training, marketing, and resource allocation. Analyze local challenges, customer feedback, and sales process bottlenecks. Consider incentive programs and best-practice sharing from top-performing branches.<br>
    <b>Additional Insights:</b> If the bottom 25% of branches/products are improved by just 10%, overall sales could increase by <b>{(results_df.nsmallest(max(1, len(results_df)//4), 'TotalForecast')['TotalForecast'].sum() * 0.1):,.2f} units</b>.<br>
    """
    html = f'''
    <div class="alert alert-info"><b>Consolidated Forecast Report (All Products & Branches)</b></div>
    <b>Pivot Table (Total Forecasted Sales):</b>
    <div class="table-responsive">{pivot_html}</div>
    <div class="alert alert-success">{insight}</div>
    '''
    return HTMLResponse(html)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AUTOGEN_API_KEY = os.getenv("AUTOGEN_API_KEY")

# Helper to call Gemini 1.5 Flash for GenAI insights
def get_genai_insights(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "<div class='alert alert-warning'>GEMINI_API_KEY not set in .env</div>"
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 512}
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        # Extract HTML from the response
        html = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return html
    except Exception as e:
        return f"<div class='alert alert-danger'>GenAI error: {str(e)}</div>"

@app.get("/genai/insights", response_class=HTMLResponse)
def genai_insights(request: Request):
    import pandas as pd
    global forecast_df
    if forecast_df is None or forecast_df.empty:
        return HTMLResponse("<div class='alert alert-warning'>No forecast available for insights.</div>")
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    forecast_df['month'] = forecast_df['ds'].dt.to_period('M')
    next_8_months = forecast_df[forecast_df['ds'] > pd.Timestamp.today()].groupby('month').tail(1).head(8)
    month_end_table = next_8_months[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    month_end_table['ds'] = month_end_table['ds'].dt.strftime('%B %Y')
    month_end_table = month_end_table.rename(columns={
        'ds': 'Month End', 'yhat': 'Forecast (Sales)', 'yhat_lower': 'Lower Bound (Sales)', 'yhat_upper': 'Upper Bound (Sales)'
    })
    month_end_table = month_end_table[['Month End', 'Forecast (Sales)', 'Lower Bound (Sales)', 'Upper Bound (Sales)']]
    month_end_html = month_end_table.to_html(index=False, classes='table table-bordered table-sm text-center align-middle', border=0, justify='center')
    # Executive summary with branch/manager/head details
    df = dataframes['main']
    # Compute global modes for fallback
    global_mgr = get_manager_or_head(df, 'Regional Manager')
    global_head = get_manager_or_head(df, 'Sales Head')
    branch_summary = []
    for branch in df['Ship Branch'].unique():
        branch_df = df[df['Ship Branch'] == branch]
        total_sales = branch_df['Ext Total Sales'].sum()
        regional_manager = get_manager_or_head(branch_df, 'Regional Manager', global_mgr)
        sales_head = get_manager_or_head(branch_df, 'Sales Head', global_head)
        branch_summary.append({'Branch': branch, 'TotalSales': total_sales, 'RegionalManager': regional_manager, 'SalesHead': sales_head})
    branch_summary = [b for b in branch_summary if b['RegionalManager'] and b['SalesHead']]
    branch_summary = sorted(branch_summary, key=lambda x: x['TotalSales'], reverse=True)
    best_branch = branch_summary[0]
    worst_branch = branch_summary[-1]
    # Compose friendly, clear HTML
    html = f'''
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: auto;">
      <div class="alert alert-info text-center" style="font-size:1.3rem;font-weight:bold;">Executive Forecast Insights</div>
      <ul style="font-size:1.08rem;line-height:1.7;">
        <li><b>Total forecasted sales (full period):</b> <span style="color:#007bff">{forecast_df['yhat'].sum():,.2f} units</span></li>
        <li><b>Average monthly forecasted sales:</b> <span style="color:#007bff">{forecast_df['yhat'].mean():,.2f} units</span></li>
        <li><b>Best performing branch:</b> <span style="color:#28a745">{best_branch['Branch']}</span> (Regional Manager: <b>{best_branch['RegionalManager']}</b>, Sales Head: <b>{best_branch['SalesHead']}</b>) with <b>{best_branch['TotalSales']:,.2f} units</b> in sales.</li>
        <li><b>Lowest performing branch:</b> <span style="color:#dc3545">{worst_branch['Branch']}</span> (Regional Manager: <b>{worst_branch['RegionalManager']}</b>, Sales Head: <b>{worst_branch['SalesHead']}</b>) with <b>{worst_branch['TotalSales']:,.2f} units</b> in sales.</li>
      </ul>
      <div style="font-size:1.08rem;">
        <b>Suggestions for Improvement:</b>
        <ul style="margin-bottom:0.5rem;">
          <li>Encourage <b>{worst_branch['RegionalManager']}</b> and <b>{worst_branch['SalesHead']}</b> to review successful strategies from <b>{best_branch['Branch']}</b> and adapt them locally.</li>
          <li>Consider targeted training, marketing campaigns, and customer engagement in <b>{worst_branch['Branch']}</b> to boost performance.</li>
          <li>Foster collaboration between branch heads to share best practices and address unique challenges.</li>
          <li>Monitor progress monthly and celebrate improvements to motivate teams.</li>
        </ul>
      </div>
      <div style="margin:1.2rem 0 0.7rem 0;">
        <b>Month-End Forecast Table (Next 8 Months):</b>
        <div class="table-responsive" style="font-size:1.05rem;">{month_end_html}</div>
      </div>
    </div>
    '''
    return HTMLResponse(html)

@app.get("/api/manager-head", response_class=JSONResponse)
def get_manager_head(product_id: int, branch: str):
    df = dataframes['main']
    subdf = df[(df['ProductID'] == product_id) & (df['Ship Branch'] == branch)]
    # Use robust logic: most frequent non-null, then first non-null, then empty string
    def robust_name(subdf, col):
        vals = subdf[col].dropna() if col in subdf.columns else pd.Series(dtype=str)
        if not vals.empty:
            mode = vals.mode()
            if not mode.empty and pd.notnull(mode[0]) and str(mode[0]).strip():
                return str(mode[0]).strip()
            for v in vals:
                if pd.notnull(v) and str(v).strip():
                    return str(v).strip()
        return ""
    regional_manager = robust_name(subdf, 'Regional Manager')
    sales_head = robust_name(subdf, 'Sales Head')
    return JSONResponse({
        'RegionalManager': regional_manager,
        'SalesHead': sales_head
    })

@app.get("/genai/consolidated-insights", response_class=HTMLResponse)
def genai_consolidated_insights(request: Request):
    import pandas as pd
    df = dataframes['main']
    if df.empty:
        return HTMLResponse("<div class='alert alert-warning'>No data available for insights.</div>")
    product_ids = sorted(df['ProductID'].unique())
    branches = sorted(df['Ship Branch'].unique())
    summary_rows = []
    for pid in product_ids:
        for branch in branches:
            subdf = df[(df['ProductID'] == pid) & (df['Ship Branch'] == branch)]
            if subdf.empty:
                continue
            # Use the new robust endpoint logic directly
            def robust_name(subdf, col):
                vals = subdf[col].dropna() if col in subdf.columns else pd.Series(dtype=str)
                if not vals.empty:
                    mode = vals.mode()
                    if not mode.empty and pd.notnull(mode[0]) and str(mode[0]).strip():
                        return str(mode[0]).strip()
                    for v in vals:
                        if pd.notnull(v) and str(v).strip():
                            return str(v).strip()
                return ""
            regional_manager = robust_name(subdf, 'Regional Manager')
            sales_head = robust_name(subdf, 'Sales Head')
            total_sales = subdf['Ext Total Sales'].sum()
            summary_rows.append({
                'ProductID': pid,
                'Branch': branch,
                'RegionalManager': regional_manager,
                'SalesHead': sales_head,
                'TotalSales': total_sales
            })
    if not summary_rows:
        return HTMLResponse("<div class='alert alert-warning'>No data available for consolidated insights.</div>")
    summary_df = pd.DataFrame(summary_rows)
    # Pivot for display
    pivot = summary_df.pivot(index='ProductID', columns='Branch', values='TotalSales').fillna(0)
    pivot_html = pivot.to_html(classes='table table-bordered table-sm', border=0, justify='center')
    # Executive summary
    html = f'''
    <div class="alert alert-info"><b>Consolidated GenAI Executive Insights (All Products & Branches)</b></div>
    <b>Pivot Table (Total Sales):</b>
    <div class="table-responsive">{pivot_html}</div>
    <div class="alert alert-success">
    <ul>
    '''
    for row in summary_rows:
        html += f'<li>ProductID <b>{row["ProductID"]}</b> in <b>{row["Branch"]}</b>: Regional Manager: <b>{row["RegionalManager"]}</b>, Sales Head: <b>{row["SalesHead"]}</b>, Total Sales: <b>{row["TotalSales"]:,.2f}</b></li>'
    html += '</ul></div>'
    return HTMLResponse(html)

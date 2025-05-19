import os
import matplotlib
matplotlib.use('Agg')
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
from agentic_genai import run_sales_insight_agent
import asyncio

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
    global prophet_model, forecast_df
    if prophet_model is None or forecast_df is None:
        return HTMLResponse("<div class='alert alert-warning'>No forecast available.</div>")
    fig = prophet_model.plot(forecast_df)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    img_src = f"data:image/png;base64,{img_base64}"
    return templates.TemplateResponse("plain.html", {"request": request, "img_src": img_src})

@app.get("/prophet/components", response_class=HTMLResponse)
def prophet_components(request: Request):
    global prophet_model, forecast_df
    if prophet_model is None or forecast_df is None:
        return HTMLResponse("<div class='alert alert-warning'>No forecast available.</div>")
    fig = prophet_model.plot_components(forecast_df)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    img_src = f"data:image/png;base64,{img_base64}"
    return templates.TemplateResponse("plain.html", {"request": request, "img_src": img_src})

@app.get("/prophet/forecast-plot")
def prophet_forecast_plot():
    global prophet_model, forecast_df
    if prophet_model is None or forecast_df is None:
        return JSONResponse(content={"error": "No forecast available"}, status_code=404)
    fig = prophet_model.plot(forecast_df)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return JSONResponse(content={"image_base64": img_base64})

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

# Helper to call Gemini 1.5 Flash for GenAI insights using agentic approach
async def get_genai_insights_agentic(prompt: str) -> str:
    return await run_sales_insight_agent(prompt)

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
    best_branch = branch_summary[0] if branch_summary else {'Branch': '', 'RegionalManager': '', 'SalesHead': '', 'TotalSales': 0}
    worst_branch = branch_summary[-1] if branch_summary else {'Branch': '', 'RegionalManager': '', 'SalesHead': '', 'TotalSales': 0}
    prompt = (
        "You are an executive sales analyst. "
        "Given the following forecast summary and table, provide a clear, actionable, executive-level insight. "
        "Focus on improvement opportunities, strengths, and risks. "
        "Here is the summary:\n"
        f"Total forecasted sales: {forecast_df['yhat'].sum():,.2f} units. "
        f"Average monthly forecast: {forecast_df['yhat'].mean():,.2f} units. "
        f"Best branch: {best_branch['Branch']} (Manager: {best_branch['RegionalManager']}, Head: {best_branch['SalesHead']}) with {best_branch['TotalSales']:,.2f} units. "
        f"Worst branch: {worst_branch['Branch']} (Manager: {worst_branch['RegionalManager']}, Head: {worst_branch['SalesHead']}) with {worst_branch['TotalSales']:,.2f} units. "
        "Month-end forecast table:\n"
        f"{month_end_table.to_string(index=False)}"
    )
    # Run the agentic GenAI insight generator
    try:
        genai_html = asyncio.run(get_genai_insights_agentic(prompt))
    except Exception as e:
        genai_html = f"<div class='alert alert-danger'>GenAI agentic service error: {str(e)}</div>"
    html = f"""
    <div class='alert alert-secondary'><b>GenAI Prompt (editable in code):</b><br><pre style='white-space:pre-wrap;font-size:1rem;background:#f8f9fa;border:1px solid #ccc;padding:0.5em'>{prompt}</pre></div>
    <div>{genai_html}</div>
    """
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

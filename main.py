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
from data_loader import load_and_merge_data, get_productid_to_name_mapping
from preprocessing import preprocess_for_prophet
from prophet_model import train_and_forecast, save_prophet_plots
from dotenv import load_dotenv
import requests
from agentic_genai import get_gemini_insight
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
# Ensure required columns exist to prevent KeyError
if 'ProductID' not in dataframes['main'].columns:
    dataframes['main']['Product'] = ''
if 'Ship Branch' not in dataframes['main'].columns:
    dataframes['main']['Ship Branch'] = ''

# Prophet model and forecast cache
prophet_model = None
forecast_df = None

# Load ProductID-to-name mapping
productid_to_name = get_productid_to_name_mapping()
name_to_productid = {v: k for k, v in productid_to_name.items()}

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
    if (prophet_model is None) or (forecast_df is None):
        return HTMLResponse("<div class='alert alert-warning'>No forecast available.</div>")
    # Generate the plot and return as an image in HTML
    buf = io.BytesIO()
    fig = prophet_model.plot(forecast_df)
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    img_src = f"data:image/png;base64,{img_base64}"
    return templates.TemplateResponse("plain.html", {"request": request, "img_src": img_src})

@app.get("/prophet/components", response_class=HTMLResponse)
def prophet_components(request: Request):
    global prophet_model, forecast_df
    if (prophet_model is None) or (forecast_df is None):
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
    if (prophet_model is None) or (forecast_df is None):
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
    # Map ProductIDs to names for display
    product_names = [productid_to_name.get(pid, str(pid)) for pid in products]
    branches = sorted(df['Ship Branch'].dropna().unique()) if 'Ship Branch' in df.columns else []
    selected_product_name = productid_to_name.get(user_selection['product_id'], str(user_selection['product_id']))
    return templates.TemplateResponse("index.html", {
        "request": request,
        "branches": branches,
        "products": product_names,
        "selected_branch": user_selection['branch'],
        "selected_product": selected_product_name,
        "show_selection_form": True,
        "productid_to_name": productid_to_name
    })

@app.get("/select", response_class=HTMLResponse)
def select_form(request: Request):
    df = dataframes['main']
    product_counts = df['ProductID'].value_counts()
    products = sorted(product_counts[product_counts >= 60].index.tolist()) if 'ProductID' in df.columns else []
    product_names = [productid_to_name.get(pid, str(pid)) for pid in products]
    branches = sorted(df['Ship Branch'].dropna().unique()) if 'Ship Branch' in df.columns else []
    selected_product_name = productid_to_name.get(user_selection['product_id'], str(user_selection['product_id']))
    return templates.TemplateResponse("index.html", {
        "request": request,
        "branches": branches,
        "products": product_names,
        "selected_branch": user_selection['branch'],
        "selected_product": selected_product_name,
        "show_selection_form": True,
        "productid_to_name": productid_to_name
    })

@app.post("/select", response_class=HTMLResponse)
def update_selection(request: Request, product_id: str = Form(...), branch: str = Form(...)):
    # product_id is now product name from the form
    pid = name_to_productid.get(product_id, None)
    user_selection['product_id'] = pid
    user_selection['branch'] = branch
    # Re-run model and update plots
    df = dataframes['main']
    prophet_data = preprocess_for_prophet(df, product_id=pid, branch=branch)
    global prophet_model, forecast_df
    if not prophet_data.empty:
        prophet_model, forecast_df = train_and_forecast(prophet_data, periods=90)
        save_prophet_plots(prophet_model, forecast_df)
        message = f"Model updated for Product '{product_id}' and Branch '{branch}'."
    else:
        prophet_model = None
        forecast_df = None
        message = "No data available for the selected Product and Branch."
    branches = sorted(df['Ship Branch'].dropna().unique())
    product_counts = df['ProductID'].value_counts()
    products = sorted(product_counts[product_counts >= 60].index.tolist()) if 'ProductID' in df.columns else []
    product_names = [productid_to_name.get(pid, str(pid)) for pid in products]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "branches": branches,
        "products": product_names,
        "selected_branch": branch,
        "selected_product": product_id,
        "upload_message": message,
        "show_selection_form": True,
        "productid_to_name": productid_to_name
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
    import os
    import pandas as pd
    forecast6_dir = os.path.join(os.getcwd(), 'forecast6')
    summary_csv = os.path.join(forecast6_dir, "forecast_6y_summary.csv")
    # Try to generate summary table if missing
    if not os.path.exists(summary_csv):
        try:
            _ = api_prophet_forecast_6y()
        except Exception:
            pass
    # If still missing, fallback to using main data table
    if not os.path.exists(summary_csv):
        # Fallback: use /data-table (main data) to generate a minimal insight
        df = dataframes['main']
        if df.empty:
            return HTMLResponse("<div class='alert alert-danger'>No data available to generate insights.</div>")
        # Group by Product, Branch, Region, SalesOffice, Regional Manager, Sales Head
        group_cols = ['Product', 'Ship Branch', 'Region', 'SalesOffice', 'Regional Manager', 'Sales Head']
        for col in group_cols:
            if col not in df.columns:
                df[col] = ''
        summary = df.groupby(group_cols)['Price'].sum().reset_index().rename(columns={'Ship Branch': 'Branch', 'Price': 'TotalSales'})
        total_sales = summary['TotalSales'].sum()
        summary['PercentOfTotal'] = summary['TotalSales'] / total_sales * 100 if total_sales else 0
        # Find lowest performing (product, branch) pairs
        summary = summary.sort_values('TotalSales')
        html = """
        <h3>Fallback Executive Summary: Sales Insights</h3>
        <ul>"""
        for _, row in summary.head(5).iterrows():
            html += f"<li>Product: <b>{row['Product']}</b> in <b>{row['Region']}</b> region (Branch: <b>{row['Branch']}</b>, Sales Office: <b>{row['SalesOffice']}</b>) has low sales: <b>{row['TotalSales']:,.0f}</b> (<b>{row['PercentOfTotal']:.2f}%</b> of total). Responsible: Regional Manager: <b>{row['Regional Manager']}</b>, Sales Head: <b>{row['Sales Head']}</b>.</li>"
        html += "</ul>"
        html += "<b>Recommendation:</b> Focus on improving sales for the above product/region/branch combinations. Regional Managers and Sales Heads should review sales strategies and marketing efforts."
        return HTMLResponse(html)
    # Normal path: summary table exists
    summary_df = pd.read_csv(summary_csv)
    required_cols = ['Product', 'Branch', 'ForecastedSales', 'Target', 'Difference', 'PercentDiff',
                     'Regional Manager', 'Sales Head', 'Region', 'SalesOffice']
    for col in required_cols:
        if col not in summary_df.columns:
            summary_df[col] = ''
    # Find lagging areas
    lagging = []
    for _, row in summary_df.iterrows():
        if pd.notnull(row['Difference']) and row['Difference'] < 0:
            lagging.append((row['Product'], row['Branch'], abs(row['Difference']), abs(row['PercentDiff']),
                            row['Regional Manager'], row['Sales Head'], row['Region'], row['SalesOffice']))
    # Group lagging by branch and product
    lagging_by_branch = {}
    lagging_by_product = {}
    for pname, branch, diff, pct, regional_manager, sales_head, region, sales_office in lagging:
        lagging_by_branch.setdefault(branch, []).append((pname, diff, pct, regional_manager, sales_head, region, sales_office))
        lagging_by_product.setdefault(pname, []).append((branch, diff, pct, regional_manager, sales_head, region, sales_office))
    html = """
    <h3>Lagging Areas and Recommendations</h3>
    <b>East Branch:</b> This branch faces substantial challenges across multiple products.<ul>
    """
    for pname, diff, pct, regional_manager, sales_head, region, sales_office in lagging_by_branch.get('East', []):
        html += f"<li>{pname} is forecasted to be <b>{diff:,.0f}</b> below target (<b>{pct:.1f}%</b> shortfall). Responsible: Regional Manager: <b>{regional_manager}</b>, Sales Head: <b>{sales_head}</b>, Region: <b>{region}</b>, Sales Office: <b>{sales_office}</b></li>"
    html += "</ul>"
    html += "<b>Software Product Line:</b> The Software product line performance by branch:<ul>"
    for branch in ['East', 'South', 'West', 'North']:
        for pname, diff, pct, regional_manager, sales_head, region, sales_office in lagging_by_branch.get(branch, []):
            if pname == 'Software':
                html += f"<li>{branch} branch is <b>{diff:,.0f}</b> below target (<b>{pct:.1f}%</b> shortfall). Responsible: Regional Manager: <b>{regional_manager}</b>, Sales Head: <b>{sales_head}</b>, Region: <b>{region}</b>, Sales Office: <b>{sales_office}</b></li>"
    html += "</ul>"
    html += "<b>Recommendation:</b> Re-evaluate the Software product's pricing, features, and marketing strategy, particularly in the East, South and West branches. A potential price adjustment, product enhancement, or new marketing campaign may be required to increase sales in these locations.<br>"
    html += "<b>Support Product Line:</b> The Support product line underperforms expectations in these branches:<ul>"
    for branch in ['East', 'South', 'West', 'North']:
        for pname, diff, pct, regional_manager, sales_head, region, sales_office in lagging_by_branch.get(branch, []):
            if pname == 'Support':
                html += f"<li>{branch} branch is <b>{diff:,.0f}</b> below target (<b>{pct:.1f}%</b> shortfall). Responsible: Regional Manager: <b>{regional_manager}</b>, Sales Head: <b>{sales_head}</b>, Region: <b>{region}</b>, Sales Office: <b>{sales_office}</b></li>"
    html += "</ul>"
    html += "<b>Recommendation:</b> Analyze customer feedback and sales data to understand why the Support product is underperforming. Determine whether marketing or other adjustments are needed to address the shortfall."
    return HTMLResponse(html)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.get("/genai/insights", response_class=HTMLResponse)
def genai_insights(request: Request):
    import google.generativeai as genai
    selected_pid = user_selection.get('product_id')
    selected_branch = user_selection.get('branch')
    selected_product = productid_to_name.get(selected_pid, str(selected_pid))
    # Get the future forecast for the selected product/branch
    df = dataframes['main']
    prophet_data = preprocess_for_prophet(df, product_id=selected_pid, branch=selected_branch)
    if prophet_data.empty:
        return HTMLResponse("<div class='alert alert-danger'>No data available for the selected product and branch.</div>")
    model, forecast = train_and_forecast(prophet_data, periods=90)
    # Group by SalesOffice, Regional Manager, Sales Head, Region, Branch, Product
    group_cols = ['Product', 'Branch', 'Region', 'SalesOffice', 'Regional Manager', 'Sales Head']
    # Attach extra columns if missing
    for col in group_cols:
        if col not in forecast.columns:
            forecast[col] = prophet_data[col].iloc[-1] if col in prophet_data.columns and not prophet_data[col].empty else ''
    # Use only future rows (after last date in prophet_data)
    last_train_date = prophet_data['ds'].max()
    future_forecast = forecast[forecast['ds'] > last_train_date]
    summary = future_forecast.groupby(group_cols)['yhat'].sum().reset_index().rename(columns={'yhat': 'TotalSales'})
    total_sales = summary['TotalSales'].sum()
    summary['PercentOfTotal'] = summary['TotalSales'] / total_sales * 100 if total_sales else 0
    summary = summary.sort_values('TotalSales')
    # Compose executive summary in template format
    summary_lines = []
    for _, row in summary.iterrows():
        summary_lines.append(
            f"Product: <b>{row['Product']}</b> in <b>{row['Region']}</b> region (Branch: <b>{row['Branch']}</b>, Sales Office: <b>{row['SalesOffice']}</b>) has sales: <b>{row['TotalSales']:,.0f}</b> (<b>{row['PercentOfTotal']:.2f}%</b> of this product/branch). Responsible: Regional Manager: <b>{row['Regional Manager']}</b>, Sales Head: <b>{row['Sales Head']}</b>."
        )
    # Fix prompt construction: use only f-string and concatenation, not mixed triple-quote and +
    prompt = (
        f"You are a business analyst. Given the following sales forecast data for product {selected_product} in branch {selected_branch}, generate an executive summary in HTML. Use this template:\n"
        f"<h3>Executive Summary: Sales Insights for {selected_product} ({selected_branch})</h3>\n"
        "<ul>\n"
        + "\n".join([f"<li>{line}</li>" for line in summary_lines]) +
        "\n</ul>\n"
        "<b>Recommendation:</b> Focus on improving sales for the above region/office combinations. Regional Managers and Sales Heads should review sales strategies and marketing efforts."
    )
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        html = response.text.replace('```html', '').replace('```', '')
    except Exception as e:
        html = f"<div class='alert alert-danger'>GenAI error: {e}</div>"
    return HTMLResponse(html)

@app.get("/genai/consolidated-insights", response_class=HTMLResponse)
def genai_consolidated_insights(request: Request):
    import os
    import google.generativeai as genai
    from PIL import Image
    import matplotlib.pyplot as plt
    global prophet_model, forecast_df
    df = dataframes['main']
    # Generate and save multiple images for consolidated insights
    image_paths = []
    product_ids = sorted(df['ProductID'].unique())
    branches = sorted(df['Ship Branch'].unique())
    for pid in product_ids:
        for branch in branches:
            subdf = df[(df['ProductID'] == pid) & (df['Ship Branch'] == branch)]
            if subdf.empty:
                continue
            prophet_data = preprocess_for_prophet(df, product_id=pid, branch=branch)
            if prophet_data.empty:
                continue
            model, forecast = train_and_forecast(prophet_data, periods=90)
            fig = model.plot(forecast)
            img_path = f"prophet_forecast_{pid}_{branch}.png"
            fig.savefig(img_path, format='png')
            plt.close(fig)
            image_paths.append((img_path, pid, branch))
    if not image_paths:
        return HTMLResponse("<div class='alert alert-warning'>No forecast images available for consolidated insights.</div>")
    imgs = []
    img_labels = []
    for img_path, pid, branch in image_paths:
        try:
            imgs.append(Image.open(img_path))
            img_labels.append(f"Product: {productid_to_name.get(pid, pid)} - Branch {branch}")
        except FileNotFoundError:
            continue
    if not imgs:
        return HTMLResponse("<div class='alert alert-danger'>No forecast images could be loaded for insights.</div>")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(
            contents=imgs + [
                (
                    "You are a professional sales executive. Analyze these sales forecast plots and provide a consolidated, structured, executive-level business insight in HTML. "
                    "Each plot corresponds to: " + ", ".join(img_labels) + ". "
                    "Focus only on future prospects, sales trends, growth or decline, and actionable recommendations for business strategy. "
                    "Use only sales numbers and percentages that are clearly present in the data. "
                    "Never mention technical or modeling details, N/A, 0%, missing/zero/uncertain/unavailable values, or anything you cannot see in the data. "
                    "Do not discuss how to improve the model or data. "
                    "Never mention the years explicitly or use hardcoded year values. "
                    "Never mention the images or the process of analysis. "
                    "Use bold for key numbers, bullet points for recommendations, and clear sections. "
                    "Do not show the images, only the insights."
                )
            ]
        )
        insights_html = response.text.replace('```html', '').replace('```', '')
    except Exception as e:
        return HTMLResponse(f"<div class='alert alert-danger'>An error occurred during content generation: {e}</div>")
    return HTMLResponse(insights_html)

@app.get("/api/manager-head", response_class=JSONResponse)
def get_manager_head(product_id: str, branch: str):
    # Accept product name, convert to ProductID
    pid = name_to_productid.get(product_id, None)
    df = dataframes['main']
    subdf = df[(df['ProductID'] == pid) & (df['Ship Branch'] == branch)]
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

# Store 6-year forecasts for all product/branch pairs
six_year_forecasts = dict()

# Hardcoded target values for each (product_id, branch) pair
# Example: {(product_id, branch): target_value}
target_values = {
    (1000, 'East'): 500000,
    (1000, 'West'): 450000,
    (1000, 'North'): 400000,
    (1000, 'South'): 550000,
    (1001, 'East'): 600000,
    (1001, 'West'): 500000,
    (1001, 'North'): 480000,
    (1001, 'South'): 620000,
    (1002, 'East'): 700000,
    (1002, 'West'): 650000,
    (1002, 'North'): 600000,
    (1002, 'South'): 720000,
    (1003, 'East'): 800000,
    (1003, 'West'): 750000,
    (1003, 'North'): 700000,
    (1003, 'South'): 820000,
}

@app.get("/api/prophet-forecast-6y", response_class=JSONResponse)
def api_prophet_forecast_6y():
    import os
    import pandas as pd
    from prophet import Prophet
    from preprocessing import preprocess_for_prophet
    from prophet_model import train_and_forecast
    df = dataframes['main']
    forecasts = dict()
    product_ids = sorted(df['ProductID'].unique())
    branches = sorted(df['Ship Branch'].unique())
    forecast6_dir = os.path.join(os.getcwd(), 'forecast6')
    os.makedirs(forecast6_dir, exist_ok=True)
    for pid in product_ids:
        for branch in branches:
            subdf = df[(df['ProductID'] == pid) & (df['Ship Branch'] == branch)]
            if subdf.empty:
                continue
            prophet_data = preprocess_for_prophet(df, product_id=pid, branch=branch)
            if prophet_data.empty:
                continue
            model, forecast = train_and_forecast(prophet_data, periods=2190)
            # Save forecast table to CSV in forecast6 folder, including all extra columns
            fname = os.path.join(forecast6_dir, f"forecast_6y_{pid}_{branch}.csv")
            forecast.to_csv(fname, index=False)
            forecasts[(pid, branch)] = forecast
    global six_year_forecasts
    six_year_forecasts = forecasts
    return JSONResponse({
        "status": "success",
        "forecast_files": [f"forecast6/forecast_6y_{pid}_{branch}.csv" for (pid, branch) in forecasts.keys()]
    })

@app.get("/train/all", response_class=HTMLResponse)
def train_all_report(request: Request):
    import google.generativeai as genai
    df = dataframes['main']
    # For each product/branch, get future forecast and aggregate
    product_ids = sorted(df['ProductID'].dropna().unique())
    branches = sorted(df['Ship Branch'].dropna().unique())
    all_summaries = []
    for pid in product_ids:
        for branch in branches:
            prophet_data = preprocess_for_prophet(df, product_id=pid, branch=branch)
            if prophet_data.empty:
                continue
            model, forecast = train_and_forecast(prophet_data, periods=90)
            group_cols = ['Product', 'Branch', 'Region', 'SalesOffice', 'Regional Manager', 'Sales Head']
            for col in group_cols:
                if col not in forecast.columns:
                    forecast[col] = prophet_data[col].iloc[-1] if col in prophet_data.columns and not prophet_data[col].empty else ''
            last_train_date = prophet_data['ds'].max()
            future_forecast = forecast[forecast['ds'] > last_train_date]
            summary = future_forecast.groupby(group_cols)['yhat'].sum().reset_index().rename(columns={'yhat': 'TotalSales'})
            total_sales = summary['TotalSales'].sum()
            summary['PercentOfTotal'] = summary['TotalSales'] / total_sales * 100 if total_sales else 0
            summary = summary.sort_values('TotalSales')
            summary_lines = []
            for _, row in summary.iterrows():
                summary_lines.append(
                    f"Product: <b>{row['Product']}</b> in <b>{row['Region']}</b> region (Branch: <b>{row['Branch']}</b>, Sales Office: <b>{row['SalesOffice']}</b>) has sales: <b>{row['TotalSales']:,.0f}</b> (<b>{row['PercentOfTotal']:.2f}%</b> of this product/branch). Responsible: Regional Manager: <b>{row['Regional Manager']}</b>, Sales Head: <b>{row['Sales Head']}</b>."
                )
            if summary_lines:
                all_summaries.append(f"<h3>Executive Summary: Sales Insights for {productid_to_name.get(pid, str(pid))} ({branch})</h3><ul>" + "\n".join([f"<li>{line}</li>" for line in summary_lines]) + "</ul>")
    prompt = "You are a business analyst. Given the following sales forecast summaries for all product/branch pairs, generate a consolidated executive summary in HTML. Use the same template as above.\n" + "\n".join(all_summaries) + "\n<b>Recommendation:</b> Focus on improving sales for the above region/office combinations. Regional Managers and Sales Heads should review sales strategies and marketing efforts."
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        html = response.text.replace('```html', '').replace('```', '')
    except Exception as e:
        html = f"<div class='alert alert-danger'>GenAI error: {e}</div>"
    return HTMLResponse(html)

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
    import os
    import google.generativeai as genai
    from PIL import Image
    global prophet_model, forecast_df
    if prophet_model is None or forecast_df is None:
        return HTMLResponse("<div class='alert alert-warning'>No forecast available.</div>")
    # Generate and save the plot image
    plot_path = "prophet_forecast.png"
    fig = prophet_model.plot(forecast_df)
    fig.savefig(plot_path, format='png')
    # Use Gemini to generate insights from the plot image
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        img = Image.open(plot_path)
    except FileNotFoundError:
        return HTMLResponse(f"<div class='alert alert-danger'>Error: The file {plot_path} was not found.</div>")
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(
            contents=[img, (
                "You are a business executive sales analyst. "
                "Analyze this sales forecast plot and provide structured, executive-level business insights in HTML. "
                "Focus only on future prospects, sales trends, growth or decline, and actionable recommendations for business strategy. "
                "Use only sales numbers and percentages that are clearly present in the data. "
                "Never mention technical or modeling details, N/A, 0%, missing/zero/uncertain/unavailable values, or anything you cannot see in the data. "
                "Do not discuss how to improve the model or data. "
                "Never mention the years explicitly or use hardcoded year values. "
                "Never mention the image or the process of analysis. "
                "Use bold for key numbers, bullet points for recommendations, and clear sections. "
                "Do not show the image, only the insights."
            )]
        )
        insights_html = response.text
    except Exception as e:
        return HTMLResponse(f"<div class='alert alert-danger'>An error occurred during content generation: {e}</div>")
    return HTMLResponse(insights_html)

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
    import os
    import google.generativeai as genai
    from PIL import Image
    import matplotlib.pyplot as plt
    import time
    df = dataframes['main']
    # Generate and save images for all product/branch pairs into static directory
    static_dir = os.path.join(os.getcwd(), 'static')
    os.makedirs(static_dir, exist_ok=True)
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
            img_path = os.path.join(static_dir, f"prophet_forecast_{pid}_{branch}.png")
            fig.savefig(img_path, format='png')
            plt.close(fig)
            image_paths.append((img_path, pid, branch))
    if not image_paths:
        return HTMLResponse("<div class='alert alert-warning'>No forecast images available for consolidated insights.</div>")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    insights_html = ""
    for img_path, pid, branch in image_paths:
        # Wait for the file to be available (max 10s)
        waited = 0
        while not os.path.exists(img_path) and waited < 10:
            time.sleep(0.5)
            waited += 0.5
        if not os.path.exists(img_path):
            insights_html += f"<div class='alert alert-danger'>Image for ProductID {pid} Branch {branch} could not be found after waiting.</div>"
            continue
        try:
            img = Image.open(img_path)
        except Exception as e:
            insights_html += f"<div class='alert alert-danger'>Error opening image for ProductID {pid} Branch {branch}: {e}</div>"
            continue
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            response = model.generate_content(
                contents=[img, (
                    f"You are a professional sales executive. Analyze this sales forecast plot for ProductID {pid} and Branch {branch} and provide a structured, executive-level business insight in HTML. "
                    "Focus only on future prospects, sales trends, growth or decline, and actionable recommendations for business strategy. "
                    "Use only sales numbers and percentages that are clearly present in the data. "
                    "Never mention technical or modeling details, N/A, 0%, missing/zero/uncertain/unavailable values, or anything you cannot see in the data. "
                    "Do not discuss how to improve the model or data. "
                    "Never mention the years explicitly or use hardcoded year values. "
                    "Never mention the image or the process of analysis. "
                    "Use bold for key numbers, bullet points for recommendations, and clear sections. "
                    "Do not show the image, only the insights."
                )]
            )
            insights_html += f"<div style='margin-bottom:2em'><h4>ProductID {pid} - Branch {branch}</h4>" + response.text + "</div>"
        except Exception as e:
            insights_html += f"<div class='alert alert-danger'>An error occurred for ProductID {pid} Branch {branch}: {e}</div>"
    if not insights_html:
        return HTMLResponse("<div class='alert alert-danger'>No insights could be generated from the forecast images.</div>")
    return HTMLResponse(insights_html)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.get("/genai/insights", response_class=HTMLResponse)
def genai_insights(request: Request):
    import os
    import google.generativeai as genai
    from PIL import Image
    import matplotlib.pyplot as plt
    global prophet_model, forecast_df
    # Ensure the plot image exists by generating it if needed
    plot_path = "prophet_forecast.png"
    if prophet_model is not None and forecast_df is not None:
        fig = prophet_model.plot(forecast_df)
        fig.savefig(plot_path, format='png')
        plt.close(fig)
    try:
        img = Image.open(plot_path)
    except FileNotFoundError:
        return HTMLResponse(f"<div class='alert alert-danger'>Error: The file {plot_path} was not found and could not be generated.</div>")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(
            contents=[img, (
                "You are a business executive sales analyst. "
                "Analyze this sales forecast plot and provide structured, executive-level business insights in HTML. "
                "Focus only on future prospects, sales trends, growth or decline, and actionable recommendations for business strategy. "
                "Use only sales numbers and percentages that are clearly present in the data. "
                "Never mention technical or modeling details, N/A, 0%, missing/zero/uncertain/unavailable values, or anything you cannot see in the data. "
                "Do not discuss how to improve the model or data. "
                "Never mention the years explicitly or use hardcoded year values. "
                "Never mention the image or the process of analysis. "
                "Use bold for key numbers, bullet points for recommendations, and clear sections. "
                "Do not show the image, only the insights."
            )]
        )
        insights_html = response.text
    except Exception as e:
        return HTMLResponse(f"<div class='alert alert-danger'>An error occurred during content generation: {e}</div>")
    return HTMLResponse(insights_html)

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
            image_paths.append(img_path)
    if not image_paths:
        return HTMLResponse("<div class='alert alert-warning'>No forecast images available for consolidated insights.</div>")
    imgs = []
    for img_path in image_paths:
        try:
            imgs.append(Image.open(img_path))
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
                    "You are a professional sales executive. "
                    "Analyze these sales forecast plots and provide a consolidated, structured, executive-level business insight in HTML. "
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
        insights_html = response.text
    except Exception as e:
        return HTMLResponse(f"<div class='alert alert-danger'>An error occurred during content generation: {e}</div>")
    return HTMLResponse(insights_html)

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

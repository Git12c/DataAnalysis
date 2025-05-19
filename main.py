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

# Utility: Prepare Prophet data
def prepare_prophet_data(df):
    # Use the modular preprocessing
    return preprocess_for_prophet(df)

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

user_selection = {
    'product_id': 500000,
    'branch': 'Branch 1'
}

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

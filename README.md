# Data Analysis & Forecasting API

This project provides a modular FastAPI service for data analysis and time series forecasting using Prophet. It is designed for easy integration with Power BI and includes a user-friendly web interface for data and forecast visualization.

## Project Structure

```
Data_2020.csv
Data_2021.csv
SQL Data 5 Years Order Date.csv
main.py                  # FastAPI app (serves API and web interface)
data_loader.py           # Data loading and merging logic
preprocessing.py         # Data preprocessing for Prophet
prophet_model.py         # Prophet model training and plot saving
static/                  # Contains Prophet plot images
    prophet_forecast.png
    prophet_components.png
templates/
    index.html           # Web interface template
generate_prophet_plots.py# Script to generate Prophet plots
requirements.txt         # Python dependencies
README.md                # This file
```

## Features
- **API endpoints** for Power BI and other clients:
  - `/api/data`: Get merged dataset as JSON
  - `/api/prophet-forecast`: Get Prophet forecast as JSON
- **Web interface** for:
  - Selecting Product ID (with at least 60 records) and Branch
  - Viewing Prophet forecast and component plots
  - Viewing a data table preview
- **Modular codebase** for easy maintenance and extension
- **CORS enabled** for integration with Power BI

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Prophet Plots (Optional)
This step trains the Prophet model and saves forecast/component plots as images:
```bash
python3 generate_prophet_plots.py
```

### 3. Start the API Service
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8047
```

### 4. Access the Interface & API
- **Web UI:** [http://localhost:8047](http://localhost:8047)
- **Prophet Forecast Plot:** [http://localhost:8047/prophet/plot](http://localhost:8047/prophet/plot)
- **Prophet Components Plot:** [http://localhost:8047/prophet/components](http://localhost:8047/prophet/components)
- **Data Table:** [http://localhost:8047/data-table](http://localhost:8047/data-table)
- **Product/Branch Selection:** Use the form on the main page or [http://localhost:8047/select](http://localhost:8047/select)

#### API Endpoints
- `GET /api/data` — Returns the merged dataset as JSON (for Power BI or other clients)
- `GET /api/prophet-forecast` — Returns the Prophet forecast as JSON (for Power BI or other clients)
- `GET /prophet/plot` — Returns the Prophet forecast plot (HTML with image)
- `GET /prophet/components` — Returns the Prophet model components plot (HTML with image)
- `GET /data-table` — Returns a preview of the data table (HTML)
- `GET /select` — Returns the selection form for Product ID and Branch (HTML)
- `POST /select` — Updates the model and plots based on user selection (HTML)
- `GET /` — Main web interface (HTML)

### 5. Power BI Integration
- Use the Web data source in Power BI and point it to the `/api/data` or `/api/prophet-forecast` endpoint.

## Notes
- Data files (`Data_2020.csv`, `Data_2021.csv`, `SQL Data 5 Years Order Date.csv`) must be present in the project root.
- The Product ID dropdown only shows IDs with at least 60 records for meaningful forecasting.
- The project is fully modular and easy to extend for new data sources or models.
- All plots are pre-generated and served as static images for reliability.

---

**Author:** Your Name

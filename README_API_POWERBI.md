# DataAnalysis FastAPI - Power BI Integration Guide

This guide explains how to use the available API endpoints in the DataAnalysis FastAPI project, with step-by-step instructions for integrating the endpoints into Microsoft Power BI. All endpoints return data in JSON or HTML format and are suitable for executive reporting, forecasting, and GenAI insights.

---

## Table of Contents
- [Available API Endpoints](#available-api-endpoints)
- [Power BI Integration: Step-by-Step](#power-bi-integration-step-by-step)
- [Endpoint Details](#endpoint-details)
- [Best Practices](#best-practices)

---

## Available API Endpoints

| Endpoint                        | Method | Description                                                                                 |
|---------------------------------|--------|---------------------------------------------------------------------------------------------|
| `/api/data`                     | GET    | Returns the full sales dataset as JSON.                                                     |
| `/api/prophet-forecast`         | GET    | Returns the Prophet forecast for the selected product/branch as JSON.                       |
| `/api/manager-head`             | GET    | Returns the Regional Manager and Sales Head for a given product and branch.                 |
| `/train/all`                    | GET    | Returns an HTML consolidated forecast report for all products and branches.                  |
| `/genai/insights`               | GET    | Returns GenAI executive insights for the current forecast (HTML).                           |
| `/genai/consolidated-insights`  | GET    | Returns GenAI-style executive insights for all products and branches (HTML).                |
| `/data-table`                   | GET    | Returns a preview of the data table (HTML).                                                 |

---

## Power BI Integration: Step-by-Step

### 1. Start the FastAPI Server
- Make sure your FastAPI server is running and accessible (e.g., `http://localhost:8000`).

### 2. Open Power BI Desktop
- Launch Power BI Desktop on your machine.

### 3. Get Data from Web
- Click **Home > Get Data > Web**.
- In the dialog, select **Advanced**.
- Enter the API endpoint URL you want to use (see above for options).
  - Example: `http://localhost:8000/api/data`
- Click **OK**.

### 4. Configure Request (if needed)
- For endpoints that require parameters (e.g., `/api/manager-head?product_id=1000&branch=East`), add them to the URL.
- For endpoints returning HTML, use Power BI's web scraping/table extraction features.

### 5. Transform Data
- In the Power Query Editor, transform and shape the data as needed.
- Rename columns, filter rows, and set data types.

### 6. Load Data
- Click **Close & Apply** to load the data into Power BI.

### 7. Build Visuals
- Use the loaded tables to create dashboards, charts, and executive reports.

---

## Endpoint Details

### `/api/data` (GET)
- Returns: Full sales data as a JSON array.
- Use for: Loading the entire dataset into Power BI.

### `/api/prophet-forecast` (GET)
- Returns: Prophet forecast results for the selected product/branch as JSON.
- Use for: Time series forecasting visuals.

### `/api/manager-head` (GET)
- Parameters: `product_id` (int), `branch` (str)
- Example: `/api/manager-head?product_id=1000&branch=East`
- Returns: `{ "RegionalManager": "...", "SalesHead": "..." }`
- Use for: Executive name lookups in Power BI.

### `/train/all` (GET)
- Returns: HTML report with a pivot table and executive insights for all products/branches.
- Use for: Embedding or scraping summary tables into Power BI.

### `/genai/insights` (GET)
- Returns: HTML with GenAI-generated executive insights for the current forecast.
- Use for: Executive summary visuals.

### `/genai/consolidated-insights` (GET)
- Returns: HTML with GenAI-style insights for all products/branches.
- Use for: High-level executive dashboards.

### `/data-table` (GET)
- Returns: HTML preview of the first 100 rows of the dataset.
- Use for: Quick data previews.

---

## Best Practices
- **Always use the JSON endpoints** (`/api/data`, `/api/prophet-forecast`, `/api/manager-head`) for structured data in Power BI.
- For HTML endpoints, use Power BI's web scraping features to extract tables.
- Use parameterized endpoints for dynamic queries (e.g., manager/head lookup).
- Refresh your Power BI data regularly to keep reports up to date.
- Secure your API if deploying in production.

---

For further customization or troubleshooting, refer to the project source code or contact the development team.

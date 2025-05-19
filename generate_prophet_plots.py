from data_loader import load_and_merge_data
from preprocessing import preprocess_for_prophet
from prophet_model import train_and_forecast, save_prophet_plots

if __name__ == "__main__":
    # Load and preprocess data
    df_merged = load_and_merge_data()
    prophet_data = preprocess_for_prophet(df_merged, product_id=500000, branch='Branch 1')
    # Train Prophet and save plots
    model, forecast = train_and_forecast(prophet_data, periods=90)
    save_prophet_plots(model, forecast)
    print("Prophet model trained and plots saved in 'static/' directory.")

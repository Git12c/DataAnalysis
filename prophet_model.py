from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd

def train_and_forecast(df, periods=90):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    return model, forecast

def save_prophet_plots(model, forecast, static_dir='static'):
    fig1 = model.plot(forecast, uncertainty=True, plot_cap=True, include_legend=True)
    plt.xlabel('Date')
    plt.ylabel('Total price')
    plt.tight_layout()
    fig1.savefig(f'{static_dir}/prophet_forecast.png')
    plt.close(fig1)

    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    fig2.savefig(f'{static_dir}/prophet_components.png')
    plt.close(fig2)

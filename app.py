from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import matplotlib
matplotlib.use('Agg')
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


app = Flask(__name__, static_folder='static')

# Inisialisasi dataset
DATA_PATH = 'data/kopi.csv'

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('home.html')


# Route untuk halaman edit dataset
@app.route('/edit-dataset', methods=['GET', 'POST'])
def edit_dataset():
    # Periksa apakah file CSV ada
    if not os.path.exists(DATA_PATH):
        return f"File tidak ditemukan: {DATA_PATH}", 404

    # Baca data dari CSV
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Pastikan kolom Date dalam format datetime
    df = df.dropna(subset=['Date'])

    if request.method == 'POST':
        # Tambah data baru
        if 'add_data' in request.form:
            tanggal = request.form['tanggal']
            produksi = request.form['produksi']
            new_data = pd.DataFrame({'Date': [tanggal], 'Produksi': [produksi]})
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(DATA_PATH, index=False)
            return redirect(url_for('edit_dataset'))

        # Edit data
        if 'edit_data' in request.form:
            tanggal_edit = request.form['tanggal_edit']
            produksi_edit = request.form['produksi_edit']
            df.loc[df['Date'] == tanggal_edit, 'Produksi'] = produksi_edit
            df.to_csv(DATA_PATH, index=False)
            return redirect(url_for('edit_dataset'))

        # Hapus data
        if 'delete_data' in request.form:
            tanggal_hapus = request.form['tanggal_hapus']
            df = df[df['Date'] != tanggal_hapus]
            df.to_csv(DATA_PATH, index=False)
            return redirect(url_for('edit_dataset'))

    # Membuat grafik dengan Plotly
    fig = go.Figure()

    # Tambahkan data produksi
    fig.add_trace(go.Scatter(
        x=df.index,  # Kolom 'Date' digunakan sebagai sumbu x
        y=df['Produksi'],  # Kolom 'Produksi' digunakan sebagai sumbu y
        mode='lines',
        name='Produksi Kopi',
        line=dict(color='blue', width=2)
    ))

    # Tambahkan judul dan label sumbu
    fig.update_layout(
        title="Perkembangan Produksi Kopi Provinsi Aceh",
        xaxis_title="Tanggal",
        yaxis_title="Produksi",
        template="plotly"
    )

    # Konversi grafik ke HTML
    graph_html = fig.to_html(full_html=False)

    # Render template edit dataset
    return render_template('edit_dataset.html', data=df.to_dict(orient='records'), graph_html=graph_html)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not os.path.exists(DATA_PATH):
        return "File dataset tidak ditemukan!", 404

    # Membaca parameter SARIMA dari form (jika ada)
    p = int(request.form.get('p', 1))
    d = int(request.form.get('d', 0))
    q = int(request.form.get('q', 1))
    P = int(request.form.get('P', 1))
    D = int(request.form.get('D', 0))
    Q = int(request.form.get('Q', 1))
    s = int(request.form.get('s', 12))
    forecast_periods = int(request.form.get('forecast_periods', 24))

    # Membaca data
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # Hapus baris dengan tanggal tidak valid
    df.set_index('Date', inplace=True)

    # Tangani NaN di kolom Produksi
    if df['Produksi'].isnull().any():
        print("Nilai NaN terdeteksi di kolom Produksi. Menangani NaN...")
        df['Produksi'] = df['Produksi'].fillna(df['Produksi'].mean())  # Ganti NaN dengan rata-rata

    # Normalisasi data
    df['Data_Normalisasi'] = (df['Produksi'] - df['Produksi'].min()) / (df['Produksi'].max() - df['Produksi'].min())

    # Periksa NaN setelah normalisasi
    if df['Data_Normalisasi'].isnull().any():
        return "Terjadi kesalahan dalam normalisasi data. Pastikan dataset valid.", 500

    # Cross-validation menggunakan TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    train_data, test_data = [], []
    train_test_visuals = []  # Untuk menyimpan grafik pelatihan dan pengujian

    for i, (train_index, test_index) in enumerate(tscv.split(df), start=1):
        # Pisahkan data menjadi pelatihan dan pengujian
        train_fold = df.iloc[train_index]
        test_fold = df.iloc[test_index]
        train_data.append(train_fold)
        test_data.append(test_fold)

        # Plot untuk data pelatihan
        train_fig = go.Figure()
        train_fig.add_trace(go.Scatter(
            x=train_fold.index,
            y=train_fold['Produksi'],
            mode='lines',
            name=f'Pelatihan Fold {i}',
            line=dict(color='green')
        ))
        train_fig.update_layout(
            title=f"Pelatihan Fold {i}",
            xaxis_title="Tanggal",
            yaxis_title="Produksi",
            template="plotly_white"
        )
        train_plot_html = train_fig.to_html(full_html=False)

        # Plot untuk data pengujian
        test_fig = go.Figure()
        test_fig.add_trace(go.Scatter(
            x=test_fold.index,
            y=test_fold['Produksi'],
            mode='lines+markers',
            name=f'Pengujian Fold {i}',
            line=dict(color='orange')
        ))
        test_fig.update_layout(
            title=f"Pengujian Fold {i}",
            xaxis_title="Tanggal",
            yaxis_title="Produksi",
            template="plotly_white"
        )
        test_plot_html = test_fig.to_html(full_html=False)

        # Simpan grafik untuk fold ini
        train_test_visuals.append({
            'train_html': train_plot_html,
            'test_html': test_plot_html
        })

        # Gunakan fold terakhir untuk ADF test dan visualisasi
        adf_test = adfuller(train_data[-1]['Data_Normalisasi'])

    # Plot ACF dan PACF
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(train_data[-1]['Data_Normalisasi'], ax=ax[0])
    plot_pacf(train_data[-1]['Data_Normalisasi'], lags=11, ax=ax[1])
    plt.tight_layout()
    acf_pacf_path = os.path.join('static', 'acf_pacf.png')
    plt.savefig(acf_pacf_path)
    plt.close()

    # Fitting SARIMA dengan parameter yang diberikan
    model = SARIMAX(
        train_data[-1]['Data_Normalisasi'],
        order=(p, d, q),
        seasonal_order=(P, D, Q, s)
    )
    model_fit = model.fit()
    model_summary = model_fit.summary().as_text()  # Mengambil ringkasan model SARIMA

    # Prediksi
    forecast = model_fit.forecast(steps=len(test_data[-1]))
    test_data[-1]['Forecast'] = forecast

    # Denormalisasi hasil prediksi
    min_value = df['Produksi'].min()
    max_value = df['Produksi'].max()
    test_data[-1]['Forecast_Denormalisasi'] = (test_data[-1]['Forecast'] * (max_value - min_value)) + min_value

    # Pastikan tidak ada NaN di kolom yang relevan
    if test_data[-1]['Produksi'].isnull().any() or test_data[-1]['Forecast_Denormalisasi'].isnull().any():
        print("Terdeteksi NaN pada data pengujian atau prediksi. Menangani NaN...")
        test_data[-1] = test_data[-1].dropna(subset=['Produksi', 'Forecast_Denormalisasi'])

    # Prediksi masa depan
    future_forecast = model_fit.forecast(steps=forecast_periods)
    future_forecast_denormalized = (future_forecast * (max_value - min_value)) + min_value
    future_index = pd.date_range(start=test_data[-1].index[-1], periods=forecast_periods + 1, freq='M')[1:]

    # Gabungkan hasil prediksi masa depan
    future_data = pd.DataFrame({
        'Date': future_index,
        'Forecast': future_forecast,
        'Forecast_Denormalisasi': future_forecast_denormalized
    })
    # Gabungkan hasil prediksi masa depan
    future_data = pd.DataFrame({
        'Date': future_index,
        'Forecast': future_forecast,
        'Forecast_Denormalisasi': future_forecast_denormalized
    })

    # Gabungkan data aktual dengan prediksi masa depan
    combined_df = pd.concat(
        [df[['Produksi']].reset_index(), future_data[['Date', 'Forecast_Denormalisasi']].rename(columns={'Date': 'index'})],
        ignore_index=True
    )

    # Plot Data Produksi dengan Prediksi Masa Depan menggunakan Plotly
    combined_fig = go.Figure()
    # Plot data aktual
    combined_fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Produksi'],
        mode='lines',
        name='Data Aktual',
        line=dict(color='blue')
    ))
    # Plot prediksi masa depan
    combined_fig.add_trace(go.Scatter(
        x=future_data['Date'],
        y=future_data['Forecast_Denormalisasi'],
        mode='lines+markers',
        name='Prediksi Masa Depan',
        line=dict(color='green', dash='dash')
    ))
    # Update layout
    combined_fig.update_layout(
        title="Perbandingan Data Aktual dan Prediksi Masa Depan dengan SARIMA",
        xaxis_title="Tanggal",
        yaxis_title="Produksi",
        template="plotly",
        showlegend=True
    )

    # Konversi grafik ke HTML
    combined_plot_html = combined_fig.to_html(full_html=False)

    # Residuals dari model
    residuals = model_fit.resid[1:]  # Residuals dari SARIMA model
    # Residual untuk periode tertentu
    residual_t_minus_1 = residuals[-2]  # Residual bulan sebelumnya
    residual_t_minus_12 = residuals[-13]  # Residual musiman (12 bulan sebelumnya)
    residual_t = residuals[-1]  # Residual saat ini 

    # Statistik Deskriptif Residual
    residual_stats = {
        'Mean': residuals.mean(),
        'Variance': residuals.var(),
        'Standard Deviation': residuals.std(),
        'Min': residuals.min(),
        'Max': residuals.max()
    }

    # Uji Normalitas Residual (Shapiro-Wilk Test)
    from scipy.stats import shapiro
    shapiro_stat, shapiro_p_value = shapiro(residuals)
    # Tambahkan statistik residual ke output
    residual_summary = f"""
    Residual Statistics:
    - Mean: {residual_stats['Mean']:.4f}
    - Variance: {residual_stats['Variance']:.4f}
    - Standard Deviation: {residual_stats['Standard Deviation']:.4f}
    - Min: {residual_stats['Min']:.4f}
    - Max: {residual_stats['Max']:.4f}

    Specific Residuals:
    - Residual (t): {residual_t:.4f}
    - Residual (t-1): {residual_t_minus_1:.4f}
    - Residual (t-12): {residual_t_minus_12:.4f}

    Shapiro-Wilk Test for Normality:
    - Statistic: {shapiro_stat:.4f}
    - P-value: {shapiro_p_value:.4f}
    """
    # Plot residuals dan density
    fig_residuals, ax = plt.subplots(1, 2, figsize=(12, 6))
    residuals.plot(title='Residuals', ax=ax[0])
    residuals.plot(title='Density', kind='kde', ax=ax[1])
    residuals_path = os.path.join('static', 'residuals_plot.png')
    plt.tight_layout()
    plt.savefig(residuals_path)
    plt.close(fig_residuals)

    # Plot ACF dan PACF untuk residuals
    fig_acf_pacf, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(residuals, ax=ax[0])
    ax[0].set_title("Autocorrelation Function (ACF) untuk Residual")
    plot_pacf(residuals, ax=ax[1], lags=10, method='ywm')
    ax[1].set_title("Partial Autocorrelation Function (PACF) untuk Residual")
    acf_pacf_path_residuals = os.path.join('static', 'acf_pacf_residuals.png')
    plt.tight_layout()
    plt.savefig(acf_pacf_path_residuals)
    plt.close(fig_acf_pacf)

    # Hitung metrik error
    mae = mean_absolute_error(test_data[-1]['Produksi'], test_data[-1]['Forecast_Denormalisasi'])
    mape = mean_absolute_percentage_error(test_data[-1]['Produksi'], test_data[-1]['Forecast_Denormalisasi']) * 100
    rmse = np.sqrt(mean_squared_error(test_data[-1]['Produksi'], test_data[-1]['Forecast_Denormalisasi']))

        # Grafik hasil prediksi menggunakan Plotly
    fig = go.Figure()

    # Tambahkan data aktual
    fig.add_trace(go.Scatter(
        x=df.index,  # Indeks tanggal dari data asli
        y=df['Produksi'],  # Kolom produksi aktual
        mode='lines',
        name='Data Aktual',
        line=dict(color='blue', width=2),
        hoverinfo='x+y'
    ))

    # Tambahkan data prediksi
    fig.add_trace(go.Scatter(
        x=test_data[-1].index,  # Indeks tanggal dari data prediksi
        y=test_data[-1]['Forecast_Denormalisasi'],  # Prediksi denormalisasi
        mode='lines+markers',
        name='Prediksi',
        line=dict(color='green', dash='dash', width=2),
        marker=dict(size=8, symbol='circle'),
        hoverinfo='x+y'
    ))

    # Perbarui tata letak grafik
    fig.update_layout(
        title="Perbandingan Data Aktual dan Prediksi",
        xaxis=dict(
            title="Tanggal",
            showgrid=True,
            zeroline=False,
            showline=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            title="Produksi Kopi",
            showgrid=True,
            zeroline=False,
            showline=True,
            gridcolor='lightgrey'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )

    # Gabungkan data aktual, data normalisasi, prediksi normalisasi, dan prediksi denormalisasi
    combined_table = pd.DataFrame({
        'Produksi Aktual': test_data[-1]['Produksi'],
        'Produksi Normalisasi': test_data[-1]['Data_Normalisasi'] if 'Data_Normalisasi' in test_data[-1] else None,
        'Prediksi Normalisasi': test_data[-1]['Forecast'],
        'Prediksi Denormalisasi': test_data[-1]['Forecast_Denormalisasi']
    }).reset_index()

    # Konversi tabel ke HTML
    combined_table_html = combined_table.to_html(classes='table table-bordered', index=False)

    # Tabel Future Forecast
    future_forecast_table = pd.DataFrame({
        'Tanggal': future_data['Date'],
        'Prediksi Normalisasi': future_data['Forecast'],
        'Prediksi Denormalisasi': future_data['Forecast_Denormalisasi']
    })

    # Konversi tabel Future Forecast ke HTML
    future_forecast_table_html = future_forecast_table.to_html(classes='table table-bordered', index=False)


    return render_template(
        'prediction_result.html',
        graph_html=fig.to_html(full_html=False),
        train_test_visuals=train_test_visuals,
        enumerate=enumerate,
        acf_pacf_graph=acf_pacf_path,
        combined_plot_html=combined_plot_html,
        residuals_plot=residuals_path,
        acf_pacf_residuals=acf_pacf_path_residuals,
        residual_summary=residual_summary,
        mae=mae,
        mape=mape,
        rmse=rmse,
        adf_pvalue=adf_test[1],
        test_data=test_data[-1].reset_index().to_dict(orient='records'),
        future_data=future_data.to_dict(orient='records'),
        p=p, d=d, q=q, P=P, D=D, Q=Q, s=s, forecast_periods=forecast_periods,
        model_summary=model_summary,  # Tambahkan ringkasan model
        combined_table_html=combined_table_html,  # Tambahkan tabel HTML
        future_forecast_table_html=future_forecast_table_html
    )


# Route untuk halaman "Tentang"
@app.route('/tentang', methods=['GET', 'POST'])
def tentang():
    return render_template('tentang.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
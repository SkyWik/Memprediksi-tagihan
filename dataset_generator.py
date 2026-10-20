import pandas as pd
import numpy as np

def generate_pt_maju_bersama_data(num_samples=5000):
    np.random.seed(42)
    
    # Fitur sebelum kejadian (No Data Leakage)
    invoice_amount = np.random.exponential(scale=10000000, size=num_samples) + 500000
    credit_term_days = np.random.choice([15, 30, 45, 60], size=num_samples, p=[0.2, 0.5, 0.2, 0.1])
    customer_past_delays = np.random.poisson(lam=1.5, size=num_samples)
    customer_credit_score = np.random.normal(loc=650, scale=100, size=num_samples)
    
    # Target: Keterlambatan > 30 hari (1 = Terlambat/Bermasalah, 0 = Tepat Waktu)
    # Logit berdasarkan kombinasi linier fitur
    logit = (
        0.00000005 * invoice_amount +
        0.03 * credit_term_days +
        0.5 * customer_past_delays -
        0.008 * customer_credit_score - 1.2
    )
    prob = 1 / (1 + np.exp(-logit))
    is_late = (np.random.rand(num_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'invoice_amount': invoice_amount,
        'credit_term_days': credit_term_days,
        'customer_past_delays': customer_past_delays,
        'customer_credit_score': customer_credit_score,
        'is_late_30_days': is_late
    })
    
    return df

if __name__ == '__main__':
    df = generate_pt_maju_bersama_data()
    df.to_csv('pt_maju_bersama_invoices.csv', index=False)
    print("Dataset sintetik PT Maju Bersama berhasil dibuat: pt_maju_bersama_invoices.csv")

# Studi Kasus: Deteksi Risiko Keterlambatan Pembayaran Piutang (PT Maju Bersama)

## Latar Belakang
Berdasarkan pemaparan Bpk. Andria Arisal (Research Center for Data and Information Sciences, BRIN) dalam sesi "Make Sense of Data with Analytics and AI", PT Maju Bersama mengalami fenomena paradoks: pendapatan naik 33,74%, tetapi laba bersih justru turun 14,43%, piutang melonjak 386,20%, dan arus kas operasi menjadi negatif. Persoalan utamanya terletak pada tingginya angka *unpaid invoices* yang menunggak lebih dari 30 hari setelah jatuh tempo.

## Tantangan & Prinsip Pemodelan AI
1. **Pencegahan Data Leakage**: Fitur yang digunakan untuk prediksi harus merupakan informasi yang *sudah tersedia saat invoice diterbitkan* (misal: jumlah tagihan, term pembayaran, histori keterlambatan pelanggan sebelumnya). Informasi yang baru diketahui *setelah* kejadian (seperti tanggal pembayaran aktual) wajib dikeluarkan.
2. **Ketidakseimbangan Kelas & Evaluasi Metrik**: Dalam kasus tunggakan, akurasi tinggi (misal 89%) bisa sangat menyesatkan jika nilai *Recall* mendekati 0% (model gagal mendeteksi invoice bermasalah). Oleh karena itu, evaluasi wajib fokus pada ROC-AUC, Recall, Precision, dan Confusion Matrix.
3. **Analitik Prediktif berbasis Deep Learning**: Membangun arsitektur *Multilayer Perceptron* (MLP / Deep Neural Network) menggunakan PyTorch untuk memprediksi probabilitas keterlambatan pembayaran secara otomatis.

## Struktur Project
- `Studi_Kasus_PT_Maju_Bersama.md` : Dokumentasi latar belakang bisnis, masalah *data leakage*, dan analisis metrik.
- `dataset_generator.py` : Skrip untuk menghasilkan data sintetik yang merepresentasikan transaksi PT Maju Bersama.
- `model.py` : Arsitektur Deep Neural Network (MLP) berbasis PyTorch.
- `train.py` : Skrip *preprocessing* (pembagian berdasarkan waktu), pelatihan model, penanganan imbalanced data, serta evaluasi metrik ROC-AUC & Confusion Matrix.
- `requirements.txt` : Daftar pustaka Python yang digunakan.

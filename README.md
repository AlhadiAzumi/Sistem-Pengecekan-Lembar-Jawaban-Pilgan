# Sistem Pemeriksaan Lembar Jawaban Pilihan Ganda
Praktikum Pengolahan Citra Digital — Python, OpenCV, NumPy

## 1. Struktur Folder
```
proyek_pilgan/
├── main.py                     <- program utama (satu file, langsung jalan)
├── requirements.txt
├── README.md
└── dataset/
    ├── template_kosong.jpeg    <- gambar lembar KOSONG (acuan posisi huruf A-E)
    ├── kunci_contoh.csv        <- contoh format kunci jawaban
    └── lembar/                 <- taruh foto lembar terisi di sini
```

## 2. Instalasi (sekali saja)
Buka Anaconda Prompt / Terminal / CMD, masuk ke folder proyek:
```
cd path\ke\proyek_pilgan
pip install -r requirements.txt
```

## 3. Coba Dulu dengan Data Demo
```
python main.py --demo
```
Program membuat 5 foto lembar buatan (dengan perspektif, cahaya tidak rata, lingkaran/silang/arsir,
soal kosong, soal ganda), memeriksanya, dan menampilkan nilai serta akurasi pembacaan.
Hasil ada di `hasil/demo/`.

## 4. Menjalankan dengan Data Sendiri
1. Taruh foto lembar terisi di `dataset/lembar/` (misal `lembar_01.jpg` ... `lembar_25.jpg`).
2. Buat `dataset/kunci.csv` (salin dari `kunci_contoh.csv` lalu ubah isinya):
   ```
   soal,jawaban
   1,A
   2,C
   ...
   50,E
   ```
3. Jalankan:
   ```
   python main.py
   ```
   Kalau gambar berupa hasil scan / sudah dipotong rapi (tidak ada latar meja):
   ```
   python main.py --scan
   ```

### Cara mengisi lembar
Lingkari, silang, atau arsir HURUF jawaban (A-E) yang dipilih. Satu soal satu tanda.
Soal tanpa tanda = kosong, soal dengan dua tanda kuat = ganda (keduanya dihitung salah).

### Cara foto yang baik
- Seluruh 4 sudut kertas masuk frame, kertas diletakkan di atas latar yang kontras (meja gelap).
- Pencahayaan cukup, tidak ada bayangan tangan/HP yang menutupi lembar.
- Boleh agak miring, program meluruskan otomatis.

## 5. Mengukur Akurasi (opsional, untuk bab pengujian)
Buat `dataset/truth.csv` berisi jawaban asli yang ditandai tiap lembar (dicatat manual).
Satu baris per lembar, jawaban = 50 karakter (A-E, `-` = kosong, `*` = ganda):
```
lembar,jawaban
lembar_01,ABCDEABCDE...(50 karakter)
```
Jalankan:
```
python main.py --truth dataset/truth.csv
```
Akurasi pembacaan = jumlah soal yang terbaca sama dengan truth / total soal.

## 6. Output (folder `hasil/`)
- `ringkasan.csv` : benar, salah, kosong, ganda, nilai tiap lembar.
- `hasil/<nama_lembar>/` berisi gambar tiap tahap (bahan screenshot laporan):
  `1_original`, `2_grayscale`, `3_blur`, `4_canny`, `5_kontur_lembar`,
  `6_perspective_warp`, `7_threshold`, `8_hasil` (kotak hijau = benar, merah = salah,
  biru = posisi kunci), serta `detail.csv` (kunci vs jawaban tiap soal).

Aturan nilai: jawaban sama dengan kunci -> `benar = benar + 1`, selain itu `salah = salah + 1`.
Nilai = benar / 50 x 100.

## 7. Parameter yang Bisa Diatur
| Opsi | Fungsi | Kapan diubah |
|---|---|---|
| `--min-extra 40` | minimal tinta tambahan agar huruf dianggap ditandai | Turunkan (30) jika tanda tipis/pensil tidak terbaca; naikkan (60) jika soal kosong terbaca terisi |
| `--dominance 3.0` | rasio tanda terkuat vs kedua; di bawahnya dianggap ganda | Naikkan jika jawaban ganda tidak terdeteksi; turunkan jika bekas hapusan dianggap ganda |
| `--scan` | lewati pencarian kertas | Gambar sudah scan / dipotong rapi |
| `--template`, `--key`, `--out` | ganti path file/folder | Kalau lokasi file berbeda |

## 8. Masalah Umum
| Gejala | Solusi |
|---|---|
| "Kertas tidak terdeteksi" | Foto di latar kontras, semua sudut terlihat, atau gunakan `--scan` |
| Banyak soal terbaca kosong | Turunkan `--min-extra`; periksa `7_threshold.jpg` apakah tanda terlihat |
| Kotak baca meleset dari huruf | Cek `6_perspective_warp.jpg`; pastikan kertas utuh dan template sama dengan lembar |
| Memakai template/lembar lain | Ubah koordinat di bagian KONFIGURASI pada `main.py` (`ROW_Y0`, `ROW_Y_STEP`, `BLOCK_COLS_X`) |

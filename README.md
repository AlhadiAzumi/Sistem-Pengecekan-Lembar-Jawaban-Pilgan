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
    ├── kunci.csv               <- contoh format kunci jawaban
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
Setelah itu terminal menanyakan instruksi tanda dengan menu angka:
```
1. Hanya silang (X)
2. Hanya bulat / lingkari
3. Hanya bulat hitam / arsir
4. Boleh semua (campuran)   <- Enter = pilihan ini
```
Program membuat 5 foto lembar buatan dengan tanda **bervariasi** (campuran lingkaran, silang, bulat hitam,
plus soal kosong, soal ganda, dan ganda yang salah satunya dicoret), memeriksanya, dan menampilkan nilai
serta akurasi pembacaan. Hasil ada di `hasil/demo/`. Setelah tiap lembar diproses, figure hasil
otomatis terbuka di **jendela baru**: **scroll = zoom**, **drag = geser**, `R` = reset,
`F` = ganti tampilan penuh/seluruh (default: seluruh gambar terlihat, termasuk saat maximize),
`Enter`/`Spasi` = lanjut ke lembar berikutnya, `Esc` = lewati jendela berikutnya.
Ingin melewati menu? Tambahkan langsung `--mode silang|bulat|bulat_hitam|semua`.

Catatan: jika memilih mode spesifik (mis. `1` = hanya silang), sekitar 85% tanda pada lembar demo
memakai gaya tersebut, sedangkan sisanya sengaja memakai gaya lain dan dihitung **salah** sebagai
"tanda tidak sesuai instruksi" - ini untuk memperlihatkan penegakan aturan. File `truth.csv` demo
sudah memperhitungkan aturan ini, jadi akurasi pembacaan seharusnya mendekati 100% bila pembacaan benar.

## 4. Menjalankan dengan Data Sendiri
1. Taruh foto lembar terisi di `dataset/lembar/` (misal `lembar_01.jpg` ... `lembar_25.jpg`).
2. Buat/ubah `dataset/kunci.csv` sesuai kunci jawaban:
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
   Pilih instruksi tanda bila perlu:
   ```
   python main.py --mode silang        # hanya silang
   python main.py --mode bulat         # hanya bulat/lingkar
   python main.py --mode bulat_hitam   # hanya bulat hitam/arsir
   python main.py --mode semua         # boleh semua (default)
   ```
   Kalau gambar berupa hasil scan / sudah dipotong rapi (tidak ada latar meja):
   ```
   python main.py --scan
   ```
   Secara default figure hasil terbuka di jendela baru (scroll = zoom, drag = geser,
   `R` = reset, `F` = penuh/seluruh, `Enter`/`Spasi` = lanjut, `Esc` = lewati jendela berikutnya).
   Saat di-maximize, seluruh gambar tetap terlihat (tidak terpotong); tekan `F` untuk mode penuh
   yang memenuhi jendela (tepi gambar bisa terpotong, geser dengan drag).
   Untuk batch banyak lembar tanpa jendela, tambahkan `--no-show`:
   ```
   python main.py --no-show
   ```

### Cara mengisi lembar
Tandai HURUF jawaban (A-E) sesuai instruksi (`--mode`: silang, bulat, bulat hitam, atau semua). Satu soal satu tanda.
- Soal tanpa tanda = kosong.
- Jika mode spesifik (mis. **hanya silang**), tanda yang **tidak sesuai instruksi** (mis. dilingkari) dianggap **tanda salah** dan dihitung salah.
- Soal dengan dua/lebih tanda kuat = **ganda**: diberi keterangan ganda dan dianggap **belum menjawab** (dihitung salah, masuk kolom ganda).
- Jika pada jawaban ganda salah satu tanda **dicoret** (garis `=`), coretan dianggap pembatal sehingga bukan ganda dan jawaban yang tersisa dibaca.

### Cara foto yang baik
- Seluruh 4 sudut kertas masuk frame, kertas diletakkan di atas latar yang kontras (meja gelap).
- Pencahayaan cukup, tidak ada bayangan tangan/HP yang menutupi lembar.
- Boleh agak miring, program meluruskan otomatis.
- Pastikan lembar tidak kusut.
- Kertas harus berada di tempat yang datar.

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
Untuk data demo, `truth.csv` dibuat otomatis sesuai mode: tanda yang gaya-nya tidak sesuai instruksi
tertulis sebagai `-` (memang seharusnya dibaca kosong/tanda salah).

## 6. Output (folder `hasil/`)
- `ringkasan.csv` : benar, salah, kosong, ganda, nilai tiap lembar (ganda = jawaban ganda valid, dianggap belum menjawab).
- `hasil/<nama_lembar>/` berisi satu figure `figure_tahapan.jpg` (semua tahap dalam satu grid
  berlabel: `1. Original`, `2. Grayscale`, `3. Gaussian Blur`, `4. Canny`, `5. Kontur Lembar`,
  `6. Perspective Warp`, `7. Threshold`, `8. Hasil` — kotak hijau = benar, merah = salah,
  biru = posisi kunci, label `GANDA`/`KOSONG`/`TANDA SALAH` di samping nomor soal, dan legenda warna
  di bawah gambar), serta `detail.csv` (kunci, jawaban, status, keterangan tiap soal; status/keterangan menandai ganda atau ganda yang salah satunya dicoret).

Aturan nilai: jawaban sama dengan kunci -> `benar = benar + 1`, selain itu `salah = salah + 1` (termasuk kosong dan ganda/belum menjawab).
Nilai = benar / 50 x 100.

## 7. Parameter yang Bisa Diatur
| Opsi | Fungsi | Kapan diubah |
|---|---|---|
| `--mode silang\|bulat\|bulat_hitam\|semua` | instruksi jenis tanda jawaban yang diterima (default: ditanya lewat menu saat `--demo`, selain itu `semua`) | Sesuaikan dengan aturan pengisian lembar |
| `--min-extra 40` | minimal tinta tambahan agar huruf dianggap ditandai | Turunkan (30) jika tanda tipis/pensil tidak terbaca; naikkan (60) jika soal kosong terbaca terisi |
| `--dominance 3.0` | rasio tanda terkuat vs kedua; di bawahnya dianggap ganda | Naikkan jika jawaban ganda tidak terdeteksi; turunkan jika bekas hapusan dianggap ganda |
| `--scan` | lewati pencarian kertas | Gambar sudah scan / dipotong rapi |
| `--show` / `--no-show` | tampilkan figure hasil di jendela baru dengan scroll-zoom, drag, dan tampilan menyesuaikan ukuran jendela (default: `--show`) | Pakai `--no-show` untuk batch otomatis tanpa interaksi |
| `--template`, `--key`, `--out` | ganti path file/folder | Kalau lokasi file berbeda |

## 8. Masalah Umum
| Gejala | Solusi |
|---|---|
| "Kertas tidak terdeteksi" | Foto di latar kontras, semua sudut terlihat, atau gunakan `--scan` |
| Banyak soal terbaca kosong | Turunkan `--min-extra`; periksa `7_threshold.jpg` apakah tanda terlihat |
| Kotak baca meleset dari huruf | Cek `6_perspective_warp.jpg`; pastikan kertas utuh dan template sama dengan lembar |
| Memakai template/lembar lain | Ubah koordinat di bagian KONFIGURASI pada `main.py` (`ROW_Y0`, `ROW_Y_STEP`, `BLOCK_COLS_X`) |

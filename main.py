"""
SISTEM PEMERIKSAAN LEMBAR JAWABAN PILIHAN GANDA (50 soal, opsi A-E)
Praktikum Pengolahan Citra Digital - Python, OpenCV, NumPy

Alur:
  foto lembar -> grayscale -> gaussian blur -> canny -> kontur lembar (4 titik)
  -> perspective transform -> thresholding -> bandingkan dengan template kosong
  -> baca jawaban per soal -> cocokkan kunci (benar/salah) -> nilai

"""

import argparse
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np

# ============================================================ KONFIGURASI
OPTIONS = "ABCDE"
N_SOAL = 50
REF_W, REF_H = 768, 1024                 # ukuran standar template

# Koordinat huruf A-E pada template ukuran 768x1024 (diukur dari lembar UIN).
# Kalau memakai template lain, ubah blok ini saja.
ROW_Y0, ROW_Y_STEP, ROWS_PER_BLOCK = 305.9, 23.15, 25
BLOCK_COLS_X = [
    [125.8, 159.9, 195.1, 229.9, 263.8],     # blok kiri  : soal 1-25
    [482.8, 517.4, 557.7, 598.7, 638.3],     # blok kanan : soal 26-50
]
ROI_HALF_W, ROI_HALF_H = 16, 11          # setengah ukuran kotak baca per huruf

GREEN, RED, BLUE = (0, 170, 0), (0, 0, 220), (220, 110, 0)


def letter_positions():
    """50 soal x 5 huruf -> titik pusat (x, y) tiap huruf."""
    pos = []
    for cols in BLOCK_COLS_X:
        for r in range(ROWS_PER_BLOCK):
            y = ROW_Y0 + r * ROW_Y_STEP
            pos.append([(x, y) for x in cols])
    return pos


# ================================================================ FILE I/O
def load_key(path):
    """kunci.csv berformat: soal,jawaban  (header boleh ada)."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip().isdigit() and row[1].strip().upper() in OPTIONS:
                rows.append((int(row[0]), row[1].strip().upper()))
    key = [a for _, a in sorted(rows)]
    if len(key) != N_SOAL:
        sys.exit(f"[ERROR] {path} berisi {len(key)} soal, seharusnya {N_SOAL}. "
                 f"Format tiap baris: nomor,huruf  (contoh: 1,A)")
    return key


def load_truth(path):
    """truth.csv berformat: lembar,jawaban  (jawaban = 50 karakter: A-E, '-' kosong, '*' ganda)."""
    truth = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                s = row[1].strip().upper().replace(" ", "").replace(",", "")
                if len(s) == N_SOAL:
                    truth[os.path.splitext(row[0].strip())[0]] = s
    return truth


def list_images(path):
    if os.path.isdir(path):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            files += glob.glob(os.path.join(path, ext))
        return sorted(set(files))
    return [path]


# ======================================================= PENGOLAHAN CITRA
def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


def four_point_transform(image, pts, w, h):
    rect = order_points(pts)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))


def find_sheet_contour(edged):
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return approx
    if cnts:                                            # cadangan: kotak miring terkecil
        return cv2.boxPoints(cv2.minAreaRect(cnts[0])).astype("int32").reshape(4, 1, 2)
    return None


def preprocess_and_warp(image, scan=False):
    """Kembalikan (warped BGR ukuran template, dict gambar tiap langkah)."""
    steps = {}
    if scan:
        warped = cv2.resize(image, (REF_W, REF_H))
        return warped, steps

    target_h = 800
    ratio = image.shape[0] / float(target_h)
    small = cv2.resize(image, (int(image.shape[1] / ratio), target_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)                        # grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)                              # gaussian blur
    edged = cv2.Canny(blur, 75, 200)                                      # canny
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    sheet = find_sheet_contour(edged)                                     # kontur lembar
    if sheet is None:
        raise RuntimeError("Kertas tidak terdeteksi. Pakai latar kontras atau opsi --scan.")
    outline = small.copy()
    cv2.drawContours(outline, [sheet], -1, (0, 255, 0), 3)

    warped = four_point_transform(image, sheet.reshape(4, 2) * ratio, REF_W, REF_H)
    steps.update({"1_original": small, "2_grayscale": gray, "3_blur": blur,
                  "4_canny": edged, "5_kontur_lembar": outline})
    return warped, steps


def binarize(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)


def read_answers(ink_s, ink_t, positions, min_extra, dominance):
    """Tambahan tinta per huruf = tinta foto - tinta template (di kotak sekitar huruf)."""
    answers, all_counts = [], []
    for row in positions:
        counts = []
        for (x, y) in row:
            x0, x1 = int(x - ROI_HALF_W), int(x + ROI_HALF_W)
            y0, y1 = int(y - ROI_HALF_H), int(y + ROI_HALF_H)
            delta = cv2.countNonZero(ink_s[y0:y1, x0:x1]) - cv2.countNonZero(ink_t[y0:y1, x0:x1])
            counts.append(max(delta, 0))
        order = sorted(range(5), key=lambda i: counts[i], reverse=True)
        top, second = counts[order[0]], counts[order[1]]
        if top < min_extra:
            answers.append("-")                                  # kosong
        elif second >= min_extra and top < dominance * second:
            answers.append("*")                                  # ganda
        else:
            answers.append(OPTIONS[order[0]])
        all_counts.append(counts)
    return answers, all_counts


def score(answers, key):
    benar = salah = kosong = ganda = 0
    for a, k in zip(answers, key):
        if a == k:
            benar = benar + 1          # jawaban benar -> benar + 1
        else:
            salah = salah + 1          # jawaban salah -> salah + 1
            kosong += (a == "-")
            ganda += (a == "*")
    nilai = benar / float(len(key)) * 100
    return benar, salah, kosong, ganda, nilai


def box(img, x, y, color, th):
    cv2.rectangle(img, (int(x - ROI_HALF_W), int(y - ROI_HALF_H)),
                  (int(x + ROI_HALF_W), int(y + ROI_HALF_H)), color, th)


def annotate(warped, positions, answers, key, benar, salah, nilai):
    out = warped.copy()
    for row, a, k in zip(positions, answers, key):
        kx, ky = row[OPTIONS.index(k)]
        if a == k:
            box(out, kx, ky, GREEN, 2)                       # benar
        else:
            box(out, kx, ky, BLUE, 1)                        # posisi kunci
            if a in OPTIONS:
                x, y = row[OPTIONS.index(a)]
                box(out, x, y, RED, 2)                       # jawaban salah
    cv2.putText(out, f"Benar {benar}  Salah {salah}  Nilai {nilai:.1f}", (300, 252),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
    return out


# =================================================================== PROSES
def process_sheet(path, key, ink_t, positions, args, truth):
    name = os.path.splitext(os.path.basename(path))[0]
    image = cv2.imread(path)
    if image is None:
        print(f"  [{name}] gagal dibaca, dilewati")
        return None
    try:
        warped, steps = preprocess_and_warp(image, args.scan)
    except RuntimeError as e:
        print(f"  [{name}] {e}")
        return None

    ink_s = binarize(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    answers, _ = read_answers(ink_s, ink_t, positions, args.min_extra, args.dominance)
    benar, salah, kosong, ganda, nilai = score(answers, key)

    # gambar tiap langkah (bahan screenshot laporan)
    folder = os.path.join(args.out, name)
    os.makedirs(folder, exist_ok=True)
    steps["6_perspective_warp"] = warped
    steps["7_threshold"] = ink_s
    steps["8_hasil"] = annotate(warped, positions, answers, key, benar, salah, nilai)
    for k, img in steps.items():
        cv2.imwrite(os.path.join(folder, k + ".jpg"), img)

    with open(os.path.join(folder, "detail.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["soal", "kunci", "jawaban", "status"])
        for i, (a, k) in enumerate(zip(answers, key), 1):
            w.writerow([i, k, a, "benar" if a == k else "salah"])

    akurasi = ""
    if name in truth:
        cocok = sum(a == t for a, t in zip(answers, truth[name]))
        akurasi = f"{cocok}/{N_SOAL}"
    print(f"  [{name}] benar={benar:2d} salah={salah:2d} (kosong={kosong}, ganda={ganda}) "
          f"nilai={nilai:5.1f}" + (f"  baca-akurat={akurasi}" if akurasi else ""))
    return [name, benar, salah, kosong, ganda, round(nilai, 2), akurasi]


# ===================================================================== DEMO
def buat_data_demo(template_path, folder):
    """Buat 5 foto lembar terisi buatan + kunci + ground truth, untuk uji coba."""
    random.seed(7)
    np.random.seed(7)
    tpl = cv2.resize(cv2.imread(template_path), (REF_W, REF_H))
    pos = letter_positions()
    os.makedirs(folder, exist_ok=True)
    key = [random.choice(OPTIONS) for _ in range(N_SOAL)]
    key_path = os.path.join(folder, "kunci.csv")
    with open(key_path, "w", encoding="utf-8") as f:
        f.write("soal,jawaban\n" + "".join(f"{i},{k}\n" for i, k in enumerate(key, 1)))

    truth_rows = []
    for n in range(1, 6):
        img, ans = tpl.copy(), []
        p_benar = random.uniform(0.5, 0.95)
        blank_q, dbl_q = random.randrange(N_SOAL), random.randrange(N_SOAL)
        for q in range(N_SOAL):
            a = key[q] if random.random() < p_benar else random.choice(OPTIONS)
            if q == blank_q:
                a = "-"
            elif q == dbl_q:
                a = "*"
            ans.append(a)
            targets = ["A", "B"] if a == "*" else ([] if a == "-" else [a])
            for t in targets:
                x, y = pos[q][OPTIONS.index(t)]
                x, y = int(x + random.randint(-2, 2)), int(y + random.randint(-2, 2))
                style = random.choice(["lingkar", "silang", "arsir"])
                if style == "lingkar":
                    cv2.ellipse(img, (x, y), (13, 9), random.randint(-15, 15), 0, 360, (30, 30, 30), 2)
                elif style == "silang":
                    cv2.line(img, (x - 7, y - 7), (x + 7, y + 7), (30, 30, 30), 2)
                    cv2.line(img, (x - 7, y + 7), (x + 7, y - 7), (30, 30, 30), 2)
                else:
                    cv2.circle(img, (x, y), 7, (30, 30, 30), -1)
        j = lambda: random.randint(-35, 35)
        dst = np.float32([[160 + j(), 140 + j()], [960 + j(), 170 + j()],
                          [990 + j(), 1290 + j()], [120 + j(), 1240 + j()]])
        bg = np.full((1500, 1200, 3), 50, np.uint8)
        src = np.float32([[0, 0], [REF_W, 0], [REF_W, REF_H], [0, REF_H]])
        foto = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (1200, 1500),
                                   dst=bg, borderMode=cv2.BORDER_TRANSPARENT)
        grad = np.tile(np.linspace(random.uniform(0.65, 0.85), 1.05, 1200)[None, :, None], (1500, 1, 3))
        foto = np.clip(foto * grad + np.random.randint(-6, 6, foto.shape), 0, 255).astype(np.uint8)
        nama = f"demo_{n:02d}"
        cv2.imwrite(os.path.join(folder, nama + ".jpg"), foto)
        truth_rows.append([nama, "".join(ans)])

    with open(os.path.join(folder, "truth.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lembar", "jawaban"])
        w.writerows(truth_rows)
    return key_path, os.path.join(folder, "truth.csv")


# ===================================================================== MAIN
def main():
    ap = argparse.ArgumentParser(description="Sistem Pemeriksaan Lembar Jawaban Pilihan Ganda")
    ap.add_argument("input", nargs="?", default=os.path.join("dataset", "lembar"),
                    help="file foto atau folder foto (default: dataset/lembar)")
    ap.add_argument("--template", default=os.path.join("dataset", "template_kosong.jpeg"),
                    help="gambar lembar KOSONG")
    ap.add_argument("--key", default=os.path.join("dataset", "kunci.csv"), help="file kunci jawaban")
    ap.add_argument("--truth", default=None,
                    help="(opsional) truth.csv untuk menghitung akurasi pembacaan")
    ap.add_argument("--out", default="hasil", help="folder output (default: hasil)")
    ap.add_argument("--scan", action="store_true", help="gambar sudah scan / dipotong rapi")
    ap.add_argument("--min-extra", type=int, default=40,
                    help="minimal piksel tinta tambahan agar huruf dianggap ditandai")
    ap.add_argument("--dominance", type=float, default=3.0,
                    help="rasio tanda terkuat vs kedua; di bawah ini dianggap ganda")
    ap.add_argument("--demo", action="store_true", help="buat data buatan lalu jalankan")
    args = ap.parse_args()

    if not os.path.isfile(args.template):
        sys.exit(f"[ERROR] Template tidak ditemukan: {args.template}\n"
                 f"        Taruh gambar lembar KOSONG di sana atau tentukan dengan --template.")

    if args.demo:
        demo_dir = os.path.join("dataset", "demo")
        args.key, args.truth = buat_data_demo(args.template, demo_dir)
        args.input = demo_dir
        args.out = os.path.join(args.out, "demo")
        print(f"Data demo dibuat di {demo_dir}/\n")
    else:
        if not os.path.exists(args.input):
            sys.exit(f"[ERROR] Input tidak ditemukan: {args.input}")
        if not os.path.isfile(args.key):
            sys.exit(f"[ERROR] Kunci jawaban tidak ditemukan: {args.key}\n"
                     f"        Buat file CSV: satu baris per soal, contoh '1,A'.")

    key = load_key(args.key)
    truth = load_truth(args.truth) if args.truth and os.path.isfile(args.truth) else {}
    positions = letter_positions()
    os.makedirs(args.out, exist_ok=True)

    tpl = cv2.imread(args.template)
    tpl = cv2.resize(tpl, (REF_W, REF_H))
    ink_t = binarize(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY))

    files = [f for f in list_images(args.input)
             if os.path.basename(f).lower() not in ("template_kosong.jpeg", "template_kosong.jpg")]
    if not files:
        sys.exit(f"[ERROR] Tidak ada gambar (jpg/jpeg/png) di {args.input}")

    print(f"Memeriksa {len(files)} lembar...\n")
    summary = [r for r in (process_sheet(f, key, ink_t, positions, args, truth) for f in files) if r]

    if not summary:
        sys.exit("\nTidak ada lembar yang berhasil diproses.")

    with open(os.path.join(args.out, "ringkasan.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lembar", "benar", "salah", "kosong", "ganda", "nilai", "akurasi_baca"])
        w.writerows(summary)

    rata = sum(r[5] for r in summary) / len(summary)
    print(f"\nBerhasil: {len(summary)}/{len(files)} lembar | rata-rata nilai: {rata:.1f}")
    if truth:
        total = sum(int(r[6].split("/")[0]) for r in summary if r[6])
        jumlah = sum(N_SOAL for r in summary if r[6])
        if jumlah:
            print(f"Akurasi pembacaan jawaban: {total}/{jumlah} = {100.0 * total / jumlah:.2f}%")
    print(f"Hasil tersimpan di folder '{args.out}/' (ringkasan.csv + gambar tiap lembar)")


if __name__ == "__main__":
    main()

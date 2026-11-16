"""
SISTEM PEMERIKSAAN LEMBAR JAWABAN PILIHAN GANDA (50 soal, opsi A-E)
Praktikum Pengolahan Citra Digital - Python, OpenCV, NumPy, CSV

Alur:
  foto lembar -> grayscale -> gaussian blur -> canny -> kontur lembar (4 titik)
  -> perspective transform -> thresholding -> bandingkan dengan template kosong
  -> baca jawaban per soal -> cocokkan kunci (benar/salah) -> nilai

Cara jalan paling singkat (lihat README.md untuk lengkapnya):
    python main.py --demo          # coba dengan data buatan, tanpa perlu foto sendiri
    python main.py                 # periksa semua foto di dataset/lembar
    python main.py --scan          # kalau gambarnya hasil scan / sudah dipotong rapi
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
    """truth.csv: lembar,jawaban (50 char: A-E, '-' kosong/tanda salah, '*' ganda)."""
    truth = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                s = row[1].strip().upper().replace(" ", "").replace(",", "")
                if len(s) == N_SOAL:
                    # '*' pada truth berarti jawaban ganda (dianggap belum menjawab)
                    truth[os.path.splitext(row[0].strip())[0]] = s.replace("*", "-")
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


def roi_slice(x, y, w=None, h=None):
    hw = ROI_HALF_W if w is None else w
    hh = ROI_HALF_H if h is None else h
    return int(x - hw), int(x + hw), int(y - hh), int(y + hh)


def roi_mark_score(roi, mode):
    """Skor tanda pada ROI (0-1) sesuai mode instruksi."""
    if roi.size == 0:
        return 0.0
    h, w = roi.shape[:2]
    fill = cv2.countNonZero(roi) / float(w * h)
    edges = cv2.Canny(roi, 50, 150)
    edge_ratio = cv2.countNonZero(edges) / float(w * h)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), 8)
    thin = 0.0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        per = cv2.arcLength(cnts[0], True) + 1e-6
        circ = 4.0 * np.pi * area / (per * per)   # 1: bulat padat, kecil: memanjang/tipis
        thin = max(thin, 1.0 - min(max(circ, 0.0), 1.0))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(8, h // 2),
                            minLineLength=max(8, w // 3), maxLineGap=4)
    line_score = 0.0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang < 25 or ang > 155:
                line_score = max(line_score, np.hypot(x2 - x1, y2 - y1) / float(max(w, h)))

    if mode == "bulat_hitam":
        score = 0.75 * fill + 0.25 * max(0.0, fill - 0.35 * edge_ratio)
    elif mode == "bulat":
        ring = max(0.0, edge_ratio - 0.5 * fill)
        score = 0.55 * ring + 0.35 * fill + 0.10 * line_score
    elif mode == "silang":
        score = 0.6 * thin + 0.25 * edge_ratio + 0.15 * fill
    else:  # semua
        score = max(fill, 0.55 * fill + 0.45 * edge_ratio)
    return float(max(score, 0.0))


def longest_horizontal_run(roi):
    """Panjang deret piksel horizontal terpanjang dalam satu baris."""
    best = 0
    for row in roi:
        run = 0
        for v in row:
            run = run + 1 if v else 0
            if run > best:
                best = run
    return best


def _long_horizontal_groups(roi, frac=0.62):
    """Kelompok baris yang mengandung garis horizontal panjang."""
    h, w = roi.shape[:2]
    rows = []
    for r in range(h):
        run = best = 0
        for v in roi[r]:
            run = run + 1 if v else 0
            if run > best:
                best = run
        if best >= frac * w:
            rows.append(r)
    groups = []
    for r in rows:
        if groups and r - groups[-1][-1] <= 1:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def detect_crossed_out(roi):
    """Deteksi coretan pembatal '=' (dua garis horizontal berdekatan).

    Lingkaran juga punya sisi atas/bawah yang horizontal, tetapi jaraknya jauh;
    '=' memiliki dua garis panjang yang berdekatan (sekitar 6-8 piksel).
    """
    if roi.size == 0:
        return False
    if cv2.countNonZero(roi) < 20:
        return False
    groups = _long_horizontal_groups(roi)
    if len(groups) < 2:
        return False
    centers = [sum(g) / float(len(g)) for g in groups]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if 1.0 <= abs(centers[i] - centers[j]) <= 9.0:
                return True
    return False


def classify_mark(roi):
    """Tebak jenis tanda: 'silang', 'bulat' (lingkar), atau 'bulat_hitam'."""
    if roi.size == 0:
        return None
    h, w = roi.shape[:2]
    total = cv2.countNonZero(roi)
    if total < 15:
        return None
    fill = total / float(w * h)
    center = roi[h // 3:2 * h // 3, w // 3:2 * w // 3]
    center_fill = cv2.countNonZero(center) / float(max(1, center.shape[0] * center.shape[1]))
    edges = cv2.Canny(roi, 50, 150)
    edge_ratio = cv2.countNonZero(edges) / float(w * h)

    # ukuran kotak komponen terbesar: dipakai membedakan coretan gabungan
    num, labels, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), 8)
    aspect = 1.0
    if num > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bw = stats[k, cv2.CC_STAT_WIDTH]
        bh = stats[k, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / float(max(1, min(bw, bh)))

    # lingkaran berongga: isi cukup banyak tetapi bagian tengahnya relatif kosong
    if fill >= 0.24 and center_fill < 0.30:
        return "bulat"
    # bulat padat/arsir: tepi sedikit, isi padat, atau gabungan yang memanjang
    if ((edge_ratio < 0.10 and fill < 0.21)
            or (edge_ratio < 0.118 and fill < 0.16)
            or (edge_ratio < 0.12 and fill < 0.17 and center_fill >= 0.31)
            or (fill < 0.155 and edge_ratio < 0.15 and center_fill >= 0.31)
            or (fill >= 0.22 and aspect >= 1.75)):
        return "bulat_hitam"
    return "silang"


def read_answers(ink_s, ink_t, positions, min_extra, dominance, mode="semua"):
    """Baca jawaban per soal. Jawaban ganda valid dianggap belum menjawab;
    bila mode spesifik, hanya tanda yang sesuai instruksi yang dianggap sah."""
    answers, notes, all_counts = [], [], []
    for row in positions:
        counts, crossed, styles, fills = [], [], [], []
        for (x, y) in row:
            x0, x1, y0, y1 = roi_slice(x, y)
            roi_s = ink_s[y0:y1, x0:x1]
            roi_t = ink_t[y0:y1, x0:x1]
            delta = cv2.subtract(roi_s, roi_t)
            cnt = max(cv2.countNonZero(delta), 0)
            h, w = delta.shape[:2]
            counts.append(cnt)
            fills.append(cnt / float(max(h * w, 1)))
            crossed.append(detect_crossed_out(roi_s))
            styles.append(classify_mark(delta))

        # saringan tinta: cukup banyak DAN cukup padat, supaya noise/artefak tipis tidak ikut terbaca
        effective = [counts[i] if (counts[i] >= min_extra and fills[i] >= 0.08) else 0
                     for i in range(5)]
        order = sorted(range(5), key=lambda i: effective[i], reverse=True)
        top, second = effective[order[0]], effective[order[1]]
        marked = [i for i in order if effective[i] > 0]

        if mode != "semua":
            # hanya tanda yang sesuai instruksi DAN tidak dicoret yang dihitung sebagai jawaban
            matching = [i for i in marked if styles[i] == mode]
            matching_active = [i for i in matching if not crossed[i]]
            if not matching_active:
                if marked:
                    wrong = styles[marked[0]] or "tidak jelas"
                    answers.append("-")
                    notes.append(f"tanda_salah:{wrong}")
                else:
                    answers.append("-")
                    notes.append("kosong")
                all_counts.append(effective)
                continue

            m_order = sorted(matching_active, key=lambda i: effective[i], reverse=True)
            is_ganda = (len(m_order) >= 2 and effective[m_order[1]] >= min_extra
                        and effective[m_order[0]] < dominance * effective[m_order[1]])
            if is_ganda:
                active = [i for i in m_order if not crossed[i]]
                if len(active) == 1:
                    answers.append(OPTIONS[active[0]])
                    notes.append(f"ganda_satu_dicoret:{OPTIONS[active[0]]}")
                else:
                    pool = active if active else m_order
                    answers.append("-")
                    notes.append("ganda:" + "".join(OPTIONS[i] for i in pool[:3]))
            else:
                answers.append(OPTIONS[m_order[0]])
                notes.append("terbaca")
            all_counts.append(effective)
            continue

        if top < min_extra:
            answers.append("-")
            notes.append("kosong")
        elif second >= min_extra and top < dominance * second:
            # ganda: kalau salah satu yang ditandai dicoret (=), bukan ganda
            active = [i for i in marked if not crossed[i]]
            if len(active) == 1 and len(marked) >= 2:
                answers.append(OPTIONS[active[0]])
                notes.append(f"ganda_satu_dicoret:{OPTIONS[active[0]]}")
            elif len(active) >= 2:
                answers.append("-")
                marked_letters = "".join(OPTIONS[i] for i in active[:3])
                notes.append(f"ganda:{marked_letters}")
            else:
                # semua tertanda tercoret/rancu -> jangan pilih yang dicoret, anggap ganda/belum jawab
                answers.append("-")
                marked_letters = "".join(OPTIONS[i] for i in marked[:3])
                notes.append(f"ganda:{marked_letters}")
        else:
            answers.append(OPTIONS[order[0]])
            notes.append("terbaca")
        all_counts.append(effective)
    return answers, notes, all_counts


def score(answers, key, notes=None):
    benar = salah = kosong = ganda = 0
    notes = notes or ["" for _ in answers]
    for a, k, note in zip(answers, key, notes):
        is_ganda = note.startswith("ganda:")
        is_tanda_salah = note.startswith("tanda_salah")
        if a == k:
            benar += 1                 # jawaban benar -> benar + 1
        else:
            salah += 1                 # jawaban salah/kosong/ganda -> salah + 1
            if is_ganda:
                ganda += 1             # jawaban ganda dianggap belum menjawab
            elif is_tanda_salah:
                pass                   # tanda tidak sesuai instruksi -> salah, bukan kosong
            elif a == "-":
                kosong += 1
    nilai = benar / float(len(key)) * 100
    return benar, salah, kosong, ganda, nilai


def box(img, x, y, color, th):
    cv2.rectangle(img, (int(x - ROI_HALF_W), int(y - ROI_HALF_H)),
                  (int(x + ROI_HALF_W), int(y + ROI_HALF_H)), color, th)


def annotate(warped, positions, answers, key, notes, benar, salah, kosong, ganda, nilai):
    out = warped.copy()
    for row, a, k, note in zip(positions, answers, key, notes):
        kx, ky = row[OPTIONS.index(k)]
        if a == k:
            box(out, kx, ky, GREEN, 2)                       # benar
        else:
            box(out, kx, ky, BLUE, 1)                        # posisi kunci
            if a in OPTIONS:
                x, y = row[OPTIONS.index(a)]
                box(out, x, y, RED, 2)                       # jawaban salah
        ax, ay = row[0]
        if note.startswith("ganda:"):
            cv2.putText(out, "GANDA", (int(ax - 78), int(ay + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 2)
        elif note.startswith("tanda_salah"):
            cv2.putText(out, "TANDA SALAH", (int(ax - 108), int(ay + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 140, 255), 2)
        elif a == "-":
            cv2.putText(out, "KOSONG", (int(ax - 88), int(ay + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 0, 0), 2)
    cv2.putText(out, f"Benar {benar}  Salah {salah}  Kosong {kosong}  Ganda {ganda}  Nilai {nilai:.1f}", (230, 252),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    # legenda warna di bawah hasil (2 baris agar muat)
    out = cv2.copyMakeBorder(out, 0, 60, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    legend = [
        (GREEN, "Benar"),
        (RED, "Jawaban salah"),
        (BLUE, "Posisi kunci"),
        ((0, 0, 220), "Ganda/belum jawab"),
        ((180, 0, 0), "Kosong"),
        ((0, 140, 255), "Tanda tidak sesuai"),
    ]
    for i, (color, text) in enumerate(legend):
        col, rowi = i % 3, i // 3
        x = 24 + col * 250
        y = out.shape[0] - 40 + rowi * 24
        cv2.rectangle(out, (x, y - 11), (x + 16, y + 3), color, -1)
        cv2.putText(out, text, (x + 22, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
    return out


def make_steps_figure(steps, title, cell_w=360, cell_h=480, cols=4):
    """Gabungkan semua langkah jadi satu figure bergrid berlabel."""
    labels = {
        "1_original": "1. Original",
        "2_grayscale": "2. Grayscale",
        "3_blur": "3. Gaussian Blur",
        "4_canny": "4. Canny",
        "5_kontur_lembar": "5. Kontur Lembar",
        "6_perspective_warp": "6. Perspective Warp",
        "7_threshold": "7. Threshold",
        "8_hasil": "8. Hasil",
    }
    items = [(labels.get(k, k), img) for k, img in steps.items()]
    if not items:
        return None

    cols = max(1, min(cols, len(items)))
    rows = (len(items) + cols - 1) // cols
    label_h, title_h, pad = 34, 46, 10

    tiles = []
    for label, img in items:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = min((cell_w - 2 * pad) / w, (cell_h - label_h - 2 * pad) / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        tile = np.full((cell_h, cell_w, 3), 255, np.uint8)
        x, y = (cell_w - nw) // 2, label_h + (cell_h - label_h - nh) // 2
        tile[y:y + nh, x:x + nw] = resized
        cv2.putText(tile, label, (pad, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (cell_w - 1, cell_h - 1), (200, 200, 200), 1)
        tiles.append(tile)

    fig_w = cols * cell_w + (cols + 1) * pad
    fig_h = title_h + rows * cell_h + (rows + 1) * pad
    fig = np.full((fig_h, fig_w, 3), 255, np.uint8)
    cv2.putText(fig, f"Tahapan Pengolahan Citra - {title}", (pad, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = title_h + pad + r * (cell_h + pad)
        fig[y:y + cell_h, x:x + cell_w] = tile
    return fig


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
    answers, notes, _ = read_answers(ink_s, ink_t, positions, args.min_extra, args.dominance, args.mode)
    benar, salah, kosong, ganda, nilai = score(answers, key, notes)

    folder = os.path.join(args.out, name)
    os.makedirs(folder, exist_ok=True)
    steps["6_perspective_warp"] = warped
    steps["7_threshold"] = ink_s
    steps["8_hasil"] = annotate(warped, positions, answers, key, notes, benar, salah, kosong, ganda, nilai)
    figure = make_steps_figure(steps, name)
    cv2.imwrite(os.path.join(folder, "figure_tahapan.jpg"), figure)

    with open(os.path.join(folder, "detail.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["soal", "kunci", "jawaban", "status", "keterangan"])
        for i, (a, k, note) in enumerate(zip(answers, key, notes), 1):
            is_ganda = note.startswith("ganda:")
            is_tanda_salah = note.startswith("tanda_salah")
            if a == k:
                status = "benar"
            elif is_ganda:
                status = "ganda_belum_menjawab"
            elif is_tanda_salah:
                status = "tanda_tidak_sesuai"
            elif a == "-":
                status = "kosong"
            else:
                status = "salah"
            if is_ganda:
                ket = "jawaban ganda, dianggap belum menjawab"
            elif is_tanda_salah:
                jenis = note.split(":", 1)[1] if ":" in note else "?"
                ket = f"tanda tidak sesuai instruksi (terdeteksi: {jenis})"
            else:
                ket = note
            w.writerow([i, k, "" if a == "-" else a, status, ket])

    akurasi = ""
    if name in truth:
        truth_ans = truth[name] if isinstance(truth[name], str) else truth[name][0]
        cocok = sum(a == t for a, t in zip(answers, truth_ans))
        akurasi = f"{cocok}/{N_SOAL}"
    n_tanda_salah = sum(note.startswith("tanda_salah") for note in notes)
    print(f"  [{name}] benar={benar:2d} salah={salah:2d} "
          f"(kosong={kosong}, ganda={ganda}, tanda_salah={n_tanda_salah}) "
          f"nilai={nilai:5.1f} mode={args.mode}" + (f"  baca-akurat={akurasi}" if akurasi else ""))
    return [name, benar, salah, kosong, ganda, round(nilai, 2), akurasi]


# ===================================================================== DEMO
def draw_mark(img, x, y, style):
    x, y = int(x + random.randint(-2, 2)), int(y + random.randint(-2, 2))
    if style == "lingkar":
        cv2.ellipse(img, (x, y), (13, 9), random.randint(-15, 15), 0, 360, (30, 30, 30), 2)
    elif style == "silang":
        cv2.line(img, (x - 7, y - 7), (x + 7, y + 7), (30, 30, 30), 2)
        cv2.line(img, (x - 7, y + 7), (x + 7, y - 7), (30, 30, 30), 2)
    elif style == "bulat_hitam":
        cv2.circle(img, (x, y), 7, (30, 30, 30), -1)
    else:  # arsir/bulat tipis
        cv2.circle(img, (x, y), 7, (30, 30, 30), 2)


def buat_data_demo(template_path, folder, mode="semua"):
    """Buat 5 foto lembar terisi buatan (tanda bervariasi) + kunci + truth.
    Jika instruksi spesifik, mayoritas tanda memakai gaya tersebut, sisanya variasi
    untuk memperlihatkan penolakan tanda yang tidak sesuai instruksi. truth.csv
    berisi hasil baca yang diharapkan (tanda tak sesuai mode ditulis '-')."""
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
        img, ans, read_ans = tpl.copy(), [], []
        p_benar = random.uniform(0.7, 0.95)
        blank_q, dbl_q, crossed_q = random.randrange(N_SOAL), random.randrange(N_SOAL), random.randrange(N_SOAL)
        for q in range(N_SOAL):
            a = key[q] if random.random() < p_benar else random.choice(OPTIONS)
            crossed = None
            if q == blank_q:
                a = "-"
            elif q == dbl_q:
                a = "*"                         # ganda murni -> belum menjawab
            elif q == crossed_q:
                crossed = random.choice([o for o in OPTIONS if o != a])  # a tetap jawaban, crossed dicoret
            ans.append(a)

            # demo bervariasi: mayoritas mengikuti instruksi (jika spesifik), sisanya gaya lain
            semua_gaya = ["lingkar", "silang", "bulat_hitam"]
            # mode "bulat" digambar sebagai "lingkar", sesuai hasil klasifikasi
            draw_style = "lingkar" if mode == "bulat" else mode
            if mode == "semua":
                style = random.choice(semua_gaya)
            elif random.random() < 0.85:
                style = draw_style
            else:
                style = random.choice([g for g in semua_gaya if g != draw_style])

            # yang seharusnya terbaca: tanda dengan gaya tidak sesuai instruksi -> kosong/tanda_salah
            if a == "-":
                read_ans.append("-")
            elif mode == "semua" or style == draw_style:
                read_ans.append(a)
            else:
                read_ans.append("-")

            targets = ["A", "B"] if a == "*" else ([] if a == "-" else [a])
            for t in targets:
                x, y = pos[q][OPTIONS.index(t)]
                draw_mark(img, x, y, style)
            if crossed:
                x, y = pos[q][OPTIONS.index(crossed)]
                draw_mark(img, x, y, style)
                # coretan '=' agar jelas terbaca sebagai pembatal
                cv2.line(img, (int(x - 11), int(y - 3)), (int(x + 11), int(y - 3)), (30, 30, 30), 2)
                cv2.line(img, (int(x - 11), int(y + 3)), (int(x + 11), int(y + 3)), (30, 30, 30), 2)

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
        truth_rows.append([nama, "".join(read_ans)])

    with open(os.path.join(folder, "truth.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lembar", "jawaban"])
        w.writerows(truth_rows)
    return key_path, os.path.join(folder, "truth.csv")


# ===================================================================== MAIN
def pilih_mode_interaktif():
    """Menu pilihan instruksi tanda saat demo dijalankan."""
    pilihan = {"1": "silang", "2": "bulat", "3": "bulat_hitam", "4": "semua"}
    print("Pilih instruksi penandaan jawaban untuk demo:")
    print("  1. Hanya silang (X)")
    print("  2. Hanya bulat / lingkari")
    print("  3. Hanya bulat hitam / arsir")
    print("  4. Boleh semua (campuran)")
    while True:
        try:
            jawab = input("Masukkan nomor pilihan [1-4] (Enter = 4): ").strip()
        except EOFError:
            return "semua"
        if jawab == "":
            return "semua"
        if jawab in pilihan:
            return pilihan[jawab]
        print("  Pilihan tidak valid, ketik angka 1-4.")


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
    ap.add_argument("--mode", default=None,
                    choices=["silang", "bulat", "bulat_hitam", "semua"],
                    help="instruksi penandaan jawaban: silang, bulat, bulat_hitam, atau semua "
                         "(default: ditanya dulu saat --demo, selain itu 'semua')")
    ap.add_argument("--min-extra", type=int, default=40,
                    help="minimal piksel tinta tambahan agar huruf dianggap ditandai")
    ap.add_argument("--dominance", type=float, default=3.0,
                    help="rasio tanda terkuat vs kedua; di bawah ini dianggap ganda")
    ap.add_argument("--demo", action="store_true", help="buat data buatan lalu jalankan")
    args = ap.parse_args()

    if not os.path.isfile(args.template):
        sys.exit(f"[ERROR] Template tidak ditemukan: {args.template}\n"
                 f"        Taruh gambar lembar KOSONG di sana atau tentukan dengan --template.")

    if args.demo and args.mode is None:
        args.mode = pilih_mode_interaktif()
    elif args.mode is None:
        args.mode = "semua"

    if args.demo:
        demo_dir = os.path.join("dataset", "demo")
        args.key, args.truth = buat_data_demo(args.template, demo_dir, args.mode)
        args.input = demo_dir
        args.out = os.path.join(args.out, "demo")
        print(f"Data demo dibuat di {demo_dir}/ (tanda bervariasi, instruksi: {args.mode})")
        if args.mode != "semua":
            print(f"Catatan: mayoritas tanda memakai '{args.mode}'; sebagian sengaja gaya lain "
                  f"dan dihitung SALAH (tanda tidak sesuai instruksi).")
        print()
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
    print(f"\nBerhasil: {len(summary)}/{len(files)} lembar | mode={args.mode} | rata-rata nilai: {rata:.1f}")
    if truth:
        total = sum(int(r[6].split("/")[0]) for r in summary if r[6])
        jumlah = sum(N_SOAL for r in summary if r[6])
        if jumlah:
            print(f"Akurasi pembacaan jawaban: {total}/{jumlah} = {100.0 * total / jumlah:.2f}%")
    print(f"Hasil tersimpan di folder '{args.out}/' (ringkasan.csv + figure tiap lembar)")


if __name__ == "__main__":
    main()

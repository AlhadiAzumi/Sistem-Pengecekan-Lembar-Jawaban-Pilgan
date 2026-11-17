"""
Web UI untuk Sistem Pemeriksaan Lembar Jawaban Pilihan Ganda
Jalankan: python app.py
Lalu buka browser: http://localhost:5000
"""

import base64
import csv
import io
import os
import sys
import tempfile

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template_string, request, url_for

# ── tambahkan direktori ini ke path supaya bisa import main ──────────────────
sys.path.insert(0, os.path.dirname(__file__))
import main as m

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "dataset", "template_kosong.jpeg")
KEY_PATH      = os.path.join(os.path.dirname(__file__), "dataset", "kunci.csv")
HASIL_DIR     = os.path.join(os.path.dirname(__file__), "hasil")

# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pemeriksaan Pilihan Ganda</title>
<style>
  :root{--blue:#3b82f6;--green:#22c55e;--red:#ef4444;--gray:#6b7280;--bg:#f8fafc;--card:#fff;--border:#e2e8f0}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:var(--bg);color:#1e293b}
  header{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:#fff;padding:1.2rem 2rem;display:flex;align-items:center;gap:1rem}
  header h1{font-size:1.4rem}header p{font-size:.85rem;opacity:.85}
  .container{max-width:1100px;margin:2rem auto;padding:0 1rem}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.07)}
  .card h2{font-size:1rem;font-weight:600;margin-bottom:1rem;color:#1e3a8a}
  .form-row{display:flex;gap:1rem;flex-wrap:wrap;align-items:flex-end}
  .form-group{display:flex;flex-direction:column;gap:.35rem;flex:1;min-width:160px}
  label{font-size:.82rem;font-weight:600;color:var(--gray)}
  select,input[type=file],input[type=number],input[type=text]{border:1px solid var(--border);border-radius:8px;padding:.5rem .75rem;font-size:.9rem;width:100%;background:#fff}
  select:focus,input:focus{outline:2px solid var(--blue);border-color:var(--blue)}
  .btn{padding:.55rem 1.4rem;border:none;border-radius:8px;font-size:.9rem;font-weight:600;cursor:pointer;transition:.15s}
  .btn-primary{background:var(--blue);color:#fff}.btn-primary:hover{background:#2563eb}
  .btn-demo{background:#7c3aed;color:#fff}.btn-demo:hover{background:#6d28d9}
  .btn-sm{padding:.35rem .9rem;font-size:.8rem}
  /* summary bar */
  .summary{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.2rem}
  .stat{flex:1;min-width:110px;border-radius:10px;padding:1rem;text-align:center;color:#fff}
  .stat-benar{background:var(--green)}.stat-salah{background:var(--red)}
  .stat-kosong{background:var(--gray)}.stat-nilai{background:var(--blue)}
  .stat-ganda{background:#f59e0b}
  .stat .num{font-size:1.8rem;font-weight:700}.stat .lbl{font-size:.75rem;opacity:.9}
  /* tabs */
  .tabs{display:flex;gap:.5rem;margin-bottom:1rem;border-bottom:2px solid var(--border);padding-bottom:0}
  .tab{padding:.5rem 1.1rem;border-radius:8px 8px 0 0;cursor:pointer;font-size:.88rem;font-weight:600;color:var(--gray);background:transparent;border:1px solid transparent}
  .tab.active{background:var(--blue);color:#fff;border-color:var(--blue)}
  .tab-content{display:none}.tab-content.active{display:block}
  /* figure grid */
  .fig-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1rem}
  .fig-item{border:1px solid var(--border);border-radius:10px;overflow:hidden;background:#fff}
  .fig-item .fig-label{padding:.5rem .75rem;font-size:.8rem;font-weight:600;background:#f1f5f9;color:#334155}
  .fig-item img{width:100%;display:block;cursor:zoom-in;transition:.15s}
  .fig-item img:hover{opacity:.88}
  /* table */
  .tbl-wrap{overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:.85rem}
  th{background:#f1f5f9;padding:.5rem .75rem;text-align:left;position:sticky;top:0}
  td{padding:.45rem .75rem;border-top:1px solid var(--border)}
  tr:hover td{background:#f8fafc}
  .badge{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.75rem;font-weight:600}
  .badge-benar{background:#dcfce7;color:#15803d}
  .badge-salah{background:#fee2e2;color:#b91c1c}
  .badge-kosong{background:#f1f5f9;color:#475569}
  .badge-ganda{background:#fef3c7;color:#92400e}
  .badge-tanda{background:#ede9fe;color:#6d28d9}
  /* lightbox */
  #lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:9999;align-items:center;justify-content:center;cursor:zoom-out}
  #lb.show{display:flex}
  #lb img{max-width:95vw;max-height:92vh;border-radius:8px}
  /* alert */
  .alert{padding:.8rem 1rem;border-radius:8px;margin-bottom:1rem;font-size:.88rem}
  .alert-err{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5}
  .alert-ok{background:#dcfce7;color:#166534;border:1px solid #86efac}
  /* multi results */
  .result-nav{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}
  .result-nav .btn{font-size:.78rem}
  /* spinner */
  #spinner{display:none;position:fixed;inset:0;background:rgba(255,255,255,.75);z-index:8888;align-items:center;justify-content:center;flex-direction:column;gap:1rem}
  #spinner.show{display:flex}
  .spin-ring{width:56px;height:56px;border:6px solid #ddd;border-top-color:var(--blue);border-radius:50%;animation:spin 1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  #spinner p{color:#334155;font-weight:600}
</style>
</head>
<body>

<header>
  <div>
    <h1>🎓 Pemeriksaan Lembar Jawaban Pilihan Ganda</h1>
    <p>Praktikum Pengolahan Citra Digital &mdash; upload foto → lihat hasil langsung di browser</p>
  </div>
</header>

<div id="spinner"><div class="spin-ring"></div><p>Memproses gambar…</p></div>

<div id="lb" onclick="this.classList.remove('show')">
  <img id="lb-img" src="" alt="preview">
</div>

<div class="container">

  <!-- ── FORM ── -->
  <div class="card">
    <h2>⚙️ Pengaturan &amp; Upload Lembar</h2>

    {% if err %}<div class="alert alert-err">{{ err }}</div>{% endif %}

    <form id="mainForm" method="POST" action="/proses" enctype="multipart/form-data">
      <div class="form-row" style="margin-bottom:1rem">
        <div class="form-group" style="flex:2">
          <label>📄 Foto Lembar (jpg/jpeg/png, boleh lebih dari 1)</label>
          <input type="file" name="lembar" multiple accept=".jpg,.jpeg,.png">
        </div>
        <div class="form-group">
          <label>🖼️ Template Kosong</label>
          <input type="file" name="template" accept=".jpg,.jpeg,.png">
          <span style="font-size:.75rem;color:var(--gray)">kosongkan = pakai default</span>
        </div>
        <div class="form-group">
          <label>🔑 Kunci Jawaban (CSV)</label>
          <input type="file" name="kunci" accept=".csv">
          <span style="font-size:.75rem;color:var(--gray)">kosongkan = pakai default</span>
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>📝 Mode Tanda</label>
          <select name="mode">
            <option value="semua">Boleh semua (campuran)</option>
            <option value="silang">Hanya silang (X)</option>
            <option value="bulat">Hanya bulat / lingkari</option>
            <option value="bulat_hitam">Hanya bulat hitam / arsir</option>
          </select>
        </div>
        <div class="form-group">
          <label>🔍 Minimal piksel tinta</label>
          <input type="number" name="min_extra" value="40" min="10" max="200">
        </div>
        <div class="form-group">
          <label>⚖️ Rasio dominan</label>
          <input type="number" name="dominance" value="3.0" min="1.5" max="10" step="0.5">
        </div>
        <div class="form-group">
          <label>📡 Mode scan (sudah rapi)</label>
          <select name="scan">
            <option value="">Tidak (foto biasa)</option>
            <option value="1">Ya (hasil scan / dipotong)</option>
          </select>
        </div>
        <div class="form-group" style="justify-content:flex-end">
          <button type="submit" class="btn btn-primary" onclick="showSpinner()">🔎 Proses Lembar</button>
        </div>
      </div>
    </form>

    <hr style="margin:1.2rem 0;border:none;border-top:1px solid var(--border)">

    <form method="POST" action="/demo" id="demoForm">
      <div class="form-row" style="align-items:center">
        <div class="form-group">
          <label>🧪 Mode Demo</label>
          <select name="mode_demo">
            <option value="semua">Campuran</option>
            <option value="silang">Silang</option>
            <option value="bulat">Bulat / lingkar</option>
            <option value="bulat_hitam">Bulat hitam / arsir</option>
          </select>
        </div>
        <div class="form-group" style="justify-content:flex-end">
          <button type="submit" class="btn btn-demo" onclick="showSpinner()">🎲 Jalankan Demo (data buatan)</button>
        </div>
      </div>
    </form>
  </div>

  <!-- ── HASIL ── -->
  {% if results %}
  <div class="card">
    <h2>📊 Hasil Pemeriksaan — {{ results|length }} lembar</h2>

    {% if results|length > 1 %}
    <div class="result-nav">
      {% for r in results %}
      <button class="btn btn-sm {% if loop.first %}btn-primary{% endif %}"
              onclick="showResult({{ loop.index0 }}, this)">{{ r.name }}</button>
      {% endfor %}
    </div>
    {% endif %}

    {% for r in results %}
    <div class="result-block" id="rb-{{ loop.index0 }}"
         style="{{ 'display:block' if loop.first else 'display:none' }}">

      <!-- summary bar -->
      <div class="summary">
        <div class="stat stat-benar"><div class="num">{{ r.benar }}</div><div class="lbl">Benar</div></div>
        <div class="stat stat-salah"><div class="num">{{ r.salah }}</div><div class="lbl">Salah</div></div>
        <div class="stat stat-kosong"><div class="num">{{ r.kosong }}</div><div class="lbl">Kosong</div></div>
        <div class="stat stat-ganda"><div class="num">{{ r.ganda }}</div><div class="lbl">Ganda</div></div>
        <div class="stat stat-nilai"><div class="num">{{ "%.1f"|format(r.nilai) }}</div><div class="lbl">Nilai</div></div>
      </div>

      <!-- tabs -->
      <div class="tabs">
        <div class="tab active" onclick="switchTab(this,'fig-{{ loop.index0 }}')">🖼️ Tahapan Citra</div>
        <div class="tab" onclick="switchTab(this,'tbl-{{ loop.index0 }}')">📋 Detail Jawaban</div>
      </div>

      <!-- figure grid -->
      <div class="tab-content active" id="fig-{{ loop.index0 }}">
        <div class="fig-grid">
          {% for step in r.steps %}
          <div class="fig-item">
            <div class="fig-label">{{ step.label }}</div>
            <img src="data:image/jpeg;base64,{{ step.b64 }}"
                 alt="{{ step.label }}"
                 onclick="zoomImg(this.src)"
                 title="Klik untuk perbesar">
          </div>
          {% endfor %}
        </div>
      </div>

      <!-- detail table -->
      <div class="tab-content" id="tbl-{{ loop.index0 }}">
        <div class="tbl-wrap">
          <table>
            <thead>
              <tr><th>#</th><th>Kunci</th><th>Jawaban</th><th>Status</th><th>Keterangan</th></tr>
            </thead>
            <tbody>
              {% for row in r.rows %}
              <tr>
                <td>{{ row.soal }}</td>
                <td><strong>{{ row.kunci }}</strong></td>
                <td>{{ row.jawaban or '–' }}</td>
                <td>
                  {% if row.status == 'benar' %}<span class="badge badge-benar">✓ Benar</span>
                  {% elif row.status == 'salah' %}<span class="badge badge-salah">✗ Salah</span>
                  {% elif row.status == 'kosong' %}<span class="badge badge-kosong">— Kosong</span>
                  {% elif row.status == 'ganda_belum_menjawab' %}<span class="badge badge-ganda">⚡ Ganda</span>
                  {% else %}<span class="badge badge-tanda">⚠ Tanda salah</span>{% endif %}
                </td>
                <td style="color:var(--gray)">{{ row.ket }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>

    </div><!-- /result-block -->
    {% endfor %}

  </div>
  {% endif %}

</div><!-- /container -->

<script>
function showSpinner(){document.getElementById('spinner').classList.add('show')}
function zoomImg(src){var lb=document.getElementById('lb');document.getElementById('lb-img').src=src;lb.classList.add('show')}
function switchTab(el,targetId){
  var card=el.closest('.card');
  card.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  card.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.getElementById(targetId).classList.add('active');
}
function showResult(idx,btn){
  document.querySelectorAll('.result-block').forEach(b=>b.style.display='none');
  document.getElementById('rb-'+idx).style.display='block';
  btn.closest('.result-nav').querySelectorAll('.btn').forEach(b=>b.classList.remove('btn-primary'));
  btn.classList.add('btn-primary');
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────── helpers ────────────────────

def img_to_b64(img, quality=85):
    """OpenCV ndarray → base64 JPEG string."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf).decode()


def make_args(mode="semua", min_extra=40, dominance=3.0, scan=False):
    """Buat namespace argparse ringan."""
    import types
    args = types.SimpleNamespace(
        mode=mode,
        min_extra=min_extra,
        dominance=dominance,
        scan=scan,
        show=False,
        out=HASIL_DIR,
    )
    return args


def proses_satu(img_path, key, ink_t, positions, args, truth=None):
    """Proses satu lembar, kembalikan dict hasil + step images."""
    truth = truth or {}
    name = os.path.splitext(os.path.basename(img_path))[0]
    image = cv2.imread(img_path)
    if image is None:
        return None, f"Gagal membaca gambar: {img_path}"
    try:
        warped, steps = m.preprocess_and_warp(image, args.scan)
    except RuntimeError as e:
        return None, str(e)

    ink_s = m.binarize(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    answers, notes, _ = m.read_answers(ink_s, ink_t, positions, args.min_extra, args.dominance, args.mode)
    benar, salah, kosong, ganda, nilai = m.score(answers, key, notes)

    steps["6_perspective_warp"] = warped
    steps["7_threshold"]        = ink_s
    steps["8_hasil"] = m.annotate(warped, positions, answers, key, notes, benar, salah, kosong, ganda, nilai, args.mode)

    # ── step images untuk web ────────────────────────────────────────────────
    LABELS = {
        "1_original":        "1. Original",
        "2_grayscale":       "2. Grayscale",
        "3_blur":            "3. Gaussian Blur",
        "4_canny":           "4. Canny Edge",
        "5_kontur_lembar":   "5. Kontur Lembar",
        "6_perspective_warp":"6. Perspective Warp",
        "7_threshold":       "7. Adaptive Threshold",
        "8_hasil":           "8. Hasil Anotasi",
    }
    step_imgs = [{"label": LABELS.get(k, k), "b64": img_to_b64(v)}
                 for k, v in steps.items()]

    # ── detail per soal ──────────────────────────────────────────────────────
    rows = []
    for i, (a, k_ans, note) in enumerate(zip(answers, key, notes), 1):
        is_ganda      = note.startswith("ganda:")
        is_tanda_salah= note.startswith("tanda_salah")
        if a == k_ans:
            status = "benar"
        elif is_ganda:
            status = "ganda_belum_menjawab"
        elif is_tanda_salah:
            status = "tanda_tidak_sesuai"
        elif a == "-":
            status = "kosong"
        else:
            status = "salah"
        ket = note if not is_ganda else "jawaban ganda, dianggap belum menjawab"
        rows.append({"soal": i, "kunci": k_ans,
                     "jawaban": a if a != "-" else "",
                     "status": status, "ket": ket})

    # ── simpan ke disk (opsional) ────────────────────────────────────────────
    folder = os.path.join(args.out, name)
    os.makedirs(folder, exist_ok=True)
    fig = m.make_steps_figure(steps, name)
    if fig is not None:
        cv2.imwrite(os.path.join(folder, "figure_tahapan.jpg"), fig)
    with open(os.path.join(folder, "detail.csv"), "w", newline="", encoding="utf-8") as f:
        wt = csv.writer(f)
        wt.writerow(["soal","kunci","jawaban","status","keterangan"])
        for row in rows:
            wt.writerow([row["soal"], row["kunci"], row["jawaban"], row["status"], row["ket"]])

    return dict(name=name, benar=benar, salah=salah, kosong=kosong,
                ganda=ganda, nilai=nilai, steps=step_imgs, rows=rows), None


# ─────────────────────────────────────────────── routes ─────────────────────

@app.route("/")
def index():
    return render_template_string(HTML, results=[], err=None)


@app.route("/proses", methods=["POST"])
def proses():
    # ── template ─────────────────────────────────────────────────────────────
    tpl_file = request.files.get("template")
    if tpl_file and tpl_file.filename:
        tpl_buf = np.frombuffer(tpl_file.read(), np.uint8)
        tpl     = cv2.imdecode(tpl_buf, cv2.IMREAD_COLOR)
    else:
        if not os.path.isfile(TEMPLATE_PATH):
            return render_template_string(HTML, results=[], err="Template kosong tidak ditemukan. Upload manual atau taruh di dataset/template_kosong.jpeg")
        tpl = cv2.imread(TEMPLATE_PATH)
    tpl   = cv2.resize(tpl, (m.REF_W, m.REF_H))
    ink_t = m.binarize(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY))

    # ── kunci ─────────────────────────────────────────────────────────────────
    kunci_file = request.files.get("kunci")
    if kunci_file and kunci_file.filename:
        tmp_key = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        kunci_file.save(tmp_key.name)
        key_path = tmp_key.name
    else:
        if not os.path.isfile(KEY_PATH):
            return render_template_string(HTML, results=[], err="Kunci jawaban tidak ditemukan. Upload file kunci.csv atau taruh di dataset/kunci.csv")
        key_path = KEY_PATH
    try:
        key = m.load_key(key_path)
    except SystemExit as e:
        return render_template_string(HTML, results=[], err=str(e))

    # ── args ──────────────────────────────────────────────────────────────────
    mode       = request.form.get("mode", "semua")
    min_extra  = int(request.form.get("min_extra", 40))
    dominance  = float(request.form.get("dominance", 3.0))
    scan       = bool(request.form.get("scan", ""))
    args       = make_args(mode, min_extra, dominance, scan)

    positions  = m.letter_positions()
    os.makedirs(HASIL_DIR, exist_ok=True)

    # ── foto lembar ───────────────────────────────────────────────────────────
    lembar_files = request.files.getlist("lembar")
    lembar_files = [f for f in lembar_files if f and f.filename]
    if not lembar_files:
        return render_template_string(HTML, results=[], err="Pilih minimal 1 foto lembar untuk diproses.")

    results, errors = [], []
    for lf in lembar_files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(lf.filename)[-1])
        lf.save(tmp.name)
        res, err = proses_satu(tmp.name, key, ink_t, positions, args)
        os.unlink(tmp.name)
        if res:
            # pakai nama file asli sebagai label
            res["name"] = os.path.splitext(lf.filename)[0]
            results.append(res)
        else:
            errors.append(f"{lf.filename}: {err}")

    err_msg = "; ".join(errors) if errors else None
    return render_template_string(HTML, results=results, err=err_msg)


@app.route("/demo", methods=["POST"])
def demo():
    if not os.path.isfile(TEMPLATE_PATH):
        return render_template_string(HTML, results=[], err="Template kosong tidak ditemukan: " + TEMPLATE_PATH)

    mode_demo = request.form.get("mode_demo", "semua")
    demo_dir  = os.path.join(os.path.dirname(__file__), "dataset", "demo")
    out_dir   = os.path.join(HASIL_DIR, "demo")
    os.makedirs(HASIL_DIR, exist_ok=True)

    key_path, truth_path = m.buat_data_demo(TEMPLATE_PATH, demo_dir, mode_demo)
    key       = m.load_key(key_path)
    truth     = m.load_truth(truth_path)
    positions = m.letter_positions()

    tpl   = cv2.resize(cv2.imread(TEMPLATE_PATH), (m.REF_W, m.REF_H))
    ink_t = m.binarize(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY))

    args = make_args(mode_demo)
    args.out = out_dir

    files = [f for f in m.list_images(demo_dir)
             if os.path.basename(f).lower() not in ("template_kosong.jpeg", "template_kosong.jpg")]

    results, errors = [], []
    for fp in sorted(files):
        res, err = proses_satu(fp, key, ink_t, positions, args, truth)
        if res:
            results.append(res)
        else:
            errors.append(f"{os.path.basename(fp)}: {err}")

    err_msg = "; ".join(errors) if errors else None
    return render_template_string(HTML, results=results, err=err_msg)


if __name__ == "__main__":
    import threading
    import webbrowser

    def _buka_browser():
        import time
        time.sleep(1.2)          # tunggu Flask siap dulu
        webbrowser.open("http://localhost:5000")

    threading.Thread(target=_buka_browser, daemon=True).start()

    print("=" * 55)
    print("  Web UI Pemeriksaan Pilihan Ganda")
    print("  Browser akan terbuka otomatis...")
    print("  Atau buka manual: http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=5000)


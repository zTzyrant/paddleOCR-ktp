"""
Test akurasi OCR pada seluruh dataset KTP.

Validasi NIK menggunakan regex:
    ^(1[1-9]|21|[37][1-6]|5[1-3]|6[1-5]|[89][12])\d{2}\d{2}([04][1-9]|[1256][0-9]|[37][01])(0[1-9]|1[0-2])\d{2}\d{4}$

Skoring per image:
    - NIK valid secara regex        → skor 50
    - Tanggal lahir dari NIK cocok  → skor +50 (total 100)

Usage:
    python test_ocr_accuracy.py [--debug] [--limit 10]
"""

import re
import os
import sys
import json
import glob
import time
import warnings
from typing import List, Optional, Dict
from contextlib import redirect_stdout, redirect_stderr

# Set environment before any paddle imports
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["GLOG_minloglevel"] = "3"  # Suppress PaddlePaddle C++ logs
os.environ["FLAGS_log_level"] = "5"   # Suppress FLAGS logs

warnings.filterwarnings("ignore")

from paddleocr import PaddleOCR
from ocr import run_ocr, extract_nik, extract_nama, extract_tanggal_lahir, extract_optional_fields

# ──────────────────────────────────────────────
# Regex NIK KTP
# ──────────────────────────────────────────────
NIK_REGEX = re.compile(
    r"^(1[1-9]|21|[37][1-6]|5[1-3]|6[1-5]|[89][12])"  # 6 digit pertama: provinsi+kabupaten+kecamatan
    r"\d{2}"                                             # 2 digit kabupaten/kota
    r"\d{2}"                                             # 2 digit kecamatan
    r"([04][1-9]|[1256][0-9]|[37][01])"                 # 2 digit tanggal (valid: 01-31, hindari 38-39,40)
    r"(0[1-9]|1[0-2])"                                   # 2 digit bulan (01-12)
    r"\d{2}"                                             # 2 digit tahun
    r"\d{4}$"                                            # 4 digit unique + checksum
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def extract_birthdate_from_nik(nik: str) -> Optional[str]:
    """
    Ekstrak tanggal lahir dari 16 digit NIK.
    Format NIK: PP KK TT DD MM YY UUUU
      DD = tanggal (bisa 01-31 atau 40-71 untuk wanita → DD-40)
      MM = bulan (01-12)
      YY = tahun (misal 90 = 1990, 05 = 2005)
    Returns: string "DD-MM-YYYY" atau None
    """
    if len(nik) != 16:
        return None

    day_str = nik[6:8]
    month_str = nik[8:10]
    year_str = nik[10:12]

    # Decode tanggal: jika day > 40 berarti wanita (day - 40)
    day = int(day_str)
    if day > 40:
        day -= 40
    day = f"{day:02d}"

    month = int(month_str)
    month_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                   "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    month_name = month_names[month - 1] if 1 <= month <= 12 else None
    if month_name is None:
        return None

    # Decode tahun: asumsi 00-99 → 1900-1999 atau 2000-2099
    year_num = int(year_str)
    if year_num >= 0 and year_num <= 30:
        year = 2000 + year_num
    else:
        year = 1900 + year_num

    return f"{day}-{month:02d}-{year}"


def validate_nik(nik: str) -> dict:
    """
    Validasi NIK dan hitung skor.
    Returns: {"nik": str, "regex_valid": bool, "birthdate_from_nik": str|None, "score": int}
    """
    result = {
        "nik": nik or "",
        "regex_valid": False,
        "birthdate_from_nik": None,
        "score": 0,
    }

    if not nik or len(nik) != 16 or not nik.isdigit():
        return result

    # Cek regex
    if NIK_REGEX.match(nik):
        result["regex_valid"] = True
        result["score"] += 50

    # Ekstrak tanggal lahir dari NIK
    bd_nik = extract_birthdate_from_nik(nik)
    result["birthdate_from_nik"] = bd_nik

    return result


# ──────────────────────────────────────────────
# Main test
# ──────────────────────────────────────────────
def run_test(image_dir: str, debug: bool = False, limit: int = 0) -> list:
    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if limit > 0:
        images = images[:limit]

    total = len(images)
    print(f"[*] Menemukan {total} gambar KTP di: {image_dir}")
    print(f"[*] Memulai test akurasi OCR...\n")

    # Initialize OCR once (suppress init messages)
    print("[*] Inisialisasi PaddleOCR...", flush=True)
    
    # Temporarily replace print to suppress model loading spam
    _original_print = print
    def _silent_print(*args, **kwargs):
        pass
    
    import builtins
    builtins.print = _silent_print
    try:
        ocr = PaddleOCR(lang="id")
    finally:
        builtins.print = _original_print
    
    print("[*] Model siap digunakan.\n", flush=True)

    results = []
    scores_100 = 0
    scores_50 = 0
    scores_0 = 0
    failed_ocr = 0

    start_time = time.time()

    for idx, img_path in enumerate(images, 1):
        filename = os.path.basename(img_path)
        print(f"[{idx}/{total}] {filename} ... ", end="", flush=True)

        try:
            lines = run_ocr(ocr, img_path)
            data = {
                "nik": extract_nik(lines),
                "nama": extract_nama(lines),
                "tanggal_lahir": extract_tanggal_lahir(lines),
                **extract_optional_fields(lines),
            }
        except Exception as e:
            print(f" ERROR: {e}")
            failed_ocr += 1
            results.append({"file": filename, "error": str(e), "score": 0})
            scores_0 += 1
            continue

        nik = data.get("nik")
        tanggal_lahir_ocr = data.get("tanggal_lahir")

        val = validate_nik(nik or "")
        bd_nik = val["birthdate_from_nik"]

        # Bandingkan tanggal lahir dari OCR dengan tanggal lahir dari NIK
        if val["regex_valid"] and bd_nik and tanggal_lahir_ocr:
            # Normalisasi format untuk perbandingan
            if bd_nik == tanggal_lahir_ocr:
                val["score"] += 50

        # Update counter
        if val["score"] == 100:
            scores_100 += 1
        elif val["score"] == 50:
            scores_50 += 1
        else:
            scores_0 += 1

        score_label = "✓100" if val["score"] == 100 else ("~50" if val["score"] == 50 else "✗0")
        print(f" {score_label} (NIK={nik or 'N/A'}, TL={tanggal_lahir_ocr or 'N/A'})")

        results.append({
            "file": filename,
            "nik": nik,
            "tanggal_lahir_ocr": tanggal_lahir_ocr,
            "regex_valid": val["regex_valid"],
            "birthdate_from_nik": bd_nik,
            "score": val["score"],
            "nama": data.get("nama"),
            "jenis_kelamin": data.get("jenis_kelamin"),
        })

    elapsed = time.time() - start_time

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  HASIL TEST AKURASI OCR")
    print("=" * 60)
    print(f"  Total gambar    : {total}")
    print(f"  Skor 100 (NIK+TL valid) : {scores_100} ({scores_100/total*100:.1f}%)")
    print(f"  Skor 50  (NIK valid)    : {scores_50} ({scores_50/total*100:.1f}%)")
    print(f"  Skor 0   (gagal)        : {scores_0} ({scores_0/total*100:.1f}%)")
    print(f"  OCR error       : {failed_ocr}")
    print(f"  Waktu total     : {elapsed:.1f}s")
    print(f"  Rata-rata/image : {elapsed/total:.1f}s")
    print(f"  Skor rata-rata  : {sum(r['score'] for r in results)/total:.1f}/100")
    print("=" * 60)

    # ── Save report ──
    report_path = os.path.join(os.path.dirname(__file__), "ocr_accuracy_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": total,
            "scores_100": scores_100,
            "scores_50": scores_50,
            "scores_0": scores_0,
            "failed_ocr": failed_ocr,
            "elapsed_seconds": elapsed,
            "average_score": sum(r["score"] for r in results) / total,
            "details": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Report disimpan ke: {report_path}")

    # ── Print failed items ──
    failed = [r for r in results if r["score"] == 0]
    if failed:
        print(f"\n[*] Daftar file dengan skor 0 ({len(failed)} items):")
        for r in failed:
            print(f"    - {r['file']}")

    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    debug = "--debug" in sys.argv
    limit = 0

    # Parse --limit N
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    # Default directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "train")

    if not os.path.isdir(image_dir):
        print(f"[ERROR] Directory tidak ditemukan: {image_dir}")
        sys.exit(1)

    run_test(image_dir, debug=debug, limit=limit)


if __name__ == "__main__":
    main()

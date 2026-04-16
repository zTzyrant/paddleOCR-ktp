"""
KTP OCR menggunakan PaddleOCR
Ekstrak: NIK (prioritas utama), Nama, Tanggal Lahir, dan field lainnya

Install dependencies:
    pip install paddlepaddle paddleocr

Usage:
    python ktp_ocr.py <path_gambar_ktp>
    python ktp_ocr.py ktp.jpg
"""

import re
import sys
import os
import warnings
from typing import List, Optional
from contextlib import redirect_stdout, redirect_stderr

warnings.filterwarnings("ignore")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_cudnn_deterministic"] = "0"
os.environ["GLOG_minloglevel"] = "3"  # Suppress PaddlePaddle C++ logs
os.environ["FLAGS_log_level"] = "5"   # Suppress FLAGS logs

from paddleocr import PaddleOCR


# ──────────────────────────────────────────────
# Inisialisasi PaddleOCR
# ──────────────────────────────────────────────
def init_ocr() -> PaddleOCR:
    # Suppress verbose model loading messages
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            return PaddleOCR(lang="id")


# ──────────────────────────────────────────────
# Raw OCR → list of text lines
# ──────────────────────────────────────────────
def run_ocr(ocr: PaddleOCR, image_path: str) -> List[str]:
    result = ocr.predict(image_path)
    lines: List[str] = []
    for page in result:
        rec_texts = page.get("rec_texts", [])
        for text in rec_texts:
            if text and text.strip():
                lines.append(text.strip())
    return lines


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────
def clean(text: str) -> str:
    return text.strip(" :-\t")


def find_value_after_key(lines: List[str], *keywords: str) -> Optional[str]:
    """Cari nilai setelah label keyword, bisa di baris yang sama (setelah ':') atau baris berikutnya."""
    keywords_lower = [k.lower() for k in keywords]
    label_hints = [
        "nama", "nik", "lahir", "jenis", "alamat", "rt", "rw",
        "kel", "kec", "agama", "status", "pekerjaan", "kewarganegaraan"
    ]
    for i, line in enumerate(lines):
        ll = line.lower()
        if any(kw in ll for kw in keywords_lower):
            if ":" in line:
                val = clean(line.split(":", 1)[1])
                if val:
                    return val
            if i + 1 < len(lines):
                next_line = clean(lines[i + 1])
                if next_line and not any(h in next_line.lower() for h in label_hints):
                    return next_line
    return None


# ──────────────────────────────────────────────
# Extractor: NIK (prioritas 1)
# ──────────────────────────────────────────────
def extract_nik(lines: List[str]) -> Optional[str]:
    # Strategi 1: label NIK diikuti 16 digit
    for i, line in enumerate(lines):
        if re.search(r'\bnik\b', line, re.IGNORECASE):
            digits = re.sub(r'\D', '', line)
            if len(digits) == 16:
                return digits
            if i + 1 < len(lines):
                digits = re.sub(r'\D', '', lines[i + 1])
                if len(digits) == 16:
                    return digits

    # Strategi 2: cari 16-digit murni di seluruh teks
    full_text = " ".join(lines)
    matches = re.findall(r'\b\d{16}\b', full_text)
    if matches:
        return matches[0]

    # Strategi 3: gabungkan digit yang berdekatan
    all_digits = "".join(re.findall(r'\d+', full_text))
    if len(all_digits) >= 16:
        return all_digits[:16]

    return None


# ──────────────────────────────────────────────
# Extractor: Nama (prioritas 2)
# ──────────────────────────────────────────────
def extract_nama(lines: List[str]) -> Optional[str]:
    val = find_value_after_key(lines, "nama")
    if val:
        val = re.sub(r'[^A-Za-z\s\']', '', val).strip()
        return val.upper() if val else None
    return None


# ──────────────────────────────────────────────
# Extractor: Tanggal Lahir (prioritas 3)
# ──────────────────────────────────────────────
def extract_tanggal_lahir(lines: List[str]) -> Optional[str]:
    date_pattern = re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b')

    for i, line in enumerate(lines):
        if re.search(r'lahir', line, re.IGNORECASE):
            m = date_pattern.search(line)
            if m:
                return m.group(1)
            for j in range(i + 1, min(i + 3, len(lines))):
                m = date_pattern.search(lines[j])
                if m:
                    return m.group(1)

    # Fallback: pola tanggal di mana saja
    for line in lines:
        m = date_pattern.search(line)
        if m:
            return m.group(1)

    return None


# ──────────────────────────────────────────────
# Extractor: Field opsional
# ──────────────────────────────────────────────
def extract_optional_fields(lines: List[str]) -> dict:
    result = {}

    # Jenis Kelamin
    for line in lines:
        ll = line.lower()
        if "perempuan" in ll:
            result["jenis_kelamin"] = "PEREMPUAN"
            break
        elif "laki" in ll:
            result["jenis_kelamin"] = "LAKI-LAKI"
            break

    # Tempat Lahir (kota sebelum tanggal)
    for line in lines:
        if re.search(r'lahir', line, re.IGNORECASE) and ":" in line:
            val = clean(line.split(":", 1)[1])
            # Pisahkan kota dan tanggal jika ada koma
            parts = val.split(",")
            if len(parts) >= 2:
                result["tempat_lahir"] = parts[0].strip()
            elif val and not re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', val):
                result["tempat_lahir"] = val
            break

    # Alamat
    alamat = find_value_after_key(lines, "alamat")
    if alamat:
        result["alamat"] = alamat

    # RT/RW
    rtrw = find_value_after_key(lines, "rt/rw", "rt")
    if rtrw:
        result["rt_rw"] = rtrw

    # Kelurahan/Desa
    kel = find_value_after_key(lines, "kel/desa", "kelurahan", "desa")
    if kel:
        result["kelurahan"] = kel

    # Kecamatan
    kec = find_value_after_key(lines, "kecamatan")
    if kec:
        result["kecamatan"] = kec

    # Agama
    for line in lines:
        for ag in ["islam", "kristen", "katolik", "hindu", "buddha", "konghucu"]:
            if ag in line.lower():
                result["agama"] = ag.upper()
                break
        if "agama" in result:
            break

    # Status Perkawinan
    for line in lines:
        ll = line.lower()
        if "belum kawin" in ll:
            result["status_perkawinan"] = "BELUM KAWIN"
            break
        elif "cerai" in ll:
            result["status_perkawinan"] = "CERAI"
            break
        elif "kawin" in ll:
            result["status_perkawinan"] = "KAWIN"
            break

    # Pekerjaan
    pekerjaan = find_value_after_key(lines, "pekerjaan")
    if pekerjaan:
        result["pekerjaan"] = pekerjaan

    # Kewarganegaraan
    for line in lines:
        if "wni" in line.upper():
            result["kewarganegaraan"] = "WNI"
            break
        elif "wna" in line.upper():
            result["kewarganegaraan"] = "WNA"
            break

    # Berlaku Hingga
    for line in lines:
        if re.search(r'seumur|hidup', line, re.IGNORECASE):
            result["berlaku_hingga"] = "SEUMUR HIDUP"
            break
    if "berlaku_hingga" not in result:
        berlaku = find_value_after_key(lines, "berlaku")
        if berlaku:
            result["berlaku_hingga"] = berlaku

    return result


# ──────────────────────────────────────────────
# Main extractor
# ──────────────────────────────────────────────
def extract_ktp(image_path: str, debug: bool = False) -> dict:
    """
    Ekstrak data KTP dari gambar.

    Args:
        image_path: Path ke file gambar KTP (.jpg / .png)
        debug: Tampilkan semua baris teks OCR mentah

    Returns:
        dict dengan keys: nik, nama, tanggal_lahir, dan field opsional lainnya
    """
    print(f"[*] Memproses: {image_path}\n")

    ocr = init_ocr()
    lines = run_ocr(ocr, image_path)

    if debug:
        print(f"[DEBUG] {len(lines)} baris teks terdeteksi:")
        for i, line in enumerate(lines):
            print(f"  {i:02d}: {line}")
        print()

    result = {
        "nik":           extract_nik(lines),             # PRIORITAS 1
        "nama":          extract_nama(lines),             # PRIORITAS 2
        "tanggal_lahir": extract_tanggal_lahir(lines),   # PRIORITAS 3
        **extract_optional_fields(lines),                 # Opsional
    }

    return result


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python ktp_ocr.py <path_gambar_ktp> [--debug]")
        sys.exit(1)

    image_path = sys.argv[1]
    debug = "--debug" in sys.argv

    if not os.path.exists(image_path):
        print(f"[ERROR] File tidak ditemukan: {image_path}")
        sys.exit(1)

    data = extract_ktp(image_path, debug=debug)

    print("=" * 50)
    print("  HASIL EKSTRAKSI KTP")
    print("=" * 50)

    priority = ["nik", "nama", "tanggal_lahir"]
    for field in priority:
        val = data.get(field)
        print(f"  {field.upper():<20}: {val or '(tidak terdeteksi)'}")

    optional_keys = [k for k in data if k not in priority]
    if optional_keys:
        print()
        print("  Field Tambahan:")
        for key in optional_keys:
            print(f"  {key.upper():<20}: {data[key]}")

    print("=" * 50)

    return data


if __name__ == "__main__":
    main()
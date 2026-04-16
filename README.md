# KTP OCR - Indonesian ID Card Text Extraction

Ekstraksi data dari KTP (Kartu Tanda Penduduk) menggunakan PaddleOCR dengan validasi NIK berbasis regex.

## Fitur

- **OCR PaddleOCR v5** - Deteksi dan pengenalan teks KTP otomatis
- **Ekstraksi field utama** - NIK, Nama, Tanggal Lahir, Jenis Kelamin, Agama, Status Perkawinan, Pekerjaan, dll
- **Validasi NIK** - Regex 16 digit KTP Indonesia (provinsi, kabupaten, kecamatan, tanggal, bulan, tahun)
- **Skoring akurasi** - Skor 50 (NIK valid) atau 100 (NIK + tanggal lahir cocok)
- **Batch testing** - Test akurasi pada seluruh dataset

## Install

```bash
# Requires Python 3.9+
pip install paddlepaddle==3.2.0 paddleocr
```

> **Note:** Gunakan `paddlepaddle==3.2.0`. Versi 3.3.1 memiliki bug oneDNN di Windows.

## Usage

### OCR Single Image

```bash
python ocr.py <path_gambar_ktp>

# Contoh
python ocr.py 1_jpg.rf.4rqPz7VxqbQvst71eNLT.jpg

# Dengan debug info
python ocr.py ktp.jpg --debug
```

**Output:**
```
==================================================
  HASIL EKSTRAKSI KTP
==================================================
  NIK                 : 3215012205890002
  NAMA                : ENTIS SUTRISNA
  TANGGAL_LAHIR       : 04-03-2019

  Field Tambahan:
  JENIS_KELAMIN       : LAKI-LAKI
  AGAMA               : ISLAM
  STATUS_PERKAWINAN   : KAWIN
  PEKERJAAN           : KARYAWANSWASTA
  BERLAKU_HINGGA      : SEUMUR HIDUP
==================================================
```

### Batch Accuracy Test

```bash
# Test semua gambar di folder train/
python test_ocr_accuracy.py

# Test sebagian (lebih cepat)
python test_ocr_accuracy.py --limit 10

# Dengan debug info
python test_ocr_accuracy.py --debug
```

**Skoring:**
| Skor | Kriteria |
|------|----------|
| 100  | NIK valid regex + tanggal lahir dari NIK cocok dengan OCR |
| 50   | NIK valid regex |
| 0    | Gagal ekstrak NIK atau NIK tidak valid |

Hasil report disimpan ke `ocr_accuracy_report.json`.

## Struktur NIK KTP

Format 16 digit: `PP KK TT DD MM YY UUUU`

| Digit | Keterangan |
|-------|------------|
| PP | Kode Provinsi (11-94) |
| KK | Kode Kabupaten/Kota (01-99) |
| TT | Kode Kecamatan (01-99) |
| DD | Tanggal lahir (01-31, wanita = DD+40) |
| MM | Bulan lahir (01-12) |
| YY | Tahun lahir (2 digit) |
| UUUU | Unique code + checksum |

**Regex validasi:**
```
^(1[1-9]|21|[37][1-6]|5[1-3]|6[1-5]|[89][12])\d{2}\d{2}([04][1-9]|[1256][0-9]|[37][01])(0[1-9]|1[0-2])\d{2}\d{4}$
```

## Struktur Proyek

```
ktp-only.coco-segmentation/
├── ocr.py                    # Main OCR & KTP extractor
├── test_ocr_accuracy.py      # Batch accuracy test
├── regex.txt                 # NIK validation regex
├── train/                    # Dataset gambar KTP
│   ├── _annotations.coco.json
│   └── *.jpg
├── .gitignore
└── README.md
```

## API Usage

```python
from ocr import extract_ktp, validate_nik, extract_birthdate_from_nik

# Ekstrak data KTP
data = extract_ktp("ktp.jpg")
print(data["nik"])           # "3215012205890002"
print(data["nama"])          # "ENTIS SUTRISNA"
print(data["tanggal_lahir"]) # "04-03-2019"

# Validasi NIK
result = validate_nik("3215012205890002")
print(result["regex_valid"])       # True
print(result["birthdate_from_nik"]) # "22-05-1989"
print(result["score"])             # 50 (valid regex)
```

## Dependencies

| Package | Version | Keterangan |
|---------|---------|------------|
| paddlepaddle | 3.2.0 | Deep learning engine |
| paddleocr | 3.4.1 | OCR toolkit |
| paddlex | 3.4.3 | Model management |

Model akan auto-download ke `C:\Users\<user>\.paddlex\official_models\` saat pertama kali run.

## License

See [README.roboflow.txt](README.roboflow.txt) for dataset license.

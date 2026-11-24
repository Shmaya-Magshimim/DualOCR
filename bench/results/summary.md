| Benchmark | System | Main metric | Word acc (exact) | Letters acc | Char acc | BoW F1 | Pages |
|---|---|---|---|---|---|---|---|
| funsd | Tesseract (raw image, psm 3) | word F1 49.7 (P 63.4 / R 40.9) |  |  |  | 49.7 | 50 |
| funsd | Tesseract (DualOCR-prepped page, psm 3) | word F1 49.7 (P 64.1 / R 40.6) |  |  |  | 49.7 | 50 |
| funsd | Surya 2.5 (llama.cpp) | word F1 80.0 (P 83.0 / R 77.1) |  |  |  | 80.0 | 50 |
| funsd | DualOCR fast | word F1 79.3 (P 82.2 / R 76.6) |  |  |  | 79.3 | 50 |
| funsd | DualOCR balanced | word F1 79.0 (P 82.2 / R 76.1) |  |  |  | 79.0 | 50 |
| he_wikisource | Tesseract (raw image, psm 3) | span word acc 43.6; ref-word recall 47.3; char acc (no spaces) 50.5 | 22.8 | 31.8 | 54.6 | 45.9 | 57 |
| he_wikisource | Tesseract (DualOCR-prepped page, psm 3) | span word acc 43.8; ref-word recall 47.2; char acc (no spaces) 50.9 | 23.4 | 32.3 | 55.0 | 45.9 | 57 |
| he_wikisource | Surya 2.5 (llama.cpp) | span word acc 60.6; ref-word recall 66.9; char acc (no spaces) 63.1 | 39.7 | 46.9 | 64.2 | 66.1 | 57 |
| he_wikisource | DualOCR fast | span word acc 62.0; ref-word recall 69.6; char acc (no spaces) 62.8 | 41.1 | 48.4 | 64.1 | 68.4 | 57 |
| he_wikisource | DualOCR balanced | span word acc 60.7; ref-word recall 68.1; char acc (no spaces) 61.4 | 40.2 | 47.2 | 62.8 | 67.2 | 57 |
| olmocr_old_scans | Tesseract (raw image, psm 3) | tests passed 15.9% (printed 22.8%, handwritten/little text 9.3%) |  |  |  |  | 50 |
| olmocr_old_scans | Tesseract (DualOCR-prepped page, psm 3) | tests passed 17.1% (printed 25.2%, handwritten/little text 9.3%) |  |  |  |  | 50 |
| olmocr_old_scans | Surya 2.5 (llama.cpp) | tests passed 31.7% (printed 45.5%, handwritten/little text 18.6%) |  |  |  |  | 50 |
| olmocr_old_scans | DualOCR fast | tests passed 32.5% (printed 46.3%, handwritten/little text 19.4%) |  |  |  |  | 50 |
| olmocr_old_scans | DualOCR balanced | tests passed 32.1% (printed 44.7%, handwritten/little text 20.2%) |  |  |  |  | 50 |
| recbench_en | Tesseract (raw image, psm 3) | Surya line score 89.7; char acc (no spaces) 82.5 | 79.5 | 81.9 | 83.1 | 90.2 | 40 |
| recbench_en | Tesseract (DualOCR-prepped page, psm 3) | Surya line score 90.6; char acc (no spaces) 83.2 | 80.2 | 82.6 | 83.9 | 90.7 | 40 |
| recbench_en | Surya 2.5 (llama.cpp) | Surya line score 97.5; char acc (no spaces) 94.1 | 90.0 | 90.5 | 94.0 | 95.4 | 40 |
| recbench_en | DualOCR fast | Surya line score 97.0; char acc (no spaces) 93.1 | 89.4 | 90.4 | 93.1 | 95.4 | 40 |
| recbench_en | DualOCR balanced | Surya line score 96.9; char acc (no spaces) 93.1 | 89.2 | 90.6 | 93.1 | 95.5 | 40 |
| recbench_he | Tesseract (raw image, psm 3) | Surya line score 98.6; char acc (no spaces) 98.5 | 96.6 | 97.2 | 98.5 | 97.9 | 10 |
| recbench_he | Tesseract (DualOCR-prepped page, psm 3) | Surya line score 98.6; char acc (no spaces) 98.5 | 96.6 | 97.2 | 98.5 | 97.9 | 10 |
| recbench_he | Surya 2.5 (llama.cpp) | Surya line score 99.1; char acc (no spaces) 98.8 | 94.9 | 95.4 | 98.8 | 96.0 | 10 |
| recbench_he | DualOCR fast | Surya line score 99.1; char acc (no spaces) 99.0 | 96.2 | 96.8 | 98.9 | 97.5 | 10 |
| recbench_he | DualOCR balanced | Surya line score 99.0; char acc (no spaces) 98.8 | 95.9 | 97.0 | 98.7 | 97.7 | 10 |

he_wikisource by group (span word acc / order-free recall of the reference words / char acc no-spaces):

| System | square_script (34 p.) | rashi_script (13 p.) | with_niqqud (10 p.) |
|---|---|---|---|
| Tesseract (raw image, psm 3) | 66.0 / 70.1 / 63.8 | 13.9 / 16.0 / 34.3 | 26.6 / 34.3 / 35.3 |
| Tesseract (DualOCR-prepped page, psm 3) | 66.4 / 70.1 / 64.7 | 13.8 / 16.0 / 34.3 | 26.6 / 33.3 / 34.4 |
| Surya 2.5 (llama.cpp) | 73.8 / 80.6 / 67.3 | 40.7 / 45.9 / 54.3 | 60.6 / 68.3 / 74.4 |
| DualOCR fast | 76.2 / 83.7 / 67.3 | 41.2 / 48.9 / 53.8 | 60.1 / 67.6 / 72.5 |
| DualOCR balanced | 75.6 / 83.0 / 67.1 | 40.1 / 47.7 / 52.3 | 53.3 / 59.3 / 63.9 |

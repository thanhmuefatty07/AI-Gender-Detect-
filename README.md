#  AI-Gender-Fusion (Minimal Colab Project)

**M?c tiêu:** Hu?n luy?n fusion model (vision + audio) tr?c ti?p trên **Google Colab**.
**Tri?t lý:** Notebook-first, t?i gi?n file/folder. Không Docker, không API, không CI/CD.

---

##  C?u trúc t?i gi?n
```
AI-Gender-Fusion/
 notebooks/
    Gender_Fusion_Training.ipynb   # Notebook chính (Colab)
 requirements.txt                   # Thu vi?n t?i thi?u
 auto_bot.py                        # Bot t?i d? li?u Youtube (ch?y local)
 README.md                          # Hu?ng d?n
```
> Toàn b? logic train/pipeline nên d?t trong `notebooks/Gender_Fusion_Training.ipynb`.

---

##  Quy trình làm vi?c (Notebook-first)
1) Trên local: dùng `auto_bot.py` d? crawl YouTube theo t? khóa, luu vào `Gender_Raw_Data/` (nên d?t trong Google Drive for Desktop d? auto-sync).
2) Upload/Sync `Gender_Raw_Data` lên Google Drive.
3) Trên Colab: m? `notebooks/Gender_Fusion_Training.ipynb`, mount Drive, tr? t?i thu m?c data, x? lý + train.

---

##  Cài d?t (Colab)
```bash
!pip install -r requirements.txt
```
**requirements.txt (t?i gi?n):**
```
numpy<2.0
mediapipe
speechbrain
torchaudio
torchmetrics
ffmpeg-python
yt-dlp
```

---

##  Bot crawl d? li?u (ch?y local)
- File: `auto_bot.py`
- Ch?nh `SEARCH_KEYWORDS`, `LIMIT_PER_KEYWORD`, `SAVE_DIR`.
- Ch?y: `python auto_bot.py` (c?n `yt-dlp`, khuy?n ngh? cài FFmpeg).
- M?o: Ð?t `SAVE_DIR` bên trong thu m?c Google Drive d? t? d?ng b? lên cloud.

---

##  FAQ
- **T?i sao không Docker/API/CI/CD?** D? án sinh viên, ch?y trên Colab  uu tiên don gi?n, d? debug.
- **Luu model ? dâu?** Google Drive ho?c t?i xu?ng tr?c ti?p t? notebook.
- **N?u c?n thêm module .py?** Ð?t cùng thu m?c v?i notebook d? d? import.

---

##  Ghi chú
- Repo dã du?c d?n s?ch các thành ph?n MLOps cu.
- N?u c?n b? sung, hãy thêm tr?c ti?p vào notebook ho?c m?t file `.py` duy nh?t.
- Commit notebook sau khi hoàn thi?n thí nghi?m d? luu k?t qu?.

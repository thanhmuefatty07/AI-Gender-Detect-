# 🎯 AI-Gender-Fusion (Minimal Colab Project)

**Mục tiêu:** Huấn luyện fusion model (vision + audio) trực tiếp trên **Google Colab**.  
**Triết lý:** Notebook-first, tối giản file/folder. Không Docker, không API, không CI/CD.

---

## 📂 Cấu trúc tối giản
```
AI-Gender-Fusion/
├── notebooks/
│   └── Gender_Fusion_Training.ipynb   # Notebook chính (Colab)
├── requirements.txt                   # Thư viện tối thiểu
└── README.md                          # Hướng dẫn
```

> Lưu ý: Toàn bộ logic (tải data → xử lý → train → eval) đặt trong **Gender_Fusion_Training.ipynb**.

---

## 🛠️ Cài đặt (trên Colab)
```bash
# Trong Colab cell đầu tiên
!pip install -r requirements.txt
```

**requirements.txt (đã rút gọn):**
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

## 🚀 Quy trình làm việc (Notebook-first)
1) Mở `notebooks/Gender_Fusion_Training.ipynb` trên Google Colab.  
2) Chạy cell cài đặt dependencies.  
3) Thực hiện pipeline trong notebook:
   - Tải dữ liệu (YouTube/Wikimedia, tùy bạn)  
   - Tiền xử lý (ảnh + audio)  
   - Huấn luyện fusion model (vision + audio)  
   - Đánh giá & lưu checkpoint (tùy chọn: drive/weights)  
4) Xuất kết quả/metric trực tiếp từ notebook.

---

## ❓ FAQ
- **Tại sao không Docker/API/CI/CD?**  
  Dự án sinh viên, chạy trên Colab → ưu tiên đơn giản, dễ debug.

- **Tôi nên đặt code ở đâu?**  
  Gọn trong notebook chính; nếu cần thêm file `.py`, để cùng thư mục với notebook.

- **Lưu model ở đâu?**  
  Gợi ý: Google Drive hoặc tải xuống trực tiếp từ notebook.

---

## 📌 Ghi chú
- Repo đã được dọn sạch khỏi các thành phần MLOps cũ (Docker, API, monitoring, collectors...).  
- Nếu cần bổ sung, hãy thêm trực tiếp vào notebook hoặc một file `.py` duy nhất.  
- Hãy commit notebook sau khi hoàn thiện thí nghiệm để lưu lại kết quả.
#!/bin/bash

# ============================================
# Script: setup_structure.sh
# Tạo toàn bộ cấu trúc project cho Gender-Age Classifier
# ============================================

echo "🏗️  Creating Gender-Age Classifier Project Structure..."
echo "======================================================="

# Kiểm tra xem đã ở trong thư mục project chưa
if [ ! -d "gender_age_classifier" ]; then
    echo "❌ Không tìm thấy thư mục gender_age_classifier"
    echo "Chạy: mkdir gender_age_classifier && cd gender_age_classifier"
    exit 1
fi

cd gender_age_classifier

# ============================================
# 1. CONFIG - Cấu hình
# ============================================
echo "📁 Creating config directories..."
mkdir -p config/{environments,models,deployment}

# ============================================
# 2. DATA COLLECTION - Thu thập dữ liệu
# ============================================
echo "📥 Creating data collection directories..."
mkdir -p data_collection/{youtube,tiktok,instagram,base}
mkdir -p data_collection/youtube/{downloader,processor,metadata}
mkdir -p data_collection/tiktok/{downloader,processor,metadata}
mkdir -p data_collection/instagram/{downloader,processor,metadata}

# ============================================
# 3. DATASETS - Lưu trữ dữ liệu
# ============================================
echo "💾 Creating datasets directories..."
mkdir -p datasets/collected/{youtube,tiktok,instagram}/{raw_videos,processed,annotations}
mkdir -p datasets/academic/{utkface,fairface,imdb_wiki,other}
mkdir -p datasets/prepared/{train,val,test}/{images,audio,metadata}
mkdir -p datasets/augmented

# ============================================
# 4. MODELS - AI/ML Models
# ============================================
echo "🤖 Creating models directories..."
mkdir -p models/vision/{architectures,checkpoints,exports}
mkdir -p models/audio/{architectures,checkpoints,exports}
mkdir -p models/fusion/{architectures,checkpoints,exports}
mkdir -p models/pretrained

# ============================================
# 5. PREPROCESSING - Tiền xử lý
# ============================================
echo "🔄 Creating preprocessing directories..."
mkdir -p preprocessing/{face_detection,audio_extraction,quality_check,augmentation}

# ============================================
# 6. TRAINING - Huấn luyện
# ============================================
echo "🎯 Creating training directories..."
mkdir -p training/{vision,audio,fusion}/{scripts,configs,logs}

# ============================================
# 7. EVALUATION - Đánh giá
# ============================================
echo "📊 Creating evaluation directories..."
mkdir -p evaluation/{metrics,reports,visualizations}

# ============================================
# 8. INFERENCE - Suy luận
# ============================================
echo "🚀 Creating inference directories..."
mkdir -p inference/{api,batch,realtime}

# ============================================
# 9. APP - Ứng dụng web
# ============================================
echo "🌐 Creating app directories..."
mkdir -p app/{frontend,backend,static,templates}
mkdir -p app/frontend/{components,pages,styles,assets}
mkdir -p app/backend/{routes,services,models,middleware}
mkdir -p app/static/{css,js,images,fonts}

# ============================================
# 10. MONITORING - Giám sát
# ============================================
echo "📈 Creating monitoring directories..."
mkdir -p monitoring/{dashboards,alerts,logs}

# ============================================
# 11. DEPLOYMENT - Triển khai
# ============================================
echo "🐳 Creating deployment directories..."
mkdir -p deployment/{docker,kubernetes,terraform,scripts}

# ============================================
# 12. SCRIPTS - Tiện ích
# ============================================
echo "🔧 Creating scripts directories..."
mkdir -p scripts/{data,training,evaluation,deployment,utils}

# ============================================
# 13. TESTS - Kiểm thử
# ============================================
echo "🧪 Creating tests directories..."
mkdir -p tests/{unit,integration,e2e,fixtures}

# ============================================
# 14. NOTEBOOKS - Jupyter notebooks
# ============================================
echo "📓 Creating notebooks directories..."
mkdir -p notebooks/{exploration,experiments,analysis,demos}

# ============================================
# 15. UTILS - Tiện ích chung
# ============================================
echo "🛠️ Creating utils directories..."
mkdir -p utils/{logging,visualization,metrics,io}

# ============================================
# 16. DOCS - Tài liệu
# ============================================
echo "📚 Creating docs directories..."
mkdir -p docs/{api,architecture,guides,tutorials}

# ============================================
# 17. LOGS - Nhật ký
# ============================================
echo "📝 Creating logs directories..."
mkdir -p logs/{training,inference,api,errors}

# ============================================
# 18. CACHE - Bộ nhớ đệm
# ============================================
echo "💾 Creating cache directories..."
mkdir -p cache/{models,data,requests}

# ============================================
# 19. RESULTS - Kết quả
# ============================================
echo "📊 Creating results directories..."
mkdir -p results/{experiments,benchmarks,reports}

# ============================================
# 20. Tạo __init__.py files cho Python packages
# ============================================
echo "🐍 Creating __init__.py files..."
find . -type d -name "*" | while read dir; do
    if [[ "$dir" != "." && "$dir" != "./logs" && "$dir" != "./cache" && "$dir" != "./results" && "$dir" != "./datasets" ]]; then
        touch "$dir/__init__.py" 2>/dev/null || true
    fi
done

echo ""
echo "======================================================="
echo "✅ Folder structure created successfully!"
echo ""

# Đếm số lượng thư mục
TOTAL_DIRS=$(find . -type d | wc -l)
echo "📊 Total directories created: $TOTAL_DIRS"

# Hiển thị cấu trúc chính
echo ""
echo "🏗️  Main structure:"
echo "├── config/           # ⚙️  Configuration files"
echo "├── data_collection/  # 📥 Data collection modules"
echo "├── datasets/         # 💾 Data storage"
echo "├── models/           # 🤖 AI/ML models"
echo "├── preprocessing/    # 🔄 Data preprocessing"
echo "├── training/         # 🎯 Model training"
echo "├── evaluation/       # 📊 Model evaluation"
echo "├── inference/        # 🚀 Model inference"
echo "├── app/              # 🌐 Web application"
echo "├── monitoring/       # 📈 System monitoring"
echo "├── deployment/       # 🐳 Deployment configs"
echo "├── scripts/          # 🔧 Utility scripts"
echo "├── tests/            # 🧪 Test suites"
echo "├── notebooks/        # 📓 Jupyter notebooks"
echo "├── utils/            # 🛠️ Utility functions"
echo "├── docs/             # 📚 Documentation"
echo "├── logs/             # 📝 System logs"
echo "├── cache/            # 💾 Cache storage"
echo "└── results/          # 📊 Experiment results"

echo ""
echo "🎯 Next steps:"
echo "1. Run: python verify_structure.py"
echo "2. Copy config files from templates"
echo "3. Install dependencies: pip install -r requirements.txt"
echo "4. Start coding! 🚀"

echo ""
echo "======================================================="

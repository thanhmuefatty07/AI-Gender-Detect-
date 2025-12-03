# 🏗️ CẤU TRÚC TỔNG THỂ DỰ ÁN GENDER AGE CLASSIFICATION

## 📁 Cấu Trúc Thư Mục Chi Tiết

```
gender_age_classifier/
├── 📁 config/                          # Cấu hình hệ thống
│   ├── 📄 collector_config.yaml        # Cấu hình thu thập dữ liệu
│   ├── 📄 env.template                 # Template biến môi trường
│   └── 📄 logging_config.yaml          # Cấu hình logging
├── 📁 data_collection/                 # Module thu thập dữ liệu
│   ├── 📄 __init__.py                  # Khởi tạo package
│   ├── 📄 base_collector.py            # Lớp cơ sở cho tất cả collectors
│   ├── 📄 youtube_collector.py         # Collector YouTube
│   ├── 📄 tiktok_collector.py          # Collector TikTok
│   ├── 📄 instagram_collector.py       # Collector Instagram
│   ├── 📄 bilibili_collector.py        # Collector Bilibili (tương lai)
│   └── 📄 video_processor.py           # Xử lý video nâng cao
├── 📁 models/                          # Mô hình AI/ML
│   ├── 📄 __init__.py
│   ├── 📁 face_detection/              # Mô hình phát hiện khuôn mặt
│   │   ├── 📄 mediapipe_model.py
│   │   ├── 📄 opencv_model.py
│   │   └── 📄 dlib_model.py
│   ├── 📁 gender_classification/       # Mô hình phân loại giới tính
│   │   ├── 📄 cnn_model.py
│   │   ├── 📄 transformer_model.py
│   │   └── 📄 ensemble_model.py
│   └── 📁 preprocessing/                # Tiền xử lý dữ liệu
│       ├── 📄 face_aligner.py
│       ├── 📄 image_normalizer.py
│       └── 📄 feature_extractor.py
├── 📁 app/                             # Giao diện ứng dụng
│   ├── 📄 __init__.py
│   ├── 📄 monitoring_dashboard.py      # Dashboard giám sát
│   ├── 📄 data_explorer.py             # Công cụ khám phá dữ liệu
│   ├── 📄 quality_inspector.py         # Công cụ kiểm tra chất lượng
│   └── 📁 api/                         # REST API
│       ├── 📄 __init__.py
│       ├── 📄 routes.py                # Định tuyến API
│       ├── 📄 models.py                # Mô hình API
│       └── 📄 middleware.py            # Middleware xử lý
├── 📁 scripts/                         # Scripts tiện ích
│   ├── 📄 __init__.py
│   ├── 📄 academic_datasets_merger.py  # Merge datasets học thuật
│   ├── 📄 run_monitoring.py            # Chạy dashboard
│   ├── 📄 test_system.py               # Test hệ thống
│   ├── 📄 setup_environment.py         # Setup môi trường
│   ├── 📄 backup_data.py               # Sao lưu dữ liệu
│   └── 📄 export_dataset.py            # Export dataset
├── 📁 datasets/                        # Quản lý dữ liệu
│   ├── 📁 collected/                   # Dữ liệu đã thu thập
│   │   ├── 📁 youtube/                 # Dữ liệu YouTube
│   │   │   ├── 📁 raw_videos/          # Video thô
│   │   │   └── 📁 processed/           # Video đã xử lý
│   │   ├── 📁 tiktok/                  # Dữ liệu TikTok
│   │   │   ├── 📁 raw_videos/
│   │   │   └── 📁 processed/
│   │   ├── 📁 instagram/               # Dữ liệu Instagram
│   │   │   ├── 📁 raw_videos/
│   │   │   └── 📁 processed/
│   │   └── 📁 metadata/                # Metadata tổng hợp
│   ├── 📁 academic/                    # Datasets học thuật
│   │   ├── 📁 raw/                     # Dữ liệu thô
│   │   ├── 📁 processed/               # Dữ liệu đã xử lý
│   │   └── 📁 merged/                  # Dữ liệu đã merge
│   └── 📁 temp/                        # Dữ liệu tạm thời
├── 📁 logs/                            # Logs hệ thống
│   ├── 📄 collector_*.log              # Logs thu thập
│   ├── 📄 processor_*.log              # Logs xử lý
│   ├── 📄 system_*.log                 # Logs hệ thống
│   └── 📄 error_*.log                  # Logs lỗi
├── 📁 notebooks/                       # Jupyter notebooks
│   ├── 📄 data_analysis.ipynb          # Phân tích dữ liệu
│   ├── 📄 model_training.ipynb         # Train model
│   ├── 📄 quality_assessment.ipynb     # Đánh giá chất lượng
│   ├── 📄 visualization.ipynb          # Trực quan hóa
│   └── 📄 experiments.ipynb            # Thí nghiệm
├── 📁 tests/                           # Unit tests
│   ├── 📄 __init__.py
│   ├── 📄 test_collectors.py           # Test collectors
│   ├── 📄 test_processors.py           # Test processors
│   ├── 📄 test_models.py               # Test models
│   ├── 📄 test_utils.py                # Test utilities
│   └── 📁 fixtures/                    # Test data
├── 📁 utils/                           # Utilities
│   ├── 📄 __init__.py
│   ├── 📄 data_validator.py            # Validate dữ liệu
│   ├── 📄 file_manager.py              # Quản lý file
│   ├── 📄 metrics_calculator.py        # Tính metrics
│   ├── 📄 config_manager.py            # Quản lý config
│   └── 📄 api_client.py                # Client API
├── 📁 docs/                            # Documentation
│   ├── 📄 README.md                    # Hướng dẫn chính
│   ├── 📄 API_REFERENCE.md             # Tài liệu API
│   ├── 📄 DATA_FORMAT.md               # Định dạng dữ liệu
│   ├── 📄 DEPLOYMENT.md                # Hướng dẫn deployment
│   ├── 📄 TROUBLESHOOTING.md           # Xử lý sự cố
│   └── 📁 images/                      # Hình ảnh tài liệu
├── 📄 requirements.txt                 # Dependencies chính
├── 📄 requirements-dev.txt             # Dependencies development
├── 📄 setup.py                         # Setup script
├── 📄 .gitignore                       # Git ignore rules
├── 📄 .env.example                     # Environment variables
├── 📄 docker-compose.yml               # Docker compose
├── 📄 Dockerfile                       # Docker image
└── 📄 PROJECT_STRUCTURE.md             # Cấu trúc dự án (file này)
```

---

## 🔧 Cấu Trúc Code Theo Modules

### **1. Data Collection Module**
```python
data_collection/
├── base_collector.py          # Abstract base class
├── collectors/                # Concrete collectors
│   ├── youtube_collector.py
│   ├── tiktok_collector.py
│   ├── instagram_collector.py
│   └── bilibili_collector.py
├── processors/                # Data processors
│   ├── video_processor.py
│   ├── audio_processor.py
│   └── image_processor.py
└── validators/                # Data validators
    ├── quality_validator.py
    └── content_validator.py
```

### **2. Models Module**
```python
models/
├── base/                      # Base classes
│   ├── base_model.py
│   └── base_processor.py
├── face_detection/            # Face detection models
├── gender_classification/     # Classification models
├── preprocessing/             # Preprocessing utilities
└── evaluation/                # Model evaluation
```

### **3. API Module**
```python
app/api/
├── routes/                    # API routes
│   ├── collection.py          # Collection endpoints
│   ├── processing.py          # Processing endpoints
│   ├── models.py              # Model endpoints
│   └── monitoring.py          # Monitoring endpoints
├── models/                    # Pydantic models
├── middleware/                # Custom middleware
└── dependencies/              # Dependencies
```

---

## 📊 Cấu Trúc Database/Data Flow

### **Data Pipeline Architecture**
```
Raw Data Sources
    ↓
Data Collectors (YouTube, TikTok, Instagram)
    ↓
Raw Data Storage (datasets/collected/raw_videos/)
    ↓
Video Processor (Face Extraction + Audio Features)
    ↓
Processed Data (datasets/collected/processed/)
    ↓
Quality Filter & Validation
    ↓
Clean Dataset (datasets/final/)
    ↓
Model Training Pipeline
    ↓
Trained Models (models/checkpoints/)
    ↓
Inference API (app/api/)
    ↓
Monitoring Dashboard (app/monitoring/)
```

### **Metadata Structure**
```json
{
  "item_id": "unique_identifier",
  "source": "youtube|tiktok|instagram",
  "url": "original_url",
  "title": "content_title",
  "description": "content_description",
  "duration": 300,
  "collected_at": "2024-01-01T00:00:00Z",
  "quality_score": 0.85,
  "inferred_gender": "male|female|null",
  "inferred_age": 25,
  "faces_extracted": 15,
  "audio_features": {...},
  "processing_metadata": {...}
}
```

---

## 🚀 Cấu Trúc Deployment

### **Docker Container Structure**
```dockerfile
# Multi-stage build
FROM python:3.9-slim as base
# Base dependencies

FROM base as collector
# Collection-specific setup

FROM base as processor
# Processing-specific setup

FROM base as api
# API server setup

FROM base as dashboard
# Dashboard setup
```

### **Kubernetes Structure**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gender-classifier
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: collector
        image: gender-classifier:collector
      - name: processor
        image: gender-classifier:processor
      - name: api
        image: gender-classifier:api
      - name: dashboard
        image: gender-classifier:dashboard
```

---

## 🔄 Cấu Trúc Workflow CI/CD

### **GitHub Actions Workflow**
```yaml
.github/workflows/
├── ci.yml                      # Continuous Integration
├── cd.yml                      # Continuous Deployment
├── test.yml                    # Automated Testing
└── release.yml                 # Release Management
```

### **Testing Structure**
```python
tests/
├── unit/                       # Unit tests
│   ├── test_collectors.py
│   ├── test_processors.py
│   └── test_models.py
├── integration/                # Integration tests
│   ├── test_pipeline.py
│   └── test_api.py
├── e2e/                        # End-to-end tests
│   └── test_full_workflow.py
└── fixtures/                   # Test data
```

---

## 📈 Cấu Trúc Monitoring & Observability

### **Logging Architecture**
```
Application Logs
├── collector_*.log             # Collection activities
├── processor_*.log             # Processing activities
├── api_*.log                   # API requests/responses
├── error_*.log                 # Error tracking
└── audit_*.log                 # Security audit logs
```

### **Metrics Collection**
```python
monitoring/
├── system_metrics.py           # System resources
├── business_metrics.py         # Business KPIs
├── quality_metrics.py          # Data quality
└── performance_metrics.py      # Performance tracking
```

---

## 🔒 Cấu Trúc Security

### **Authentication & Authorization**
```python
security/
├── auth.py                     # Authentication handlers
├── permissions.py              # Permission management
├── rate_limiting.py            # Rate limiting
└── encryption.py               # Data encryption
```

### **API Security**
```python
app/api/security/
├── jwt_handler.py              # JWT token management
├── oauth_handler.py            # OAuth integration
├── cors_middleware.py          # CORS handling
└── input_validation.py         # Input sanitization
```

---

## 📋 Cấu Trúc Configuration Management

### **Configuration Hierarchy**
```
1. Default Config (config/default.yaml)
2. Environment Config (config/{env}.yaml)
3. Local Override (config/local.yaml)
4. Runtime Override (Environment Variables)
5. Command Line Args
```

### **Configuration Files**
```yaml
# collector_config.yaml
sources:
  youtube:
    enabled: true
    api_key: ${YOUTUBE_API_KEY}
    rate_limit: 100

processing:
  face_detection:
    method: mediapipe
    confidence: 0.7

output:
  base_path: ./datasets
  formats: [jpg, wav, json]
```

---

## 🔧 Cấu Trúc Development Tools

### **Development Scripts**
```bash
scripts/
├── dev_setup.sh                # Development environment setup
├── run_tests.sh                # Run test suite
├── build_docs.sh               # Build documentation
├── deploy_local.sh             # Local deployment
└── cleanup.sh                  # Clean up artifacts
```

### **Code Quality Tools**
```python
# .pre-commit-config.yaml
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.4.0
  hooks:
  - id: trailing-whitespace
  - id: end-of-file-fixer
  - id: check-yaml
  - id: check-added-large-files

- repo: https://github.com/psf/black
  rev: 22.12.0
  hooks:
  - id: black
    language_version: python3

- repo: https://github.com/pycqa/flake8
  rev: 6.0.0
  hooks:
  - id: flake8
```

---

## 📚 Cấu Trúc Documentation

### **Documentation Structure**
```
docs/
├── index.md                    # Main documentation
├── api/                        # API documentation
│   ├── collection.md
│   ├── processing.md
│   └── monitoring.md
├── guides/                     # User guides
│   ├── getting_started.md
│   ├── data_collection.md
│   └── model_training.md
├── tutorials/                  # Tutorials
│   ├── basic_collection.md
│   ├── advanced_processing.md
│   └── custom_models.md
└── reference/                  # Reference docs
    ├── config_reference.md
    ├── data_formats.md
    └── troubleshooting.md
```

---

## 🎯 Cấu Trúc Project Management

### **Project Files**
```
├── 📄 pyproject.toml             # Python project configuration
├── 📄 setup.cfg                 # Setuptools configuration
├── 📄 MANIFEST.in               # Package manifest
├── 📄 LICENSE                   # Project license
├── 📄 CODE_OF_CONDUCT.md        # Code of conduct
├── 📄 CONTRIBUTING.md           # Contributing guidelines
├── 📄 CHANGELOG.md              # Change log
└── 📄 .github/                  # GitHub configuration
    ├── 📄 ISSUE_TEMPLATE.md     # Issue templates
    ├── 📄 PULL_REQUEST_TEMPLATE.md
    └── 📄 CODEOWNERS            # Code ownership
```

---

## 🚀 Cấu Trúc Deployment Environments

### **Environment Structure**
```
environments/
├── local/                       # Local development
│   ├── docker-compose.yml
│   ├── .env
│   └── config.yaml
├── staging/                     # Staging environment
│   ├── docker-compose.yml
│   ├── .env
│   └── config.yaml
└── production/                  # Production environment
    ├── docker-compose.yml
    ├── .env
    ├── config.yaml
    └── k8s/                     # Kubernetes manifests
        ├── deployment.yaml
        ├── service.yaml
        ├── ingress.yaml
        └── configmap.yaml
```

---

## 📊 Cấu Trúc Analytics & Reporting

### **Analytics Structure**
```python
analytics/
├── data_quality.py              # Data quality analytics
├── collection_metrics.py        # Collection performance
├── model_performance.py         # Model evaluation metrics
└── business_intelligence.py     # BI dashboards
```

### **Reporting Structure**
```python
reports/
├── daily_collection_report.py   # Daily collection summary
├── weekly_quality_report.py     # Weekly quality assessment
├── monthly_performance_report.py # Monthly performance review
└── custom_reports.py            # Custom report generator
```

---

## 🔄 Cấu Trúc Migration & Updates

### **Migration Structure**
```python
migrations/
├── data/                        # Data migrations
│   ├── v1_to_v2.py
│   └── schema_updates.py
├── config/                      # Configuration migrations
│   ├── config_v1_to_v2.py
└── model/                       # Model migrations
    ├── model_updates.py
```

---

## 🎉 **Tóm Tắt**

Cấu trúc này được thiết kế theo nguyên tắc:

- **🔧 Modular**: Mỗi module độc lập, dễ maintain
- **📈 Scalable**: Dễ mở rộng theo nhu cầu
- **🧪 Testable**: Test coverage cao
- **🚀 Deployable**: CI/CD ready
- **📊 Observable**: Monitoring comprehensive
- **🔒 Secure**: Security-first approach
- **👥 Collaborative**: Team development friendly

**Cấu trúc hỗ trợ full lifecycle từ development → testing → deployment → monitoring**


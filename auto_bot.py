import os
import subprocess

# --- C?U HÌNH BOT ---
# Thu m?c luu d? li?u (d?t vào thu m?c Google Drive n?u mu?n auto-sync)
SAVE_DIR = "Gender_Raw_Data"

# S? lu?ng video mu?n t?i cho m?i t? khóa
LIMIT_PER_KEYWORD = 5

# Danh sách t? khóa (Bot s? t? tìm và t?i)
SEARCH_KEYWORDS = {
    "Male": [
        "Ph?ng v?n Tr?n Thành",
        "Talkshow Vietcetera Nam",
        "Ph?ng v?n Son Tùng MTP",
        "Vlog Khoai Lang Thang",
        "Ph?ng v?n Ðen Vâu",
        "Talkshow MC Quy?n Linh",
    ],
    "Female": [
        "Ph?ng v?n M? Tâm",
        "Talkshow Thùy Minh",
        "Ph?ng v?n Suboi",
        "Vlog Khánh Vy",
        "Talkshow Hari Won",
        "Ph?ng v?n Hoa H?u Thùy Tiên",
    ],
}


def crawl_youtube(keyword: str, output_folder: str, limit: int):
    """Dùng yt-dlp d? tìm ki?m và t?i t? d?ng theo t? khóa"""
    os.makedirs(output_folder, exist_ok=True)

    print(f" Ðang tìm ki?m và t?i: '{keyword}' (Max: {limit})...")

    # Cú pháp tìm ki?m: ytsearchN:keyword
    query = f"ytsearch{limit}:{keyword}"

    cmd = [
        "yt-dlp",
        query,  # L?nh tìm ki?m
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",  # Ch?t lu?ng t?t nh?t
        "-o",
        f"{output_folder}/%(title)s.%(ext)s",  # Tên file theo tiêu d?
        "--no-playlist",  # Không t?i playlist
        "--match-filter",
        "duration > 60 & duration < 1200",  # Ch? l?y video 1-20 phút
        "--ignore-errors",  # G?p l?i b? qua, ti?p t?c
    ]

    try:
        subprocess.call(cmd)
    except Exception as e:
        print(f" L?i khi ch?y l?nh: {e}")


def main():
    print(" BOT B?T Ð?U HO?T Ð?NG...")

    for gender, keywords in SEARCH_KEYWORDS.items():
        print(f"\n ÐANG X? LÝ NHÓM: {gender.upper()}")
        save_path = os.path.join(SAVE_DIR, gender)

        for kw in keywords:
            crawl_youtube(kw, save_path, LIMIT_PER_KEYWORD)

    print(f"\n HOÀN T?T! Ki?m tra thu m?c '{SAVE_DIR}'.")
    print(" Ð?t SAVE_DIR trong thu m?c Google Drive d? auto-sync lên cloud.")


if __name__ == "__main__":
    main()

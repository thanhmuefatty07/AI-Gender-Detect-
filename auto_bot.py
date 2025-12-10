import os
import subprocess

# --- CẤU HÌNH BOT ---
# Thư mục lưu dữ liệu (mặc định trỏ vào Google Drive for Desktop trên Windows).
# Nếu bạn dùng đường dẫn khác, sửa biến này cho đúng.
SAVE_DIR = r"G:\My Drive\Gender_Raw_Data"

# S? lu?ng video mu?n t?i cho m?i t? kh�a
LIMIT_PER_KEYWORD = 5

# Danh s�ch t? kh�a (Bot s? t? t�m v� t?i)
SEARCH_KEYWORDS = {
    "Male": [
        "Ph?ng v?n Tr?n Th�nh",
        "Talkshow Vietcetera Nam",
        "Ph?ng v?n Son T�ng MTP",
        "Vlog Khoai Lang Thang",
        "Ph?ng v?n �en V�u",
        "Talkshow MC Quy?n Linh",
    ],
    "Female": [
        "Ph?ng v?n M? T�m",
        "Talkshow Th�y Minh",
        "Ph?ng v?n Suboi",
        "Vlog Kh�nh Vy",
        "Talkshow Hari Won",
        "Ph?ng v?n Hoa H?u Th�y Ti�n",
    ],
}


def crawl_youtube(keyword: str, output_folder: str, limit: int):
    """D�ng yt-dlp d? t�m ki?m v� t?i t? d?ng theo t? kh�a"""
    os.makedirs(output_folder, exist_ok=True)

    print(f" �ang t�m ki?m v� t?i: '{keyword}' (Max: {limit})...")

    # C� ph�p t�m ki?m: ytsearchN:keyword
    query = f"ytsearch{limit}:{keyword}"

    cmd = [
        "yt-dlp",
        query,  # L?nh t�m ki?m
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",  # Ch?t lu?ng t?t nh?t
        "-o",
        f"{output_folder}/%(title)s.%(ext)s",  # T�n file theo ti�u d?
        "--no-playlist",  # Kh�ng t?i playlist
        "--match-filter",
        "duration > 60 & duration < 1200",  # Ch? l?y video 1-20 ph�t
        "--ignore-errors",  # G?p l?i b? qua, ti?p t?c
    ]

    try:
        subprocess.call(cmd)
    except Exception as e:
        print(f" L?i khi ch?y l?nh: {e}")


def main():
    print(" BOT B?T �?U HO?T �?NG...")

    for gender, keywords in SEARCH_KEYWORDS.items():
        print(f"\n �ANG X? L� NH�M: {gender.upper()}")
        save_path = os.path.join(SAVE_DIR, gender)

        for kw in keywords:
            crawl_youtube(kw, save_path, LIMIT_PER_KEYWORD)

    print(f"\n HO�N T?T! Ki?m tra thu m?c '{SAVE_DIR}'.")
    print(" �?t SAVE_DIR trong thu m?c Google Drive d? auto-sync l�n cloud.")


if __name__ == "__main__":
    main()

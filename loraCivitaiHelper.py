import os
import json
import sqlite3
import customtkinter as ctk
import hashlib
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import requests
import cv2
from PIL import Image, ImageTk
from pathlib import Path
import ctypes
import webbrowser  # 웹브라우저 제어를 위한 모듈 추가
import threading
import time
from queue import Queue

DB_FILE = "model_info.db"

# 전역 변수 추가
current_video_thread = None
video_stop_event = threading.Event()
video_queue = Queue()
current_video_image = None  # 현재 재생 중인 이미지 객체 참조 저장
current_video_button = None  # 현재 재생 중인 버튼 참조 저장

# DB 초기화 및 테이블 생성
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS models (
            modelId INTEGER PRIMARY KEY,
            safetensor TEXT,
            modelname TEXT,
            trainedWords TEXT
        )
    """)
    conn.commit()
    conn.close()

# DB에 메타데이터 저장
def insert_model_data(modelId, safetensor, modelname, trainedWords):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("REPLACE INTO models (modelId, safetensor, modelname, trainedWords) VALUES (?, ?, ?, ?)",
              (modelId, safetensor, modelname, ", ".join(trainedWords)))
    conn.commit()
    conn.close()

# DB에서 검색
def search_models(keyword):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if keyword.strip() == "":
        query = "SELECT modelId, safetensor, modelname FROM models"
        c.execute(query)
    else:
        query = """
            SELECT modelId, safetensor, modelname FROM models
            WHERE safetensor LIKE ? OR modelname LIKE ?
        """
        c.execute(query, (f"%{keyword}%", f"%{keyword}%"))
    results = c.fetchall()
    conn.close()
    return results

# SHA256 해시 계산
def compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    hex_digest = sha256_hash.hexdigest()
    #print(f"[DEBUG] {file_path.name} SHA256: {hex_digest}")
    return hex_digest

# civitai에서 모델 정보 조회
def fetch_model_info_by_hash(sha256):
    url_by_hash = f"https://civitai.com/api/v1/model-versions/by-hash/{sha256}"
    response = requests.get(url_by_hash)
    if response.status_code != 200:
        print(f"요청 실패: {response.status_code}")
        return None

    version_info = response.json()
    model_version_id = version_info.get("id")
    if not model_version_id:
        print("모델 버전 ID를 찾을 수 없습니다.")
        return None

    url_by_id = f"https://civitai.com/api/v1/model-versions/{model_version_id}"
    response = requests.get(url_by_id)
    if response.status_code != 200:
        print(f"상세 정보 요청 실패: {response.status_code}")
        return None

    detailed_info = response.json()
    return detailed_info

# 미리보기 이미지 또는 동영상 다운로드
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

# 대표 미리보기 URL 선택
def get_preview_url(version):
    for img in version.get("images", []):
        url = img.get("url")
        if not url:
            continue
        if url.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return url, "image"
        elif url.endswith((".mp4", ".webm")):
            return url, "video"
    return None, None

# safetensors 파일 처리 및 메타데이터 저장
def process_safetensors_files(folder_path):
    folder = Path(folder_path)
    for file in folder.glob("*.safetensors"):
        print(f"처리 중: {file.name}")
        
        # 미리보기와 JSON 파일이 모두 존재하는지 확인
        info_path = file.with_suffix(".civitai.info.json")
        preview_path = file.with_suffix(".preview.png")
        video_preview_path = file.with_suffix(".preview.mp4")
        
        if info_path.exists() and (preview_path.exists() or video_preview_path.exists()):
            print(f"스킵: 이미 모든 파일이 존재합니다: {file.name}")
            continue
        
        # 파일이 존재하지 않는 경우만 처리
        sha256 = compute_sha256(file)
        version = fetch_model_info_by_hash(sha256)

        if version:
            # JSON 파일이 이미 존재하는지 확인
            if not info_path.exists():
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(version, f, ensure_ascii=False, indent=4)
                print(f"메타데이터 저장 완료: {info_path.name}")
            else:
                print(f"스킵: 이미 존재하는 메타데이터 파일: {info_path.name}")

            preview_url, media_type = get_preview_url(version)
            if preview_url:
                ext = ".preview.png" if media_type == "image" else ".preview.mp4"
                save_path = file.with_suffix(ext)
                # 미리보기 파일이 이미 존재하는지 확인
                if not save_path.exists():
                    if download_file(preview_url, save_path):
                        print(f"{media_type.upper()} 미리보기 저장 완료: {save_path.name}")
                else:
                    print(f"스킵: 이미 존재하는 미리보기 파일: {save_path.name}")
        else:
            print(f"⚠️ 모델 정보를 찾을 수 없습니다: {file.name}")

def round_corners(image, radius):
    """이미지의 모서리를 라운드 처리하는 함수"""
    # 이미지가 RGBA 모드가 아니면 변환
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # 새로운 이미지 생성
    rounded = Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    # 마스크 생성 (2배 크기로 생성하여 안티앨리어싱 효과 적용)
    mask_size = (image.size[0] * 2, image.size[1] * 2)
    mask = Image.new('L', mask_size, 0)
    draw = ImageDraw.Draw(mask)
    
    # 라운드 처리된 사각형 그리기 (2배 크기로)
    draw.rounded_rectangle([(0, 0), mask_size], radius * 2, fill=255)
    
    # 마스크 크기 조정 (안티앨리어싱 적용)
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    
    # 마스크를 사용하여 이미지 합성
    rounded.paste(image, mask=mask)
    return rounded

# 전역 변수로 앱 인스턴스 관리
app = None
is_resizing = False
resize_timer = None
last_width = 0
last_height = 0
update_grid_timer = None  # update_grid 디바운스용 타이머

# GUI 시작
def launch_gui(preview_folder):
    global app, detail_frame, result_frame, is_resizing, resize_timer
    
    is_resizing = False
    resize_timer = None
    
    # 메인 윈도우 생성
    app = ctk.CTk()
    app.title("Lora Viewer")
    app.geometry("1200x900")
    app.minsize(1000, 800)
    
    # 창 닫기 이벤트 처리
    def on_closing():
        global app
        if app is not None:
            app.destroy()
            app = None
        try:
            app.quit()
        except:
            pass
    
    app.protocol("WM_DELETE_WINDOW", on_closing)
    
    # 메인 컨테이너
    main_container = ctk.CTkFrame(app)
    main_container.pack(fill="both", expand=True, padx=10, pady=10)
    
    # 검색 프레임
    search_frame = ctk.CTkFrame(main_container)
    search_frame.pack(fill="x", pady=(0, 10))
    
    # 검색 입력창
    search_entry = ctk.CTkEntry(
        search_frame, 
        placeholder_text="모델 이름 또는 safetensor 검색",
        height=36,
        font=("Arial", 12)
    )
    search_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
    
    # 결과를 표시할 프레임 (스크롤 가능한 캔버스 포함)
    result_frame = ctk.CTkFrame(main_container, border_width=0)
    result_frame.pack(fill="both", expand=True, pady=(0, 10))
    
    # 캔버스와 스크롤바 생성
    canvas = ctk.CTkCanvas(result_frame)
    canvas.configure(bg=main_container._apply_appearance_mode(main_container._fg_color))
    scrollbar = ctk.CTkScrollbar(result_frame, orientation="vertical", command=canvas.yview)
    scrollable_frame = ctk.CTkFrame(canvas, border_width=0)
    
    # 스크롤 가능한 프레임 구성
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    
    # 캔버스에 스크롤 가능한 프레임 추가
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # 마우스 휠 이벤트 처리
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    # 마우스 휠 이벤트 바인딩
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    # 캔버스와 스크롤바 배치
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # 상세 정보를 표시할 프레임 (하단 고정 높이)
    detail_frame = ctk.CTkFrame(main_container, height=200)
    detail_frame.pack(fill="x", pady=(0, 10))
    detail_frame.pack_propagate(False)  # 고정 높이 유지
    
    # 창 크기 조절 관련 변수
    resize_timer = None
    last_width = 0
    last_height = 0
    
    # 창 크기 조절 이벤트 처리
    def on_resize(event):
        global resize_timer, last_width, last_height
        
        # 이벤트가 발생한 위젯이 메인 윈도우인 경우에만 처리
        if event.widget != app:
            return
            
        # 크기가 실제로 변경되었는지 확인
        if event.width == last_width and event.height == last_height:
            return
            
        # 이전 타이머 취소
        if resize_timer:
            app.after_cancel(resize_timer)
        
        # 새 타이머 설정 (300ms 후에 업데이트)
        resize_timer = app.after(300, lambda: handle_resize(event.width, event.height))
    
    def handle_resize(width, height):
        global last_width, last_height
        if width != last_width or height != last_height:
            last_width = width
            last_height = height
            print(f"창 크기 변경: {width}x{height}")
            update_grid()
    
    # 창 크기 조절 이벤트 바인딩
    app.bind("<Configure>", on_resize)
    
    # update_grid 함수 정의
    def update_grid(*args):
        global update_grid_timer
        
        # 이전 타이머가 있으면 취소
        if update_grid_timer:
            app.after_cancel(update_grid_timer)
        
        # 새로운 타이머 설정 (100ms 후에 실제 업데이트 실행)
        update_grid_timer = app.after(100, _update_grid)
    
    def _update_grid():
        try:
            print("그리드 업데이트 시작")  # 디버그 메시지
            # 기존 위젯 제거
            for widget in scrollable_frame.winfo_children():
                try:
                    widget.destroy()
                except Exception as e:
                    print(f"위젯 제거 중 오류 발생: {e}")
                    continue
            
            if not hasattr(on_search, 'results') or not on_search.results:
                return
            
            # 기본 크기 설정
            preview_size = 180
            padding = 5
            
            # 실제 컨테이너 너비 계산 (스크롤바 공간 제외)
            # 창 크기가 1200x900으로 고정되어 있으므로, 실제 사용 가능한 너비는 1200 - 20(스크롤바) - 20(패딩)
            container_width = 1160  # 1200 - 20 - 20
            
            if container_width < 1:
                return
            
            # 한 줄에 들어갈 수 있는 아이템 수 계산
            items_per_row = max(1, int(container_width // (preview_size + 2 * padding)))
            
            # 전체 결과 수에 대한 행 수 계산
            total_rows = (len(on_search.results) + items_per_row - 1) // items_per_row
            
            # 마지막 줄의 아이템 수 계산
            last_row_items = len(on_search.results) % items_per_row
            if last_row_items == 0:
                last_row_items = items_per_row
                
            # 마지막 줄이 프레임 밖으로 나가는지 확인
            item_width = preview_size + 2 * padding
            last_row_width = item_width * last_row_items
            print(f"마지막 줄 너비: {last_row_width}, 컨테이너 너비: {container_width}")
            print(f"아이템 크기(패딩 포함): {item_width}")
            if last_row_width > container_width and total_rows > 1:
                # 한 줄에 표시되는 아이템 수를 줄임
                items_per_row = max(1, items_per_row - 1)
                print(f"마지막 줄이 프레임을 벗어나 줄 수 조정: {items_per_row}")
            
            print(f"실제 컨테이너 너비: {container_width}")
            print(f"아이템 크기(패딩 포함): {item_width}")
            print(f"한 줄당 아이템 수: {items_per_row}")
            
            # 이미지 로딩을 위한 스레드 풀 생성
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=4)
            
            def load_preview_image(idx, modelId, safetensor, modelname):
                try:
                    # 미리보기 파일 경로 생성
                    safetensor_path = Path(preview_folder) / safetensor
                    thumb_path = safetensor_path.with_suffix(".preview.png")
                    video_path = safetensor_path.with_suffix(".preview.mp4")
                    
                    # 이미지/비디오 로딩
                    preview_loaded = False
                    preview_img = None
                    is_video = False
                    
                    if video_path.exists():
                        try:
                            cap = cv2.VideoCapture(str(video_path))
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                preview_img = Image.fromarray(frame)
                                preview_loaded = True
                                is_video = True
                            cap.release()
                        except Exception as e:
                            print(f"Error loading video {video_path}: {e}")
                    
                    if not preview_loaded and thumb_path.exists():
                        try:
                            preview_img = Image.open(thumb_path)
                            preview_loaded = True
                        except Exception as e:
                            print(f"Error loading image {thumb_path}: {e}")
                    
                    # GUI 업데이트는 메인 스레드에서 실행
                    if preview_loaded:
                        app.after(0, lambda: create_preview_widgets(idx, modelId, modelname, preview_img, is_video, video_path))
                    else:
                        app.after(0, lambda: create_no_preview_widget(idx, modelId, modelname))
                    
                except Exception as e:
                    print(f"Error in load_preview_image: {e}")
            
            def create_preview_widgets(idx, modelId, modelname, preview_img, is_video, video_path):
                try:
                    if not scrollable_frame.winfo_exists():
                        return
                        
                    # 미리보기 프레임 생성
                    row = idx // items_per_row
                    col = idx % items_per_row
                    
                    preview_frame = ctk.CTkFrame(scrollable_frame, width=preview_size, height=preview_size)
                    preview_frame.grid(row=row, column=col, padx=padding, pady=padding, sticky="nsew")
                    preview_frame.grid_propagate(False)
                    
                    # 이미지 크기 조정 (비율 유지)
                    preview_img.thumbnail((preview_size, preview_size))
                    radius = int(preview_size * 0.1)
                    preview_img = round_corners(preview_img, radius)
                    
                    # 배경색 설정
                    bg_color = ("gray90", "gray20")
                    
                    # 이미지 컨테이너
                    container = ctk.CTkFrame(preview_frame, fg_color=bg_color)
                    container.place(relx=0.5, rely=0.5, anchor="center", relwidth=1, relheight=1)
                    
                    # 이미지 버튼
                    ctk_img = ctk.CTkImage(preview_img, size=(preview_size, preview_size))
                    img_btn = ctk.CTkButton(
                        container,
                        image=ctk_img,
                        text="",
                        fg_color="transparent",
                        hover_color=("gray70", "gray30"),
                        command=lambda mid=modelId: show_details(mid)
                    )
                    img_btn.place(relx=0.5, rely=0.5, anchor="center")
                    
                    # 모델 이름 표시 (상단)
                    name_frame = ctk.CTkFrame(container, fg_color=("gray80", "gray30"), height=30)
                    name_frame.place(relx=0.5, rely=0, anchor="n", relwidth=1)
                    
                    name_label = ctk.CTkLabel(
                        name_frame,
                        text=modelname,
                        font=("Arial", 12, "bold"),
                        text_color=("black", "white")
                    )
                    name_label.place(relx=0.5, rely=0.5, anchor="center")
                    
                    # 텍스트 애니메이션 설정
                    if len(modelname) > 20:
                        name_label.text = modelname
                        name_label.original_text = modelname
                        name_label.animation_id = None
                        name_label.position = 0
                        
                        def animate_text(label):
                            if not label.winfo_exists():
                                return
                            label.position = (label.position + 1) % len(label.original_text)
                            display_text = label.original_text[label.position:] + label.original_text[:label.position]
                            label.configure(text=display_text)
                            label.animation_id = label.after(200, lambda: animate_text(label))
                        
                        animate_text(name_label)
                    
                    # 더블클릭 이벤트
                    def on_double_click(event, mid=modelId):
                        url = f"https://civitai.com/models/{mid}/"
                        webbrowser.open(url)
                    
                    img_btn.bind("<Double-Button-1>", on_double_click)
                    
                    # 동영상 처리
                    if is_video:
                        def on_enter(event, vpath=video_path, psize=preview_size, cont=container, btn=img_btn):
                            start_video_playback(vpath, psize, cont, btn)
                        
                        def on_leave(event):
                            stop_video_playback()
                        
                        img_btn.bind("<Enter>", on_enter)
                        img_btn.bind("<Leave>", on_leave)
                        
                        # 동영상 아이콘
                        icon_frame = ctk.CTkFrame(container, fg_color="transparent")
                        icon_frame.place(relx=0.95, rely=0.95, anchor="se")
                        
                        video_icon = ctk.CTkLabel(
                            icon_frame,
                            text="VIDEO",
                            font=("Arial", 14, "bold"),
                            text_color="white"
                        )
                        video_icon.pack()
                        
                except Exception as e:
                    print(f"Error creating preview widgets: {e}")
            
            def create_no_preview_widget(idx, modelId, modelname):
                try:
                    if not scrollable_frame.winfo_exists():
                        return
                        
                    row = idx // items_per_row
                    col = idx % items_per_row
                    
                    preview_frame = ctk.CTkFrame(scrollable_frame, width=preview_size, height=preview_size)
                    preview_frame.grid(row=row, column=col, padx=padding, pady=padding, sticky="nsew")
                    preview_frame.grid_propagate(False)
                    
                    no_preview = ctk.CTkLabel(
                        preview_frame,
                        text="No Preview\n" + modelname[:20] + ("..." if len(modelname) > 20 else ""),
                        wraplength=preview_size-20,
                        justify="center",
                        font=("Arial", 12)
                    )
                    no_preview.place(relx=0.5, rely=0.5, anchor="center")
                    
                    def on_double_click(event, mid=modelId):
                        url = f"https://civitai.com/models/{mid}/"
                        webbrowser.open(url)
                    
                    no_preview.bind("<Double-Button-1>", on_double_click)
                    
                except Exception as e:
                    print(f"Error creating no preview widget: {e}")
            
            # 이미지 로딩 작업 제출
            futures = []
            for idx, (modelId, safetensor, modelname) in enumerate(on_search.results):
                future = executor.submit(load_preview_image, idx, modelId, safetensor, modelname)
                futures.append(future)
            
            # 그리드의 열 가중치 설정
            for i in range(items_per_row):
                result_frame.columnconfigure(i, weight=1)
            
            # 작업 완료 대기 및 GUI 업데이트
            def check_futures():
                if all(future.done() for future in futures):
                    executor.shutdown()
                    return
                
                # 아직 완료되지 않은 작업이 있으면 계속 체크
                app.after(100, check_futures)
            
            # 비동기 작업 시작
            app.after(0, check_futures)
            
        except Exception as e:
            print(f"Error in update_grid: {e}")
    
    # 검색 함수 정의
    def on_search():
        keyword = search_entry.get()
        on_search.results = search_models(keyword)
        
        # 검색 결과가 없을 때 처리
        if not on_search.results:
            # 기존 위젯 제거
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
                
            no_result = ctk.CTkLabel(
                scrollable_frame, 
                text="검색 결과가 없습니다.", 
                font=ctk.CTkFont(size=12, weight="bold")
            )
            no_result.place(relx=0.5, rely=0.5, anchor="center")
            return
        
        # 검색 결과가 있으면 그리드 업데이트
        update_grid()
    
    # 검색 버튼
    search_btn = ctk.CTkButton(
        search_frame, 
        text="검색", 
        width=80, 
        height=36,
        font=("Arial", 12, "bold"),
        command=on_search
    )
    search_btn.pack(side="right", padx=(5, 0))
    
    # 갱신 버튼
    def refresh_data():
        # 기존 검색 결과 저장
        current_keyword = search_entry.get()
        
        # DB 갱신
        scan_and_update_db(preview_folder)
        process_safetensors_files(preview_folder)
        
        # 검색 결과 업데이트 및 화면 갱신
        on_search.results = search_models(current_keyword)
        
        # 기존 위젯 제거
        for widget in scrollable_frame.winfo_children():
            widget.destroy()
            
        # 검색 결과가 없을 때 처리
        if not on_search.results:
            no_result = ctk.CTkLabel(
                scrollable_frame, 
                text="검색 결과가 없습니다.", 
                font=ctk.CTkFont(size=12, weight="bold")
            )
            no_result.place(relx=0.5, rely=0.5, anchor="center")
        else:
            # 검색 결과가 있으면 그리드 업데이트
            update_grid()
        
        # 갱신 완료 메시지
        refresh_btn.configure(text="갱신 완료!")
        app.after(2000, lambda: refresh_btn.configure(text="갱신"))

    refresh_btn = ctk.CTkButton(
        search_frame,
        text="갱신",
        width=80,
        height=36,
        font=("Arial", 12, "bold"),
        command=refresh_data
    )
    refresh_btn.pack(side="right", padx=(5, 0))
    
    # 검색창 엔터 이벤트 바인딩
    def on_enter(event):
        on_search()
    search_entry.bind("<Return>", on_enter)
    
    # 윈도우 크기 변경 이벤트 바인딩
    result_frame.bind("<Configure>", update_grid)
    
    # 초기 검색 실행
    on_search.results = []
    on_search()  # 초기 검색 실행
    
    # 메인 루프 시작
    try:
        app.mainloop()
    except KeyboardInterrupt:
        if app is not None:
            app.destroy()
    except Exception as e:
        print(f"An error occurred: {e}")
        if app is not None:
            app.destroy()

def show_details(modelId):
    global app
    if not app:
        return
        
    # 기존 상세 정보 제거
    for widget in detail_frame.winfo_children():
        widget.destroy()
    
    # DB에서 모델 정보 조회
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM models WHERE modelId = ?", (modelId,))
    row = c.fetchone()
    conn.close()
    
    if row:
        # 상세 정보 표시
        detail_text = ctk.CTkTextbox(detail_frame, wrap="word", height=200)
        detail_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 모델 정보 포맷팅
        model_info = f"Model ID: {row[0]}\n"
        model_info += f"File: {row[1]}\n"
        model_info += f"Name: {row[2]}\n"
        
        # trainedWords 처리 (eval 대신 안전한 파싱 사용)
        if row[3]:  # trainedWords가 있는 경우
            try:
                # 먼저 JSON 형식으로 파싱 시도
                import json
                trained_words = json.loads(row[3])
                if isinstance(trained_words, list):
                    model_info += f"Trained Words: {', '.join(trained_words) if trained_words else 'None'}\n"
                else:
                    model_info += f"Trained Words: {trained_words}\n"
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 그대로 표시
                model_info += f"Trained Words: {row[3]}\n"
        else:
            model_info += "Trained Words: None\n"
        
        detail_text.insert("1.0", model_info)
        detail_text.configure(state="disabled")  # 편집 불가능하도록 설정

# JSON 파싱 후 DB 저장 루틴
def scan_and_update_db(folder):
    init_db()
    for json_path in Path(folder).glob("*.civitai.info.json"):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        modelId = data.get("modelId")
        modelname = data.get("model", {}).get("name", "")
        trainedWords = data.get("trainedWords", [])
        safetensor_file = json_path.name.replace(".civitai.info.json", ".safetensors")
        if modelId:
            insert_model_data(modelId, safetensor_file, modelname, trainedWords)

# 동영상 재생 스레드 함수
def video_playback_thread(video_path, preview_size, container, img_btn):
    global video_stop_event, current_video_image, current_video_button
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        # 첫 프레임을 미리 로드
        ret, frame = cap.read()
        if not ret:
            return
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview_img = Image.fromarray(frame)
        preview_img.thumbnail((preview_size, preview_size))
        preview_img = round_corners(preview_img, int(preview_size * 0.1))
        
        # 초기 이미지 설정
        current_video_image = ctk.CTkImage(preview_img, size=(preview_size, preview_size))
        img_btn.configure(image=current_video_image)
        current_video_button = img_btn
        
        while not video_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_img = Image.fromarray(frame)
            preview_img.thumbnail((preview_size, preview_size))
            preview_img = round_corners(preview_img, int(preview_size * 0.1))
            
            # 새로운 이미지 생성
            new_image = ctk.CTkImage(preview_img, size=(preview_size, preview_size))
            
            # 이미지 업데이트를 메인 스레드에서 실행
            if not video_stop_event.is_set():
                img_btn.after(1, lambda img=new_image: update_video_frame(img))
            
            time.sleep(frame_delay)
            
    except Exception as e:
        print(f"Error in video playback: {e}")
    finally:
        cap.release()
        if current_video_button == img_btn:
            current_video_image = None
            current_video_button = None

def update_video_frame(new_image):
    global current_video_image, current_video_button
    
    if current_video_button and not video_stop_event.is_set():
        current_video_image = new_image
        current_video_button.configure(image=new_image)

# 동영상 재생 시작 함수
def start_video_playback(video_path, preview_size, container, img_btn):
    global current_video_thread, video_stop_event, current_video_button
    
    # 이전 재생 중지
    stop_video_playback()
    
    # 새로운 재생 시작
    video_stop_event.clear()
    current_video_button = img_btn
    current_video_thread = threading.Thread(
        target=video_playback_thread,
        args=(video_path, preview_size, container, img_btn),
        daemon=True
    )
    current_video_thread.start()

# 동영상 재생 중지 함수
def stop_video_playback():
    global current_video_thread, video_stop_event, current_video_image, current_video_button
    
    if current_video_thread and current_video_thread.is_alive():
        video_stop_event.set()
        current_video_thread.join(timeout=1.0)
        current_video_thread = None
        current_video_image = None
        current_video_button = None

if __name__ == "__main__":
    scan_and_update_db("lora 경로를 이곳에")
    process_safetensors_files("lora 경로를 이곳에")
    launch_gui("lora 경로를 이곳에")

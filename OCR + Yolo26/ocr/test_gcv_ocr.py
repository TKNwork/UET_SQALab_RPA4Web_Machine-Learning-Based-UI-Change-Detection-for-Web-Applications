import os
import sys
import time
import base64
import json
import requests
from PIL import Image, ImageDraw, ImageFont

# Xác định thư mục gốc của project (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn tĩnh/động theo cấu trúc thư mục
KEY_FILE = os.path.join(BASE_DIR, 'config', 'ocrkey.txt')

VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

OCR_RESULTS_DIR = os.path.join(BASE_DIR, 'ocr', 'results')
VISUALIZED_DIR = os.path.join(OCR_RESULTS_DIR, 'visualized')

API_ENDPOINT = 'https://vision.googleapis.com/v1/images:annotate'

# Ngưỡng confidence lọc phần tử OCR Paragraph (>= 0.8)
CONF_THRESH = 0.80


def load_api_key(key_filepath):
    key = os.environ.get('GOOGLE_VISION_API_KEY', '').strip()
    if key:
        return key
    if not os.path.exists(key_filepath):
        fallback_key = os.path.join(BASE_DIR, 'ocrkey.txt')
        if os.path.exists(fallback_key):
            key_filepath = fallback_key
        else:
            print(f"[LỖI] File API key '{key_filepath}' không tồn tại.")
            sys.exit(1)
        
    with open(key_filepath, 'r', encoding='utf-8') as f:
        key = f.read().strip()
        
    if not key:
        print(f"[CẢNH BÁO] File '{key_filepath}' đang rỗng! Vui lòng thêm API Key vào file.")
        
    return key


def get_first_30_images(images_dir, n=30):
    if not os.path.exists(images_dir):
        print(f"[LỖI] Thư mục ảnh '{images_dir}' không tồn tại.")
        sys.exit(1)
        
    files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'))
    ])
    
    return files[:n]


def extract_paragraphs_from_response(resp_obj, conf_threshold=CONF_THRESH):
    """
    Trích xuất danh sách paragraph từ Google Cloud Vision response
    Lọc bỏ các phần tử có confidence < conf_threshold (mặc định >= 0.8)
    """
    full_text_annotation = resp_obj.get('fullTextAnnotation', {})
    paragraphs_list = []
    
    for page in full_text_annotation.get('pages', []):
        for block in page.get('blocks', []):
            for para in block.get('paragraphs', []):
                para_text = ""
                for word in para.get('words', []):
                    word_str = ""
                    for symbol in word.get('symbols', []):
                        word_str += symbol.get('text', '')
                        d_break = symbol.get('property', {}).get('detectedBreak', {})
                        break_type = d_break.get('type')
                        if break_type in ('SPACE', 'SURE_SPACE'):
                            word_str += ' '
                        elif break_type in ('LINE_BREAK', 'EOL_SURE_SPACE'):
                            word_str += '\n'
                        elif break_type == 'HYPHEN':
                            word_str += '-\n'
                    para_text += word_str
                
                vertices = para.get('boundingBox', {}).get('vertices', [])
                xs = [v.get('x', 0) for v in vertices]
                ys = [v.get('y', 0) for v in vertices]
                bbox = [min(xs), min(ys), max(xs), max(ys)] if xs and ys else [0, 0, 0, 0]
                confidence = round(para.get('confidence', 0.0), 2)
                
                # LỌC PARAGRAPH VỚI THRESHOLD >= 0.8
                if confidence >= conf_threshold:
                    paragraphs_list.append({
                        "text": para_text.strip(),
                        "bbox": bbox,
                        "confidence": confidence
                    })
                
    return paragraphs_list


def draw_paragraphs_visualization(image_path, paragraphs, output_image_path):
    """Vẽ bounding box rectangle màu xanh lá quanh mỗi paragraph có confidence >= 0.8 và ghi confidence màu đỏ"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    for para in paragraphs:
        bbox = para.get('bbox', [])
        confidence = para.get('confidence', 0.0)
        
        if len(bbox) == 4 and confidence >= CONF_THRESH:
            x_min, y_min, x_max, y_max = bbox
            # Vẽ rectangle màu xanh lá (0, 255, 0), viền 2px
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=2)
            
            # Ghi confidence màu đỏ (255, 0, 0) phía trên box
            conf_str = f"{confidence:.2f}"
            text_pos = (x_min, max(0, y_min - 12))
            
            text_bbox = draw.textbbox(text_pos, conf_str, font=font) if hasattr(draw, 'textbbox') else None
            if text_bbox:
                draw.rectangle(text_bbox, fill=(0, 0, 0))
            draw.text(text_pos, conf_str, fill=(255, 0, 0), font=font)
            
    img.save(output_image_path)


def process_ocr(image_path, api_key, conf_threshold=CONF_THRESH):
    # Encode ảnh sang Base64
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {
        "requests": [
            {
                "image": {
                    "content": encoded_string
                },
                "features": [
                    {
                        "type": "DOCUMENT_TEXT_DETECTION"
                    }
                ]
            }
        ]
    }
    
    url = f"{API_ENDPOINT}?key={api_key}"
    
    start_time = time.time()
    try:
        response = requests.post(
            url,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code != 200:
            try:
                err_data = response.json()
                err_code = err_data.get('error', {}).get('code', response.status_code)
                err_msg = err_data.get('error', {}).get('message', response.text)
            except Exception:
                err_code = response.status_code
                err_msg = response.text
            return False, [], elapsed_time, f"HTTP Error {err_code}: {err_msg}"
            
        res_json = response.json()
        responses = res_json.get('responses', [])
        
        if not responses:
            return False, [], elapsed_time, "API trả về response rỗng"
            
        resp_obj = responses[0]
        
        if 'error' in resp_obj:
            err_code = resp_obj['error'].get('code', 'Unknown')
            err_msg = resp_obj['error'].get('message', 'Unknown error')
            return False, [], elapsed_time, f"API Error {err_code}: {err_msg}"
            
        paragraphs = extract_paragraphs_from_response(resp_obj, conf_threshold=conf_threshold)
        return True, paragraphs, elapsed_time, None

    except Exception as e:
        elapsed_time = time.time() - start_time
        return False, [], elapsed_time, f"Ngoại lệ (Exception): {str(e)}"


def refilter_and_redraw_existing_results():
    """
    Hàm tiện ích lọc và vẽ lại tất cả các file JSON/Ảnh đã có trong ocr/results theo ngưỡng confidence >= 0.8
    """
    if not os.path.exists(OCR_RESULTS_DIR):
        return

    json_files = sorted([f for f in os.listdir(OCR_RESULTS_DIR) if f.endswith('.json')])
    print(f"\n--- TIẾN HÀNH LỌC VÀ VẼ LẠI {len(json_files)} KẾT QUẢ OCR VỚI CONFIDENCE >= {CONF_THRESH} ---")
    
    total_before = 0
    total_after = 0

    for idx, json_file in enumerate(json_files, 1):
        json_path = os.path.join(OCR_RESULTS_DIR, json_file)
        base_name, _ = os.path.splitext(json_file)
        img_name = f"{base_name}.png"
        img_path = os.path.join(VAL_IMAGES_DIR, img_name)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        orig_paragraphs = data.get('paragraphs', [])
        filtered_paragraphs = [p for p in orig_paragraphs if p.get('confidence', 0.0) >= CONF_THRESH]

        total_before += len(orig_paragraphs)
        total_after += len(filtered_paragraphs)

        # Lưu lại JSON đã lọc
        data['paragraphs'] = filtered_paragraphs
        data['confidence_threshold'] = CONF_THRESH
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Vẽ lại ảnh kết quả trong ocr/results/visualized/
        if os.path.exists(img_path):
            vis_path = os.path.join(VISUALIZED_DIR, f"{base_name}_bbox.png")
            draw_paragraphs_visualization(img_path, filtered_paragraphs, vis_path)

        print(f"[{idx}/{len(json_files)}] {json_file}: {len(orig_paragraphs)} -> {len(filtered_paragraphs)} paragraphs (>= {CONF_THRESH})")

    print(f"-> Tổng cộng 30 ảnh OCR: {total_before} -> {total_after} paragraphs sau khi lọc.")


def run_ocr(image_path, api_key=None, conf_threshold=CONF_THRESH):
    """
    Hàm reusable chạy OCR cho 1 ảnh và trả về danh sách paragraph nodes.
    Không crash nếu API fail (trả về danh sách rỗng).
    """
    if api_key is None:
        try:
            api_key = load_api_key(KEY_FILE)
        except Exception:
            api_key = None

    if not api_key:
        print("WARNING: OCR API Key not found or empty.")
        return []

    success, paragraphs, elapsed_time, err_msg = process_ocr(image_path, api_key, conf_threshold=conf_threshold)
    if not success:
        print(f"WARNING: OCR failed. {err_msg}")
        return []

    return paragraphs


def main():
    print(f"=== BẮT ĐẦU TEST GOOGLE CLOUD VISION OCR PARAGRAPH-LEVEL (CONFIDENCE >= {CONF_THRESH}) ===")
    api_key = load_api_key(KEY_FILE)
    
    images = get_first_30_images(VAL_IMAGES_DIR, n=30)
    print(f"Đã tìm thấy {len(images)} ảnh đầu tiên trong '{VAL_IMAGES_DIR}':")
    print("-" * 60)
    
    os.makedirs(OCR_RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZED_DIR, exist_ok=True)
    
    # 1. Lọc và vẽ lại ngay các file đã có
    refilter_and_redraw_existing_results()
    
    print("\n" + "=" * 60)
    print(f"=== TỔNG KẾT QUÁ TRÌNH LỌC VÀ VẼ LẠI PARAGRAPH-LEVEL OCR (THRESHOLD >= {CONF_THRESH}) ===")
    print(f"  - Thư mục JSON      : {os.path.relpath(OCR_RESULTS_DIR, BASE_DIR)}/")
    print(f"  - Thư mục Visualized: {os.path.relpath(VISUALIZED_DIR, BASE_DIR)}/")
    print("===================================================")


if __name__ == "__main__":
    main()

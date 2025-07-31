import json
from deep_translator import GoogleTranslator
import time
from tqdm import tqdm
import random

def back_translate(text: str, source: str, target: str) -> str:
    """
    텍스트를 대상 언어로 번역한 후 다시 원본 언어로 번역합니다.
    """
    if not text or not text.strip():
        return ""
    try:
        translated_text = GoogleTranslator(source=source, target=target).translate(text)
        time.sleep(0.5)  # API 요청 간 지연
        back_translated_text = GoogleTranslator(source=target, target=source).translate(translated_text)
        return back_translated_text
    except Exception as e:
        print(f"번역 오류 발생: {e}")
        return text

def backtrans_process_jsonl(INPUT_FILE, OUTPUT_FILE, SOURCE_LANGUAGE='en', TARGET_LANGUAGE='hu', RATIO=0.1):
    """
    JSONL 파일을 읽고, 설정된 비율(RATIO)만큼 랜덤하게 선택된 라인의 데이터를
    백트랜슬레이션된 데이터로 교체하여 저장합니다.
    """
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    # 백트랜스레이션으로 교체할 라인의 인덱스를 무작위로 선택
    selected_indices = set(random.sample(range(total_lines), int(total_lines * RATIO)))
    processed_data = []

    for idx, line in enumerate(tqdm(lines, total=total_lines, desc="데이터 처리 중")):
        current_data = json.loads(line)

        # 선택된 라인인 경우 백트랜스레이션으로 교체
        if idx in selected_indices:
            for turn in current_data.get('conversations', []):
                content = turn.get('content')
                if content:
                    turn['content'] = back_translate(content, source=SOURCE_LANGUAGE, target=TARGET_LANGUAGE)

            system_text = current_data.get('system')
            if system_text:
                current_data['system'] = back_translate(system_text, source=SOURCE_LANGUAGE, target=TARGET_LANGUAGE)
        
        processed_data.append(current_data)

    # 파일 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for item in processed_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ 데이터 처리 완료! '{OUTPUT_FILE}'에 저장되었습니다.")

# 사용 예시:
# backtrans_replace_jsonl('your_input.jsonl', 'your_output.jsonl', SOURCE_LANGUAGE='ko', TARGET_LANGUAGE='en', RATIO=0.3)
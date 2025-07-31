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

def backtrans_process_jsonl(INPUT_FILE, OUTPUT_FILE, SOURCE_LANGUAGE='en', TARGET_LANGUAGE='ko', RATIO=0.3):
    """
    JSONL 파일을 읽고, 원본 데이터를 모두 저장하고,
    설정된 비율(RATIO)만큼 랜덤하게 선택된 라인의 백트랜슬레이션 데이터를 추가로 저장합니다.
    """
    # 파일 전체 읽기
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    selected_indices = set(random.sample(range(total_lines), int(total_lines * RATIO)))

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for idx, line in enumerate(tqdm(lines, total=total_lines, desc="데이터 증강 중")):
            original_data = json.loads(line)
            # 1. 원본 라인 먼저 저장
            outfile.write(json.dumps(original_data, ensure_ascii=False) + '\n')

            # 2. 선택된 라인만 백트랜슬레이션 후 추가 저장
            if idx in selected_indices:
                augmented_data = json.loads(line)  # 원본 복사
                for turn in augmented_data.get('conversations', []):
                    content = turn.get('content')
                    if content:
                        turn['content'] = back_translate(content, source=SOURCE_LANGUAGE, target=TARGET_LANGUAGE)

                system_text = augmented_data.get('system')
                if system_text:
                    augmented_data['system'] = back_translate(system_text, source=SOURCE_LANGUAGE, target=TARGET_LANGUAGE)

                # 백트랜슬레이션 버전 추가 저장
                outfile.write(json.dumps(augmented_data, ensure_ascii=False) + '\n')

    print(f"\n✅ 데이터 증강 완료! 백트랜슬레이션 데이터가 '{OUTPUT_FILE}'에 저장되었습니다.")

import json
from deep_translator import GoogleTranslator
import time
from tqdm import tqdm

def back_translate(text: str, source: str, target: str) -> str:
    """
    텍스트를 대상 언어로 번역한 후 다시 원본 언어로 번역합니다.
    """
    if not text or not text.strip():
        return ""
    try:
        # 소스 -> 타겟 언어로 번역
        translated_text = GoogleTranslator(source=source, target=target).translate(text)
        # 안정적인 API 요청을 위해 약간의 지연 시간 추가
        time.sleep(0.5)
        # 타겟 -> 소스 언어로 다시 번역
        back_translated_text = GoogleTranslator(source=target, target=source).translate(translated_text)
        return back_translated_text
    except Exception as e:
        print(f"번역 오류 발생: {e}")
        return text

def process_jsonl(INPUT_FILE, OUTPUT_FILE, SOURCE_LANGUAGE='en', TARGET_LANGUAGE='ko'):
    """
    JSONL 파일을 읽고, 각 라인을 백트랜슬레이션하여 새 파일에 저장합니다.
    """
    # 원본 파일의 총 라인 수를 세어 진행률 표시에 사용
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f)

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:

        # tqdm을 사용하여 진행 상황 시각화
        for line in tqdm(infile, total=total_lines, desc="데이터 증강 중"):
            original_data = json.loads(line)
            augmented_data = json.loads(line) # 원본 구조 복사

            # 'conversations' 필드 번역
            for turn in augmented_data.get('conversations', []):
                original_content = turn.get('content')
                if original_content:
                    turn['content'] = back_translate(original_content, source=SOURCE_LANGUAGE, target=TARGET_LANGUAGE)
            
            # 'system' 필드 번역
            original_system = augmented_data.get('system')
            if original_system:
                augmented_data['system'] = back_translate(original_system, source=SOURCE_LANGUAGE, target=TARGET_LANGUAGE)

            # 증강된 데이터를 새 파일에 JSON 형식으로 저장
            outfile.write(json.dumps(augmented_data, ensure_ascii=False) + '\n')

    print(f"\n✅ 데이터 증강 완료! 결과가 '{OUTPUT_FILE}' 파일에 저장되었습니다.")


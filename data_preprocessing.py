import json
import os

def update_system_field(input_filepath: str, output_filepath: str, new_system_message: str):
    """
    각 레코드의 'system' 필드를 업데이트

    Args:
        input_filepath (str): 원본 JSON Lines 파일 경로.
        output_filepath (str): 수정된 데이터를 저장할 새로운 JSON Lines 파일 경로.
        new_system_message (str): 'system' 필드에 설정할 새로운 메시지.
    """
    updated_data = []

    if not os.path.exists(input_filepath):
        print(f"오류: 입력 파일 '{input_filepath}'을(를) 찾을 수 없습니다.")
        return

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 'system' 필드 업데이트
                    data["system"] = new_system_message
                    
                    updated_data.append(json.dumps(data, ensure_ascii=False))
                except json.JSONDecodeError:
                    print(f"경고: {input_filepath} 파일의 {line_num}번째 줄에서 JSON 파싱 오류가 발생했습니다. 이 줄은 건너뜁니다.")
                except Exception as e:
                    print(f"경고: {input_filepath} 파일의 {line_num}번째 줄 처리 중 알 수 없는 오류 발생: {e}. 이 줄은 건너뜝니다.")

    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {e}")
        return

    try:
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for item in updated_data:
                outfile.write(item + '\n')
        print(f"'{input_filepath}' 파일의 'system' 필드가 성공적으로 업데이트되어 '{output_filepath}'에 저장되었습니다.")
        print(f"새로운 system 메시지: '{new_system_message}'")
    except Exception as e:
        print(f"파일 쓰기 중 오류 발생: {e}")

if __name__ == "__main__":
    # 1. 원본 데이터셋 파일 경로 (실제 파일 경로로 변경해주세요)
    # 예시: 'data/your_original_dataset.jsonl'
    original_dataset_path = 'data/task1_dataset.jsonl'
    
    # 2. 수정된 데이터를 저장할 새로운 파일 경로
    # 예시: 'data/ai_lover_dataset.jsonl'
    new_dataset_path = 'data/processed_task1_dataset.jsonl'

    # 3. 새로운 system 메시지
    new_system_prompt = "You are a highly skilled AI Crypto Investment Analyst. Your core mission is to assist users by providing data-driven insights, identifying potential investment opportunities, and evaluating market trends within the cryptocurrency domain. Utilize your financial tools diligently to analyze token holdings, wallet activities, and market data. Always ensure your responses are precise, concise, and focused on verifiable data. You prioritize informing intelligent financial decisions, not directly executing trades or providing financial advice."

    # 예시 데이터 파일 생성 (실제 파일이 없다면 이 부분을 실행하여 테스트용 파일을 만듭니다)
    if not os.path.exists(original_dataset_path):
        sample_data = [
            {"conversations": [{"role": "user", "content": "Find wallets..."}, {"role": "assistant", "content": "The following wallets..."}], "tools": [], "system": "You are a helpful assistant."},
            {"conversations": [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hi there!"}], "tools": [], "system": "You are a helpful assistant."}
        ]
        with open(original_dataset_path, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"테스트를 위한 샘플 데이터 파일 '{original_dataset_path}'이(가) 생성되었습니다.")

    # 함수 호출
    update_system_field(original_dataset_path, new_dataset_path, new_system_prompt)

    # 수정된 파일 내용 확인 (선택 사항)
    print("\n--- 수정된 파일 내용 (처음 2줄) ---")
    try:
        with open(new_dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 2: break
                print(line.strip())
    except FileNotFoundError:
        print("수정된 파일을 찾을 수 없습니다.")
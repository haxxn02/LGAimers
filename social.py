import os
import json
import time
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv # [추가된 부분]

# 1. .env 파일 로드
load_dotenv()

# 2. 환경변수에서 토큰 가져오기
FRIENDLI_TOKEN = os.getenv("FRIENDLI_TOKEN")

# [안전장치] 토큰이 잘 불러와졌는지 확인
if not FRIENDLI_TOKEN:
    print("❌ 오류: .env 파일을 찾을 수 없거나 토큰이 없습니다.")
    print("test.py와 같은 폴더에 .env 파일이 있는지 확인해주세요.")
    exit()

client = OpenAI(
    api_key=FRIENDLI_TOKEN,
    base_url="https://api.friendli.ai/serverless/v1"
)

# 모델 ID
TEACHER_MODEL_ID = "LGAI-EXAONE/K-EXAONE-236B-A23B"

# 질문 리스트 (최소 500개 목표!)
seed_prompts = [
    "대한민국의 AI 산업 발전 방향에 대해 논리적으로 설명해줘.",
    "Explain the key differences between quantization and pruning in LLMs.",
    "사과, 바나나, 딸기의 공통점과 차이점을 비교 분석해줘.",
    "파이썬을 사용하여 퀵 정렬(Quick Sort) 알고리즘을 구현해줘.",
    "Write a C++ code to implement a binary search tree insertion.",
    "Pytorch로 간단한 CNN 모델을 정의하는 코드를 작성해.",
    "미분과 적분의 관계를 고등학생이 이해하기 쉽게 설명해줘.",
    "What is the theory of relativity?",
    "셰익스피어의 햄릿 줄거리를 3문장으로 요약해줘.",
    "AI 윤리에 대한 짧은 에세이를 작성해줘."
]

output_file = "exaone_social_dataset.jsonl"
print(f"🚀 데이터 생성을 시작합니다... (모델: {TEACHER_MODEL_ID})")

successful_count = 0
with open(output_file, "a", encoding="utf-8") as f:
    for prompt in tqdm(seed_prompts):
        
        for attempt in range(5): 
            try:
                response = client.chat.completions.create(
                    model=TEACHER_MODEL_ID,
                    messages=[
                        {"role": "system", "content": "You are EXAONE, a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,
                    # temperature=0.7 (삭제됨)
                )
                
                answer = response.choices[0].message.content
                
                data_point = {
                    "instruction": prompt,
                    "output": answer
                }
                f.write(json.dumps(data_point, ensure_ascii=False) + "\n")
                f.flush()
                successful_count += 1
                
                time.sleep(2) 
                break 
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Rate limit" in error_msg:
                    print(f"\n⏳ 너무 빨라요! 10초 대기 중... (시도 {attempt+1}/5)")
                    time.sleep(10)
                elif "422" in error_msg:
                    print(f"\n❌ 설정 오류: {e}")
                    break
                else:
                    print(f"\n❌ 알 수 없는 오류: {e}")
                    time.sleep(5)

print(f"\n✅ 완료! 총 {successful_count}개의 데이터가 '{output_file}'에 저장되었습니다.")
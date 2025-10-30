# BigQuery 연동 AI Agent 개발  

## 📌 프로젝트 개요  
본 프로젝트는 **웹사이트 및 GA4 데이터를 BigQuery에 연동**하고, 이를 기반으로 **LLM 기반 AI 에이전트**를 구축하여 데이터 분석 및 자동화된 인사이트 도출을 목표로 합니다.  

## 🏗️ 시스템 아키텍처  
1. **데이터 수집 (Website & GA4)**  
   - 사용자 행동 및 웹사이트 로그 수집  
   - GA4 이벤트 데이터 적재  

2. **데이터 저장 (BigQuery 연동)**  
   - Google BigQuery와 연동하여 대규모 데이터 처리  
   - 효율적인 쿼리 및 분석 환경 제공  

3. **AI 두뇌 구축 (LLM Agent)**  
   - LLM 기반 에이전트를 통해 자연어 질의 처리  
   - 데이터 기반 의사결정 지원 및 자동 리포팅 기능 제공  

---

## ✨ 핵심 기능 (Key Features)  
- **데이터 기반 자동 질의 응답**: 사용자가 자연어로 질문하면, 에이전트가 BigQuery 데이터를 조회하여 답변 생성  
- **대시보드 & 리포트 생성**: 주요 KPI 및 분석 결과를 시각화하여 제공  
- **예측 및 추천 기능**: 과거 데이터를 기반으로 향후 트렌드 및 사용자 행동 예측  

---

## 💻 사용 예시 (Usage Example)  

### 1️⃣ Python을 통한 BigQuery 쿼리 실행
```python
from google.cloud import bigquery

# BigQuery 클라이언트 생성
client = bigquery.Client()

# 예시 쿼리: 지난 7일간 방문자 수
query = """
SELECT
  DATE(event_timestamp) as date,
  COUNT(distinct user_pseudo_id) as users
FROM `project.dataset.analytics`
WHERE event_name = 'page_view'
  AND DATE(event_timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) AND CURRENT_DATE()
GROUP BY date
ORDER BY date
"""

results = client.query(query).to_dataframe()
print(results)
```

### 2️⃣ AI Agent와 연동하여 자연어 질의
```
from openai import OpenAI
import pandas as pd

# BigQuery 결과를 가져왔다고 가정
df = results  

# LLM에 전달할 사용자 질문
user_question = "지난 일주일 동안 일일 평균 방문자는 몇 명이야?"

# 프롬프트 생성
prompt = f"""
다음은 BigQuery에서 추출한 웹사이트 방문자 수 데이터야:
{df.to_string(index=False)}

사용자 질문: {user_question}
"""

# LLM 응답
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "너는 데이터 분석 도우미야."},
              {"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### 출력 예시
```
지난 7일간의 일일 평균 방문자는 약 1,245명입니다.
```




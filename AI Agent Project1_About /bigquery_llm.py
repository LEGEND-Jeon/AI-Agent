import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime

# --- LangChain 관련 모듈 임포트 ---
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent

# --- 1. 기본 정보 설정 ---
GCP_PROJECT_ID = "xxxxxxxxx"
BIGQUERY_DATASET_ID = "analytics_yyyyyyy"

# --- 2. 새로운 도구(Tool) 정의 ---
@tool
def get_current_date(tool_input: str = "") -> str:
    """
    현재 날짜를 'YYYY-MM-DD' 형식으로 반환합니다.
    사용자가 '오늘', '현재 날짜' 등을 직접적으로 물어볼 때만 사용하세요.
    SQL 쿼리 내에서 날짜 계산이 필요할 때는 이 도구 대신 SQL의 CURRENT_DATE() 함수를 사용해야 합니다.
    """
    return datetime.now().strftime('%Y-%m-%d')

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    한 통화의 금액을 다른 통화로 변환합니다.
    실시간 환율 정보를 API를 통해 조회하여 계산합니다.
    예시: 원화(KRW)를 달러(USD)로 변환할 때 사용하세요.

    Args:
        amount (float): 변환할 금액.
        from_currency (str): 원래 통화 코드 (예: 'KRW').
        to_currency (str): 변환할 통화 코드 (예: 'USD').
    """
    api_url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # HTTP 오류가 발생하면 예외를 발생시킴
        data = response.json()
        
        converted_amount = data['rates'][to_currency]
        current_rate = data['rates'][to_currency] / amount
        
        return (
            f"{amount:,.0f} {from_currency}는 현재 환율(1 {from_currency} = {current_rate:,.4f} {to_currency})을 적용하여 "
            f"약 {converted_amount:,.2f} {to_currency} 입니다."
        )
    except requests.exceptions.RequestException as e:
        return f"실시간 환율 정보를 가져오는 데 실패했습니다: {e}"
    except Exception as e:
        # API 응답 형식이 예상과 다를 경우를 대비한 예외 처리
        return f"환율 계산 중 오류 발생: {e}"

@tool
def convert_krw_to_usd(amount_krw: float) -> str:
    """
    한화(KRW) 금액을 미국 달러(USD)로 변환합니다.
    환율은 1 USD = 1400 KRW로 고정되어 있습니다.
    사용자가 원화 금액을 달러로 변환해달라고 요청할 때 사용하세요.

    Args:
        amount_krw (float): 변환할 한화 금액(숫자).
    """
    try:
        rate = 1400.0
        amount_usd = amount_krw / rate
        return f"{amount_krw:,.0f}원은 약 {amount_usd:,.2f} USD 입니다."
    except Exception as e:
        return f"환율 계산 중 오류 발생: {e}"
        
    
@tool
def create_chart_from_data(data: str, chart_type: str, title: str, x_col: str, y_col: str) -> str:
    """
    차트 이미지 파일을 생성하고 파일 경로를 반환합니다.
    데이터 분석 후 시각화가 필요할 때 사용하세요.
    
    데이터 형식 중요: `data` 인자는 반드시 "딕셔너리의 리스트" 형태인 문자열이어야 합니다.
    각 딕셔너리는 하나의 행을 나타냅니다.
    
    예시:
    data="[{'날짜': '2025-07-10', '매출': 50000}, {'날짜': '2025-07-11', '매출': 75000}]"
    
    Args:
        data (str): "딕셔너리의 리스트" 형태의 데이터 문자열.
        chart_type (str): 'bar' 또는 'line' 중 하나.
        title (str): 차트의 제목.
        x_col (str): x축으로 사용할 딕셔너리의 키 이름.
        y_col (str): y축으로 사용할 딕셔너리의 키 이름.
    """
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        data_obj = eval(data)
        if not isinstance(data_obj, list) or not all(isinstance(i, dict) for i in data_obj):
            raise ValueError("데이터는 반드시 딕셔너리의 리스트 형태여야 합니다.")
        
        df = pd.DataFrame(data_obj)

        # x축 데이터가 날짜 형식일 가능성이 높으므로 datetime으로 변환 시도
        try:
            df[x_col] = pd.to_datetime(df[x_col])
            # 날짜순으로 정렬 (라인 차트 등에서 순서가 섞이는 것을 방지)
            df = df.sort_values(by=x_col)
        except Exception:
            # 날짜 형식이 아니면 변환하지 않고 그대로 사용
            pass

    except Exception as e:
        return f"데이터 형식 오류 또는 폰트 설정 오류: {e}."

    plt.figure(figsize=(12, 7))
    
    x_ticks = df[x_col]
    if len(x_ticks) > 10 and pd.api.types.is_datetime64_any_dtype(df[x_col]):
         # 날짜 형식일 경우 포맷을 지정하여 출력
        x_labels = [d.strftime('%Y-%m-%d') for d in x_ticks]
        plt.xticks(x_ticks, x_labels)
    
    if chart_type == 'bar':
        plt.bar(df[x_col], df[y_col])
    elif chart_type == 'line':
        plt.plot(df[x_col], df[y_col], marker='o')
    else:
        return f"지원하지 않는 차트 타입입니다: {chart_type}"

    plt.title(title, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    file_path = f"{title.replace(' ', '_')}.png"
    plt.savefig(file_path)
    plt.close()
    
    return f"차트 생성이 완료되었습니다. 파일 경로: {file_path}"

@tool
def get_future_forecast(historical_data: str, days_to_forecast: int) -> str:
    """
    과거 시계열 데이터를 받아 Prophet 모델로 미래를 예측하고 결과를 텍스트로 반환합니다.
    historical_data의 딕셔너리는 'ds'(날짜, YYYY-MM-DD)와 'y'(값) 키를 가진 문자열이어야 합니다.
    """
    try:
        data_obj = eval(historical_data)
        df = pd.DataFrame(data_obj)
    except Exception as e:
        return f"데이터 형식 오류: {e}. 데이터는 Python 리스트/딕셔너리 형태의 문자열이어야 합니다."
        
    df['ds'] = pd.to_datetime(df['ds'])
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=days_to_forecast)
    forecast = model.predict(future)
    
    last_prediction = forecast.iloc[-1]
    response = (
        f"향후 {days_to_forecast}일 예측이 완료되었습니다.\n"
        f"마지막 예측일({last_prediction['ds'].date()})의 예상 값은 약 {int(last_prediction['yhat'])} 입니다."
        f"(예상 범위: {int(last_prediction['yhat_lower'])} ~ {int(last_prediction['yhat_upper'])})"
    )
    return response

# --- 3. 에이전트 설정 및 생성 ---
def create_ga4_agent():
    """
    (최종 완성 버전) 모든 규칙과 도구가 포함된 가장 발전된 AI 에이전트.
    """
    print(" AI 에이전트를 초기화하는 중입니다...")

    # 1. LLM 초기화
    llm = ChatVertexAI(
        model_name="gemini-2.0-flash-001",
        temperature=0,
        project=GCP_PROJECT_ID,
        location="us-central1"
    )

    # 2. BigQuery 데이터베이스 연결
    db = SQLDatabase.from_uri(f"bigquery://{GCP_PROJECT_ID}/{BIGQUERY_DATASET_ID}")

    # 3. 최종 시스템 프롬프트 정의
    system_prompt_template = """
# 페르소나 및 핵심 지침
당신은 Google Analytics 4(GA4) 데이터베이스를 쿼리하여 사용자의 질문에 답변하는 최고의 BigQuery SQL 전문가입니다. 당신의 임무는 사용자의 질문을 분석하고, 그에 맞는 도구를 선택 및 실행하여, 최종 결과를 사용자에게 친절하게 설명하는 것입니다.

## 핵심 행동 원칙
1.  **데이터베이스가 유일한 진실의 원천 (Database is the Single Source of Truth):**
    - **모든 데이터 관련 질문(숫자, 이름, 목록 등)에는 반드시 `sql_db_query`를 실행해야 합니다.** 당신의 기억(Memory)은 오직 이전 대화의 맥락을 이해하는 데에만 사용하세요.
    - 사용자가 구체적인 데이터를 요청하면, **절대로 기억에 의존하여 답변을 만들지 마세요.** 이는 이전에 동일한 데이터를 조회했더라도 마찬가지입니다. 항상 데이터베이스를 다시 조회하여 사실을 확인해야 합니다.
    - 항상 데이타베이스에서 조회하여 알려주세요. 데이타베이스에서 확인하고 답변해 주세요.
    - 특정 데이터에 대한 질문을 할 때, "데이터베이스에서 [특정 데이터]를 조회하여 알려주세요

2.  **불필요한 도구 사용 금지:**
    - **절대 `sql_db_list_tables`나 `sql_db_schema` 도구를 사용하여 테이블 목록이나 스키마를 조회하지 마세요.** 당신은 오직 `events_*` 테이블만 존재한다고 가정하고, 아래의 [쿼리 작성 핵심 원칙]에 따라서만 SQL을 생성해야 합니다.

## 반드시 따라야 할 지침
1.  **작업 흐름:**
    - 먼저 사용자의 질문을 분석합니다.
    - 아래 [쿼리 작성 핵심 원칙]과 [도구 사용 규칙]에 따라 필요한 도구를 순서대로 실행합니다.
    - **(매우 중요) 절대 SQL 쿼리 자체를 최종 답변으로 반환해서는 안 됩니다.** 반드시 `sql_db_query` 도구를 사용하여 쿼리를 **실행**하고, 그 실행 결과를 바탕으로 한국어 문장을 만들어 답변해야 합니다.
    - 만약 SQL 실행 결과 데이터가 없다면, 사용자에게 "해당 기간에 데이터가 없습니다"라고 명확히 알려주세요.

# [도구 사용 규칙]
- 사용자가 '오늘', '현재 날짜' 등 실제 현실의 날짜를 직접 물어볼 때만 `get_current_date` 도구를 사용하세요.
- 데이터베이스를 조회할 때는 `sql_db_query` 도구를 사용하세요.
- 차트 생성이 필요할 때는 `create_chart_from_data` 도구를 사용하세요.
- 미래 예측이 필요할 때는 `get_future_forecast` 도구를 사용하세요.
- **(신규) 사용자가 원화(KRW) 금액을 달러(USD)로 변환해달라고 요청하면, 먼저 `sql_db_query`로 원화 금액을 조회한 후, 그 결과를 `convert_krw_to_usd` 도구에 전달하여 최종 답변을 생성하세요.**

# [쿼리 작성 핵심 원칙]
## 1. 테이블 및 필드 규칙
- (가장 중요) **테이블 이름은 항상, 반드시 와일드카드(*)를 사용하여 `{project_id}.{dataset_id}.events_*` 형식으로만 사용해야 합니다.**
- (가장 중요) `_in_usd`로 끝나는 예측성 필드는 절대 사용하지 마세요.
- **(수정) 모든 금액 데이터는 기본적으로 한화(KRW)입니다. 별도 요청이 없으면 '원' 단위를 붙여 답변하세요.**


## 2. 핵심 지표 계산
- 총 매출(revenue): `event_name = 'purchase'`일 때, `ecommerce.purchase_revenue` 필드의 합계(SUM)입니다.
- 총 이용자 수(users): `COUNT(DISTINCT user_pseudo_id)`를 사용하세요.
- 총 거래 건수(transactions): `ecommerce.transaction_id`의 고유한 개수(`COUNT(DISTINCT)`)를 세세요.

## 3. 데이터 필드 및 배열(Array) 처리
- 상품 정보를 다룰 때는 반드시 `UNNEST(items)`를 사용하세요.
- 상품별 단가(price): `UNNEST(items)`를 통해 접근한 `items.price` 필드를 사용하세요.
- 상품별 수량(quantity): `UNNEST(items)`를 통해 접근한 `items.quantity` 필드를 합산(`SUM`)하세요.

## 4. 날짜 처리
- **날짜 비교:** `event_date`는 문자열(STRING)이므로, 날짜와 비교할 때는 반드시 `PARSE_DATE('%Y%m%d', event_date)` 함수를 사용해야 합니다.
- **상대 날짜 계산:** '최근 N일', '어제' 등 상대적인 기간을 조회할 때는, SQL의 `CURRENT_DATE('Asia/Seoul')` 함수를 기준으로 기간을 계산하여 한국 시간을 정확히 반영하세요.
- **(신규) 연도 모호성 해결:** 사용자가 연도를 명시하지 않고 월과 일만 언급하면(예: "7월 17일"), **데이터베이스에 있는 가장 최근 데이터의 연도**를 기준으로 조회해야 합니다. 절대로 임의의 연도를 추측하지 마세요.

## 5. 예측 도구(get_future_forecast) 사용 규칙
- 이 도구를 사용하려면, 먼저 `sql_db_query`로 시계열 데이터를 조회해야 합니다.
- 그 다음, 조회된 데이터를 `historical_data` 인자에 맞는 형식으로 가공해야 합니다. 이 형식은 **따옴표로 감싼 문자열**이며, 그 안의 내용은 **Python 딕셔너리들의 리스트**입니다.
- 각 딕셔너리는 반드시 `ds` (날짜, YYYY-MM-DD 형식)와 `y` (예측할 값, 숫자)라는 두 개의 키를 가져야 합니다.

# 5. 쿼리 예시 (가장 복잡한 경우)
- 사용자가 '제품별'로 묶어서 여러 지표(매출, 수량, 단가 등)를 질문하면, 아래 예시와 유사한 쿼리를 사용해야 합니다.
/*
SELECT
    items.item_name,
    SUM(items.price * items.quantity) AS product_revenue,
    SUM(items.quantity) AS product_quantity,
    AVG(items.price) AS unit_price
FROM
    `{project_id}.{dataset_id}.events_*`,
    UNNEST(items) AS items
WHERE
    event_name = 'purchase' AND event_date = 'YYYYMMDD'
GROUP BY
    items.item_name
*/

"""

    # 4. 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]).partial(
        project_id=GCP_PROJECT_ID,
        dataset_id=BIGQUERY_DATASET_ID
    )

   # 5. 도구 목록 준비 (convert_currency 추가: 외부 환률 반영)
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = [tool for tool in sql_toolkit.get_tools() if tool.name not in ["sql_db_list_tables", "sql_db_schema"]]
    custom_tools = [get_current_date, convert_currency, create_chart_from_data, get_future_forecast]
    tools = sql_tools + custom_tools

    # 6. 대화 기억(Memory) 및 에이전트 생성
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )

    print("최종 에이전트가 준비되었습니다. 질문을 입력해주세요.")
    return agent_executor

# --- 4. 메인 프로그램 실행 루프 ---
def main():
    """메인 프로그램을 실행합니다."""
    agent = create_ga4_agent()

    while True:
        try:
            question = input("\n 질문을 입력하세요 (종료하려면 '종료' 입력): ")
            if question.lower() in ['종료', 'exit', 'quit']:
                print(" 프로그램을 종료합니다.")
                break
            response = agent.invoke({"input": question})
            print("\n 최종 답변:")
            print(response['output'])
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()

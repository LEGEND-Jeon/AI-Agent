# 📢 CUK Notice Translation Agent

> **Catholic University of Korea Announcement → English Document Automation**  
> 대학 공지사항을 자동으로 수집하고, 번역 규칙을 적용하여 영어 문서(DOCX)로 정리하는 AI Agent 프로젝트

---

## 📝 Background

학교 인턴십 동안 저는 **리스트에 있는 수많은 공지사항을 영어로 번역하는 업무**를 맡게 되었습니다.  
하지만 이 과정은 단순하지 않았습니다:

- 공지 제목을 하나씩 검색해서 찾아야 했고  
- 번역 규칙(날짜 형식, 목록 유지, 불필요 UI 제거 등)을 하나하나 적용해야 했으며  
- 번역된 결과를 정리해 문서화하는 데에도 시간이 걸렸습니다  

반복적이고 시간이 많이 소요되는 업무였기 때문에,  
“이 과정을 **AI Agent**를 통해 자동화할 수 있지 않을까?” 하는 생각에서 프로젝트를 시작했습니다.  

---

## 🚀 What this project does

이 프로젝트는 가톨릭대학교 [공지사항 페이지](https://www.catholic.ac.kr/ko/campuslife/notice.do)에서 원하는 항목을 자동으로 검색/수집하고,  
AI 번역 모델(Gemini, GPT 등)을 이용해 영어로 번역하여 Word 문서(DOCX)로 정리합니다.

### 주요 기능
- 🔍 **자동 링크 수집**: Playwright를 이용해 동적으로 렌더링되는 공지 페이지에서 `articleNo`를 추출하고 정규화된 URL 생성  
- 📄 **본문 추출**: DOM 구조(`.fr-view`, `.view-cont` 등)에 맞춰 공지 본문을 줄글 형태로 파싱  
- 🌐 **AI 번역**: Gemini API(또는 OpenAI API)를 통해 지정된 번역 규칙을 적용해 자연스러운 영어 공지문 생성  
- 🗂️ **문서화**: 번역 결과를 하나의 DOCX 파일(`CUK_Announcements_EN.docx`)로 정리  
- ⚡ **반복 자동화**: 여러 공지를 리스트로 넣으면 순차적으로 처리  

---

## 🛠️ Tech Stack

- **Python 3.10+**
- [Playwright](https://playwright.dev/python/) – 동적 웹 페이지 크롤링
- [python-docx](https://python-docx.readthedocs.io/) – DOCX 문서 생성
- [google-generativeai](https://ai.google.dev/) – Gemini 번역 API
- [dotenv](https://pypi.org/project/python-dotenv/) – 환경 변수 관리

---

## 📂 Project Structure
.
├── searchURL.py             # 제목을 검색해 articleNo 포함 정규화 URL 추출
├── cuk_notice_agent.py      # URL에서 본문 추출 → 번역 → DOCX 생성 메인 파이프라인
├── outputs/
│   ├── notice_links.csv     # searchURL.py 실행 결과 (제목-URL 매핑)
│   ├── notice_results.json  # 중간 크롤링/번역 결과 저장
│   └── CUK_Announcements_EN.docx  # 최종 산출물 (영문 공지 모음)
└── README.md

---

## ⚙️ How to Run

1. Install dependencies:
   ```
   bash
   pip install playwright python-docx python-dotenv google-generativeai
   playwright install
  
2.	Add API key in .env:
   ```
  GEMINI_API_KEY=your_api_key_here
  GEMINI_MODEL=gemini-1.5-pro
  ```
3. (Optional) Run searchURL.py first to create outputs/notice_links.csv:
   ```
   python searchURL.py
    ```
4. 	Run main agent:
	 ```
    python cuk_notice_agent.py
   
5. outputs/CUK_Announcements_EN.docx

## 🔮 Future Plans
- 공지사항 학교 홈페이지에 업로드까지 자동화
- 다국어 지원
- UI/UX 제작

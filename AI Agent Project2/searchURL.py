# filename: searchURL.py
# 실행 전:
#   pip install playwright rapidfuzz
#   playwright install
# 실행:
#   python3 searchURL.py
from __future__ import annotations
import re, json, csv, time, sys
from dataclasses import dataclass
from typing import Optional, List, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from pathlib import Path

from rapidfuzz import fuzz, process

# ── 출력 폴더 고정 ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Playwright 준비 ─────────────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except Exception:
    print(" playwright 미설치. 설치 후 재실행:\n"
          "   pip install playwright rapidfuzz && playwright install")
    sys.exit(1)

# ── 설정 ────────────────────────────────────────────────────────
NOTICE_LIST_URL = "https://www.catholic.ac.kr/ko/campuslife/notice.do"
HEADLESS = True
NETWORK_IDLE_WAIT = 1000
BETWEEN_QUERIES_SLEEP = 0.8
FUZZY_THRESHOLD = 90          # 제목 유사도 임계값(0~100)


TARGET_TITLES: List[str] = [
    "[건축팀] 김수환관 승강기 4호기(전망용 승강기 중 좌측) 제어시스템 교체공사 안내",
    "[대외협력팀] 홍보 콘텐츠 제작 대외협력팀 소속 기관동아리 <CUK프렌즈> 18기 모집 공고" ]

# ── 데이터 구조 ────────────────────────────────────────────────
@dataclass
class Result:
    title: str
    raw_url: Optional[str]
    normalized_url: Optional[str]
    status: str
    note: str = ""

# ── 유틸 ────────────────────────────────────────────────────────
def normalize_cuk_notice_url(url: str) -> str:
    p = urlparse(url)
    qs = parse_qs(p.query)
    article_no = qs.get("articleNo", [None])[0]
    if not article_no:
        return url
    new_qs = {"mode": "view", "articleNo": article_no}
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(new_qs), ""))

def extract_article_no(url: str) -> Optional[str]:
    m = re.search(r"(?:[?&])articleNo=(\d+)", url)
    return m.group(1) if m else None

# ── DOM 헬퍼 ────────────────────────────────────────────────────
def find_search_input(page):
    candidates = [
        "input[name='srSearchVal']",
        "input#srSearchVal",
        "input[type='search']",
        "input[placeholder*='검색']",
        "input[title*='검색']",
        "input[name*='search']",
    ]
    for sel in candidates:
        el = page.query_selector(sel)
        if el: return el
    els = page.query_selector_all("input")
    return els[0] if els else None

def collect_result_links(page) -> List[Tuple[str, object]]:
    """
    검색 결과 리스트에서 (보이는제목, link_element) 목록을 수집
    """
    selectors = [
        "table tbody tr td a",
        "ul li a.title",
        ".board-list a",
        "a[href*='notice.do?mode=view']",
        "a[href*='articleNo=']",
    ]
    for sel in selectors:
        try:
            page.wait_for_selector(sel, timeout=5000)
            links = page.query_selector_all(sel)
            pairs = []
            for lk in links:
                txt = (lk.inner_text() or "").strip()
                if not txt:
                    # a태그 안에 span만 있는 형태 보완
                    txt = (lk.text_content() or "").strip()
                if txt:
                    pairs.append((txt, lk))
            if pairs:
                return pairs
        except PWTimeout:
            continue
    return []

def click_element_and_get_url(page, element) -> Optional[str]:
    with page.expect_navigation(timeout=15000):
        element.click()
    return page.url

def pick_by_title(pairs: List[Tuple[str, object]], target_title: str) -> Optional[object]:
    # 1) 정확 일치 우선
    for txt, el in pairs:
        if txt == target_title:
            return el
    # 2) 유사도 매칭
    choices = [txt for txt, _ in pairs]
    best = process.extractOne(target_title, choices, scorer=fuzz.token_set_ratio)
    if best and best[1] >= FUZZY_THRESHOLD:
        match_text = best[0]
        for txt, el in pairs:
            if txt == match_text:
                return el
    return None

# ── 메인 로직 ───────────────────────────────────────────────────
def search_title_and_get_url(page, title: str) -> Result:
    page.goto(NOTICE_LIST_URL, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(400)

    si = find_search_input(page)
    if not si:
        return Result(title, None, None, "fail", "검색창을 찾지 못함")

    # 검색 실행
    si.fill("")
    si.type(title)
    si.press("Enter")
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except PWTimeout:
        pass
    page.wait_for_timeout(NETWORK_IDLE_WAIT)

    # 결과 수집
    pairs = collect_result_links(page)
    if not pairs:
        return Result(title, None, None, "not_found", "검색 결과가 비어 있음")

    # 제목 일치(정확→유사)로 대상 링크 선택
    el = pick_by_title(pairs, title)

    # 그래도 못 찾으면 6번째 폴백
    if not el:
        el = pairs[5][1] if len(pairs) >= 6 else pairs[-1][1]
        fallback_note = "정확/유사 매칭 실패 → 6번째(또는 마지막) 결과로 폴백"
    else:
        fallback_note = ""

    detail_url = click_element_and_get_url(page, el)
    if not detail_url:
        return Result(title, None, None, "not_found", "상세 페이지 이동 실패")

    norm = normalize_cuk_notice_url(detail_url)
    if extract_article_no(norm):
        return Result(title, detail_url, norm, "ok", fallback_note)
    else:
        return Result(title, detail_url, norm, "warn", "articleNo 미검출" + (f" | {fallback_note}" if fallback_note else ""))

def main():
    results: List[Result] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page()
        for t in TARGET_TITLES:
            try:
                r = search_title_and_get_url(page, t)
            except Exception as e:
                r = Result(t, None, None, "error", f"{type(e).__name__}: {e}")
            results.append(r)
            time.sleep(BETWEEN_QUERIES_SLEEP)
        browser.close()

    json_path = OUTPUT_DIR / "notice_links.json"
    csv_path  = OUTPUT_DIR / "notice_links.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["title", "status", "normalized_url", "raw_url", "note"])
        for r in results:
            w.writerow([r.title, r.status, r.normalized_url or "", r.raw_url or "", r.note])

    print("\n== 결과 요약 ==")
    for r in results:
        show = r.normalized_url or r.raw_url or "N/A"
        print(f"- [{r.status}] {r.title}\n    -> {show} {'(' + r.note + ')' if r.note else ''}")
    print(f"\n 저장 위치:\n  {json_path}\n  {csv_path}")

if __name__ == "__main__":
    main()

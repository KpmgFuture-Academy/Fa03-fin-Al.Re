# 라이브러리 불러오기
import pandas as pd
from openai import OpenAI
import hashlib
from datetime import datetime, timedelta
import numpy as np
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer
import streamlit as st

# 페이지 설정
st.set_page_config(layout="wide")
 
# 모듈화된 함수 불러오기
from cart import (
    add_to_cart,
    parse_recipe,
    recommend_recipes,
    get_remaining_cart,
    recipe_serving_price
)

from data import (
    get_mysql_connection,
    load_recipes,
    load_product,
    load_preference,
    load_similarity,
    load_total_revenues
)
from log import log_event
from login import authenticate
from market import search_products, search_similar_recipes_with_vectordb
from preference import ( 
    generate_similarity_table, 
    generate_preference_table,
)
from chatbot import classify_user_intent, chatbot_recommendation, choramadb_search, gpt_select_recipe

# streamlit 함수
def render_product_cards(title: str, products_df: pd.DataFrame, recipe_key: str):

    """
    상품들을 그룹별로 카드 UI 형태로 렌더링하는 함수.

    Args:
        title: 그룹 타이틀 (예: '채소류', '고기류' 등)
        products_df: 해당 그룹에 속하는 상품 데이터프레임
        recipe_key: 레시피별 고유 식별자 (Streamlit key 중복 방지용)
    """

    # 그룹 타이틀 출력
    st.markdown(f"#### 🔍 {title}")

    # 최대 4열까지 컬럼 생성
    cols = st.columns(min(4, len(products_df)))

    # 각 상품 정보를 카드 형태로 렌더링
    for i, (_, row) in enumerate(products_df.iterrows()):
        with cols[i % 4]:
            # 상품 이미지
            st.image(row['image'], width=120)
            
            # 브랜드명 표시 (없음이면 생략)
            st.caption(row['brand'] if row['brand'] != '없음' else '')

            # 상품명, 중량/단위, 가격
            st.write(f"**{row['name']}**")
            st.write(f"{row['weight']}{row['unit']} | ₩{row['price']:,}")

            # Streamlit 컴포넌트 key 생성
            key = f"select_{recipe_key}_{row['id']}_{row['name']}"

            # 선택 여부 체크박스
            if st.checkbox("선택", key=key):
                # 선택 항목 추가
                st.session_state.selected_products.add(row['name'])  
            else:
                # 선택 해제 시 제거
                st.session_state.selected_products.discard(row['name'])  

def render_recipe_cards(recipes):

    """
    추천 레시피 목록을 카드 UI 형태로 3열로 렌더링하는 함수.

    Args:
        recipes: 추천 레시피 리스트 (각 항목은 dict 형태, 'img' 포함)
    """

    # 최대 3열로 컬럼 생성
    cols = st.columns(min(3, len(recipes)))

    for i, recipe in enumerate(recipes):
        with cols[i % 3]:
            # 레시피 이미지를 둥근 테두리와 정사각 비율로 출력 (HTML/CSS 사용)
            st.markdown(f"""
                <div style="width:180px; height:180px; margin:0 auto; overflow:hidden; border-radius:10px;">
                    <img src="{recipe['imgUrl']}" style="width:100%; height:100%; object-fit:cover;"/>
                </div>
            """, unsafe_allow_html=True)

def apply_image_proxy(url):

    """
    외부 이미지 URL을 안전하게 프록시 처리하여 Streamlit 등에서 로딩이 실패하지 않도록 보정하는 함수.

    Args:
        url: 원본 이미지 URL (str)

    Returns:
        프록시 서버를 거친 안전한 이미지 URL (str)
        (URL이 없는 경우 기본 placeholder 이미지 반환)
    """

    # URL이 없거나 비어 있을 경우 기본 placeholder 이미지로 대체
    if not url:
        return "https://via.placeholder.com/80"

    # 이미지를 프록시 서버를 통해 우회하여 로드
    # - https://images.weserv.nl 프록시 사용
    # - 이미 https://가 붙은 경우 중복을 방지하기 위해 제거하고 붙인다
    return f"https://images.weserv.nl/?url={url.replace('https://', '')}"

def safe_key(*args):

    """
    주어진 여러 인자를 조합하여 Streamlit 위젯에서 사용할 수 있는 고유한 key 문자열을 생성하는 함수.
    중복 오류(StreamlitDuplicateElementKey)를 방지하기 위해 문자열을 해시 처리함.

    Args:
        *args: 고유 키 생성을 위한 여러 요소들 (예: 레시피 ID, 상품명, 재료명 등)

    Returns:
        str: "ingredient_add_" 접두사가 붙은 MD5 해시 기반의 고유 key 문자열
    """
    
    # 전달된 인자들을 문자열로 결합 (예: "7021136_제주 무_무")
    key_raw = "_".join(map(str, args))
    
    # 문자열을 MD5 해시 처리 → 중복 방지용 유일 키 생성
    return "ingredient_add_" + hashlib.md5(key_raw.encode()).hexdigest()



def render_missing_ingredient_batch_add(selected_recipe, df_product):
    """
    선택된 레시피에서 장바구니에 없는 재료를 자동으로 검색하여,
    가장 먼저 찾은 관련 상품을 장바구니에 추가하는 함수.
    """

    # 이미 처리된 레시피인지 확인 (rerun 중복 방지)
    last_added = st.session_state.get("last_added_recipe", (None, []))
    if last_added[0] == selected_recipe.get("name"):
        return

    # 1. 레시피 파싱
    parsed = selected_recipe.get('parsedRecipe') or parse_recipe(selected_recipe.get('inputRecipe', ''))

    # 2. 이미 장바구니와 매칭된 재료
    matched = selected_recipe.get('matched', [])

    # 3. 부족한 재료 필터링
    result = [item for item in parsed if item not in matched]
    newly_added = []

    # 4. 부족한 재료별 상품 검색 및 자동 추가
    for item in result:
        ingredient = item['ingredient'] if isinstance(item, dict) else item
        results = search_products(ingredient, df_product)
        if results.empty:
            continue

        first = results.iloc[0]

        # 안전한 키 생성
        key = safe_key(first["name"], first.get("brand", ""), first.get("weight", 0))

        # 중복 상품은 건너뜀
        if key in st.session_state["cart"]:
            continue

        # 장바구니에 추가
        add_to_cart(
            st.session_state["cart"],
            first["domain"],
            first["division"],
            first["category"],
            first["name"],
            first.get("brand", ""),
            first.get("weight", 0),
            first.get("unit", ""),
            first.get("price", 0),
            first.get("image", "")
        )

        newly_added.append(first["name"])

    # 5. 레시피 정보 상태에 저장 및 rerun
    if newly_added:
        rname = selected_recipe["name"]
        st.session_state.selected_recipes.append(rname)
        st.session_state.selected_recipe_sources[rname] = newly_added

        recipe_id = selected_recipe["id"]
        if recipe_id not in st.session_state.recipe_cart:
            st.session_state.recipe_cart.append(recipe_id)

        st.session_state.show_recipe_popup = True
        st.session_state["last_added_recipe"] = (rname, newly_added)
        st.rerun()
    else:
        st.info("장바구니에 이미 있거나, 상품이 없는 재료만 있어요.")

    # 5. 추가된 재료가 있다면 상태 업데이트
    if newly_added:
        rname = selected_recipe["name"]
        st.session_state.selected_recipes.append(rname)
        st.session_state.selected_recipe_sources[rname] = newly_added

        # 레시피 ID도 recipe_cart에 추가
        recipe_id = selected_recipe["id"]
        if recipe_id not in st.session_state.recipe_cart:
            st.session_state.recipe_cart.append(recipe_id)

        # 팝업 열기
        st.session_state.show_recipe_popup = True

        # 반환값 및 rerun
        st.session_state["last_added_recipe"] = (rname, newly_added)
        st.rerun()

    else:
        # 아무것도 추가되지 않은 경우
        st.info("장바구니에 이미 있거나, 상품이 없는 재료만 있어요.")

def render_recipe_recommendation(recipes, title, key_prefix, df_product):
    """
    추천 레시피 리스트를 화면에 이미지 + 버튼 + 관련 재료 + 상품까지 렌더링하는 공통 블록.

    Args:
        recipes (list[dict]): 추천 레시피 리스트
        title (str): 추천 타이틀 (예: '취향저격 레시피')
        key_prefix (str): Streamlit 위젯 고유 키 prefix (예: 'top', 'cart', 'remain')
        df_product (pd.DataFrame): 상품 데이터프레임
    """

    st.header(title)

    if not recipes:
        st.info("현재 조건에 맞는 추천 레시피가 없어요 😢 다른 재료를 추가해보세요!")
        return

    render_recipe_cards(recipes)
    st.write("")

    empty1, col1, empty2, col2, empty3, col3 = st.columns([0.5, 4, 0.5, 4, 0.5, 4])

    cols = [col1, col2, col3]
    selected_ids = []

    for i, recipe in enumerate(recipes):
        col = cols[i % 3]  # 3열 기준 순환
        with col:
            if st.button(recipe["name"], key=safe_key(f"{key_prefix}_recipe_button", i, recipe["id"], recipe["name"])):
                st.session_state[f"{key_prefix}_selected_recipe_idx"] = i

    key_selected = f"{key_prefix}_selected_recipe_idx"
    if key_selected in st.session_state:
        selected_recipe = recipes[st.session_state[key_selected]]
        with st.expander(f"📋 {selected_recipe['name']} 레시피 재료 목록", expanded=False):

            # 해당 레시피 링크
            recipe_url = f"https://www.10000recipe.com/recipe/{selected_recipe['id']}"

            col1, col2 = st.columns([1, 7])  # 첫 번째 열은 레이블, 두 번째는 링크
            with col1:
                st.write("레시피 보기:")
            with col2:
                st.write(f"[🌐]({recipe_url})", unsafe_allow_html=True)

            # 자동 장바구니 담기
            if st.button("🛒 부족한 재료 자동 장바구니 담기", key=f"auto_add_{selected_recipe['id']}"):
                render_missing_ingredient_batch_add(selected_recipe, df_product)

            # parsedRecipe 기반 필터링
            parsed_recipe = selected_recipe["parsedRecipe"]
            matched = selected_recipe.get("matched", [])
            result = [item for item in parsed_recipe if item not in matched]

            # 중복 제거
            seen = set()
            ingredient_list = []
            for ing in result:
                if ing not in seen:
                    ingredient_list.append(ing)
                    seen.add(ing)

            # 부족한 재료별 상품 추천 및 선택 체크박스
            for ingredient in ingredient_list:
                st.markdown(f"#### ▪︎ '{ingredient}' 관련 추천 상품")
                results = search_products(ingredient, df_product)
                if results.empty:
                    st.warning(f"'{ingredient}' 관련 검색 결과가 없습니다.")
                else:
                    limited_results = results.iloc[:4]
                    cols = st.columns(4, gap="small")
                    for i, (_, r) in enumerate(limited_results.iterrows()):
                        with cols[i % 4]:
                            st.image(r["image"], use_container_width=True)
                            st.write(f"**{r['name']}**")
                            st.caption(r["brand"] if r["brand"] != "없음" else "")
                            st.write(f"{r['weight']}{r['unit']} | ₩{r['price']:,}")

                            cb_key = safe_key(f"{key_prefix}_cb", selected_recipe["id"], ingredient, r["name"], i)
                            if st.checkbox("선택", key=cb_key):
                                st.session_state.selected_products_batch.add((r["name"], ingredient))
                            else:
                                st.session_state.selected_products_batch.discard((r["name"], ingredient))

            # 선택한 상품을 한꺼번에 장바구니에 담기
            if st.session_state.get("selected_products_batch"):
                if st.button("🛒 선택한 상품 장바구니 담기"):
                    # set 복사해서 반복
                    for name, ing in list(st.session_state.selected_products_batch):
                        product_row = df_product[df_product["name"] == name]
                        if not product_row.empty:
                            r = product_row.iloc[0]

                            # 중복 방지를 위한 고유 키 생성
                            key = safe_key(r["name"], r.get("brand", ""), r.get("weight", 0))

                            # 이미 장바구니에 있으면 스킵
                            if key in st.session_state["cart"]:
                                continue

                            # 장바구니에 상품 추가
                            add_to_cart(
                                st.session_state["cart"],
                                r["domain"],
                                r["division"],
                                r["category"],
                                r["name"],
                                r.get("brand", ""),
                                r.get("weight", 0),
                                r.get("unit", ""),
                                r.get("price", 0),
                                r.get("image", "")
                            )

                    # ✅ 레시피도 함께 카트에 추가
                    recipe_id = selected_recipe["id"]
                    if recipe_id not in st.session_state.recipe_cart:
                        st.session_state.recipe_cart.append(recipe_id) 

                    # 반복 후 clear
                    st.session_state.selected_products_batch.clear()
                    st.rerun()

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

# 모델 선언
@st.cache_resource
def load_model():
    return SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device='cpu')

model = load_model()

# 세션 초기화
if "df_product" not in st.session_state:
    st.session_state["df_product"] = load_product()

if "df_recipe" not in st.session_state:
    st.session_state["df_recipe"] = load_recipes()

if "df_preference" not in st.session_state:
    st.session_state["df_preference"] = load_preference()

if "df_similarity" not in st.session_state:
    st.session_state["df_similarity"] = load_similarity()

if "df_total" not in st.session_state:
    st.session_state["df_total"] = load_total_revenues()

if "user" not in st.session_state:
    st.session_state["user"] = None

if "userNum" not in st.session_state:
    st.session_state["userNum"] = "unknown"

if "osType" not in st.session_state:
    st.session_state["osType"] = "unknown"

if "cart" not in st.session_state:
    st.session_state["cart"] = {}

if "purchased_weight" not in st.session_state:
    st.session_state["purchased_weight"] = {}

if "used_weight" not in st.session_state:
    st.session_state["used_weight"] = {}

if "selected_recipes" not in st.session_state:
    st.session_state["selected_recipes"] = []

if "selected_recipe_sources" not in st.session_state:
    st.session_state["selected_recipe_sources"] = {}

if "selected_products" not in st.session_state:
    st.session_state["selected_products"] = set()

if "recipe_cart" not in st.session_state:
    st.session_state.recipe_cart = []

if "recipe_price" not in st.session_state:
    st.session_state.recipe_price = []

# 검색 결과 캐싱 관련 세션 초기화
if "cached_recipe_query" not in st.session_state:
    st.session_state["cached_recipe_query"] = None
if "cached_recipe_results" not in st.session_state:
    st.session_state["cached_recipe_results"] = pd.DataFrame()
if "cached_product_query" not in st.session_state:
    st.session_state["cached_product_query"] = None
if "cached_product_results" not in st.session_state:
    st.session_state["cached_product_results"] = pd.DataFrame()

# 선택 상태 저장용 세션 초기화
if "selected_products_batch" not in st.session_state:
    st.session_state["selected_products_batch"] = set()

# 로그인 처리
if st.session_state["user"] is None:
    
    # 2개의 컬럼 생성
    col1, col2 = st.columns([2, 1])

    with col1:
        # 왼쪽 이미지
        st.image("local_image/login_background.png", caption="Welcome!", use_container_width=True)

    with col2:
        # 로그인 화면 제목 출력
        st.markdown(
            "<h1 style='font-size: 80px;'>AI.re</h1>",
            unsafe_allow_html=True
        )
        # 로그인 버튼 클릭 시 동작
        login_id = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")

        # 로그인 버트 클릭 시 동작
        if st.button("로그인"):

            # 인증 함수 호출
            user = authenticate(login_id, password)
            if user:
                # 로그인 성공 시 세션에 사용자 정보 저장
                st.session_state["user"] = user
                st.session_state["is_admin"] = user.get("role") == "admin"

                st.rerun()

            else:
                # 로그인 실패 시 에러 메시지 출력
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

        # 로그인 전이거나 실패한 경우 이 코드 실행 중단
        st.stop()

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

# 사이드바 부분

# 유저 정보 불러오기
user = st.session_state["user"]

with st.sidebar:
    # 로그인 정보 및 로그아웃 버튼
    st.sidebar.title("AI.Re" if not st.session_state["is_admin"] else "🔧 운영관리")
    st.sidebar.write(f"환영합니다, {st.session_state['user']['name']}님!")

    if st.sidebar.button("로그아웃"):
        st.session_state["user"] = None
        st.session_state["is_admin"] = False
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

    # 운영관리 사이드바
    if st.session_state["is_admin"]:
        page = st.sidebar.selectbox("운영관리 기능", ["Summary Board", "전략 기획", "마케팅", "공급망 관리"], key="admin_page")

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

    # 사용자 사이드바
    else:
        # 데이터 불러오기
        user = st.session_state["user"]
        cart = st.session_state["cart"]
        recipe_cart = st.session_state["recipe_cart"]
        df_recipe = st.session_state["df_recipe"]

        df_recipe['parsedRecipe'] = df_recipe['inputRecipe'].apply(parse_recipe)


        page = st.sidebar.selectbox("일반 기능", ["메인", "AIre봇", "사용자 설정", "레시피 추천 및 장바구니"], key="user_page")
        
        st.sidebar.title("🍽️ 메뉴 선택")

        # 레시피
        st.markdown("## 📜 레시피")       

        if st.session_state.recipe_cart and len(st.session_state.recipe_cart) > 0:
            # 담은 레시피 수
            recipe_cart_df = df_recipe[df_recipe["id"].isin(recipe_cart)]

            # total_price = 0

            # for _, r in recipe_cart_df.iterrows():
            #     parsed = parse_recipe(r["inputRecipe"])
            #     port_num = int(r["portNum"])
            #     total_price += recipe_serving_price(st.session_state.cart, parsed, port_num)

            # st.markdown(f"""
            # <div style='text-align: right; font-size: 20px; font-weight: 600; color: #333; margin-bottom: 4px;'>
            #     📜 담은 레시피 수: <span style='color:#d63384'>{len(recipe_cart)}개</span>
            # </div>
            # <div style='text-align: right; font-size: 20px; font-weight: 600; color: #333; margin-bottom: 4px;'>
            #     🍽️ 끼니 수: <span style='color:#0d6efd'>{len(recipe_cart)}개</span>
            # </div>
            # <div style='text-align: right; font-size: 20px; font-weight: 600; color: #333; margin-bottom: 4px;'>
            #     💰 총 가격 합계: <span style='color:#0d6efd'>{int(round(total_price)):,}원</span>
            # </div>            
            # """, unsafe_allow_html=True)
           
            if "show_recipe_popup" not in st.session_state:
                st.session_state.show_recipe_popup = False

            # 레시피 전체보기 버튼
            if st.button("🍽 전체 레시피 보기", use_container_width=True):
                st.session_state.show_recipe_popup = True

            # 3. 레시피 보기
            if st.session_state.show_recipe_popup:
                with st.expander("💌 레시피는 마음에 드시나요?", expanded=True):
                    recipe_cart_df = df_recipe[df_recipe["id"].isin(recipe_cart)]

                    for _, r in recipe_cart_df.iterrows():
                        recipe_id = r["id"]
                        recipe_name = r["name"]
                        parsed = parse_recipe(r['inputRecipe'])
                        port_num = int(r["portNum"])
                        price = recipe_serving_price(st.session_state.cart, parsed, port_num)

                        # "재료 수량" 형태로 문자열 리스트 만들기
                        ingredients_text = ", ".join(
                            f"{item['ingredient']} {item['quantity']}" if item['quantity'] else item['ingredient']
                            for item in parsed
                        )

                        # ✅ 레시피명 + 삭제 버튼을 두 컬럼에 분리해 배치
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"""
                            <div style="line-height: 1.6; margin-bottom: 18px;">
                                <p style="font-size:20px; font-weight:bold; margin: 0 0 6px 0;">{r['name']}</p>
                                <p style="font-size:16px; color:#444; margin: 0 0 6px 0;">{r['style']} / {r['category']} / {r['ingredient']}</p>
                                <p style="font-size:15px; margin: 0 0 6px 0;">[재료] {ingredients_text}</p>
                                <p style="font-size:14px; color:#555; margin: 0;">{r['time']}, {r['portNum']}인분</p>
                            </div>
                            """, unsafe_allow_html=True)
                        # <p style="font-size:18px; font-weight:bold; color:#228; text-align: right;"">1인분당 최대 가격: {int(round(price)):,}원</p>

                        with col2:
                            if st.button("❌", key=f"sidebar_remove_recipe_{recipe_id}"):
                                st.session_state.recipe_cart.remove(recipe_id)
                                if recipe_name in st.session_state.selected_recipes:
                                    st.session_state.selected_recipes.remove(recipe_name)
                                    to_remove = st.session_state.selected_recipe_sources.pop(recipe_name, [])

                                    keys_to_remove = [
                                        key for key, val in st.session_state.cart.items()
                                        if val.get("display_name") in to_remove
                                    ]
                                    for key in keys_to_remove:
                                        st.session_state.cart.pop(key, None)
                                        st.session_state.purchased_weight.pop(key, None)
                                        st.rerun()

        # 장바구니 요약 정보 출력
        st.markdown("## 🛒 장바구니")

        # 총 수량 및 금액 계산
        total_items = sum(info["qty"] if isinstance(info, dict) else info for info in cart.values())
        total_price = sum(info["qty"] * info["price"] if isinstance(info, dict) else 0 for info in cart.values())

        if total_items > 0:
            # 장바구니 정보 시각화 출력
            st.markdown(f"""
            <div style='text-align: right; font-size: 20px; font-weight: 600; color: #333; margin-bottom: 4px;'>
                🛒 총 수량: <span style='color:#d63384'>{total_items}개</span>
            </div>
            <div style='text-align: right; font-size: 22px; font-weight: 700; color: #222; margin-bottom: 16px;'>
                💰 총 금액: <span style='color:#198754'>₩{total_price:,}</span>
            </div>
            """, unsafe_allow_html=True)
            # 초기 팝업 상태 변수 정의
            if 'show_confirm_popup' not in st.session_state:
                st.session_state.show_confirm_popup = False

            # 구매 버튼 클릭 시 팝업 창 활성화
            if st.button("🛍 지금 구매하기", use_container_width=True):
                st.session_state.show_confirm_popup = True

            # 팝업 창 구성
            if st.session_state.show_confirm_popup:
                with st.expander("🛒 구매하시겠습니까?", expanded=True):
                    st.markdown("#### 🧾 구매 내역 요약")

                    # 팝업 내 장바구니 요약 출력
                    total_items = 0
                    total_price = 0
                    for key, info in st.session_state.cart.items():
                        qty = info["qty"]
                        weight = info.get("weight", 0) if isinstance(info, dict) else 0
                        unit = info.get("unit", "g") if isinstance(info, dict) else "g"
                        price = info.get("price", 0) if isinstance(info, dict) else 0
                        item_total = qty * price
                        total_items += qty
                        total_price += item_total
                        st.write(
                            f"- **{info.get('display_name', key)}**: {qty}개 x {weight}{unit} @ ₩{price:,} → ₩{item_total:,}"
                        )
                    # 구매 확인 및 취소 버튼
                    col_confirm, col_cancel = st.columns(2)
                    
                    # 구매 버튼
                    with col_confirm:
                        if st.button("✅ 예, 구매합니다"):
                            st.session_state.purchase_confirmed = True
                            
                    if st.session_state.get("purchase_confirmed"):
                        st.success("🎉 구매가 완료되었습니다!")

                        # 레시피 정보 생성
                        df_recipe['id'] = df_recipe['id'].astype(str)
                        matched_recipes = df_recipe[df_recipe['id'].isin(recipe_cart)]
                        recipe_info = matched_recipes[['id', 'name']].to_dict(orient='records')

                        # 주문번호 생성
                        user_num = int(user['userNum'])
                        now = datetime.now().strftime("%y%m%d%H%M%S")
                        order_id = f"{now}{user_num}"

                        # 재료 생성
                        filtered_items = []
                        for item in cart.values():
                            filtered_items.append({
                                "display_name": item["display_name"],
                                "price": int(item["price"]) if isinstance(item["price"], np.integer) else item["price"],
                                "weight": int(item["weight"]) if isinstance(item["weight"], np.integer) else item["weight"],
                                "unit": item["unit"],
                                "qty": item["qty"]
                            })

                        log_event(user_num=str(user_num), 
                                  os_type="Chrome", 
                                  log_type="cartPurchase", 
                                  parameter={"레시피": recipe_info, "재료": filtered_items, "주문번호": str(order_id), "총구매금액": int(total_price)})

                    # 취소 버튼
                    with col_cancel:
                        if st.button("❌ 아니요"):
                            st.session_state.show_confirm_popup = False


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
# 운영관리 메뉴
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

# 운영관리 메뉴 - 메인

if page == 'Summary Board':

    st.title('📊 Summary Board')

    st.markdown(" ")

    col1, col2 = st.columns([1, 18])

    with col1:
        st.write("")

    with col2:
        # 데이터프레임 불러오기
        df = st.session_state["df_total"]

        cols = ["month", "day", "count", "total", "ARPU"]
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

        # 오늘 기준 날짜 계산
        today = datetime.now()
        yesterday = today - timedelta(days=1)

        # 전월 동일 일자 계산 (예외처리 포함)
        try:
            prev_month_same_day = yesterday.replace(month=yesterday.month - 1)
        except ValueError:
            prev_month_last_day = yesterday.replace(day=1) - timedelta(days=1)
            prev_month_same_day = prev_month_last_day.replace(
                day=min(yesterday.day, prev_month_last_day.day)
            )

        # 어제와 전월 동일 일자의 데이터 추출
        df_yesterday = df[(df["month"] == yesterday.month) & (df["day"] == yesterday.day)].iloc[0]
        df_prev_month = df[(df["month"] == prev_month_same_day.month) & (df["day"] == prev_month_same_day.day)].iloc[0]

        # 변화율 및 방향 계산 함수
        def calc_change(curr, prev):
            if prev == 0:
                return 0, "변화 없음"
            change = ((curr - prev) / prev) * 100
            direction = "상승하였습니다" if change > 0 else "하락하였습니다" if change < 0 else "변화 없음"
            return abs(round(change, 1)), direction

        # 지표별 변화율 계산
        count_change, count_dir = calc_change(df_yesterday["count"], df_prev_month["count"])
        total_change, total_dir = calc_change(df_yesterday["total"], df_prev_month["total"])
        arpu_change, arpu_dir = calc_change(df_yesterday["ARPU"], df_prev_month["ARPU"])

        # 마크다운 리포트 출력
        summary = (f"""
        #### {yesterday.year}년 {yesterday.month}월 {yesterday.day}일 기준 요약

        - **총 상품 판매 개수**는 **{df_yesterday["count"]}개**이며,  
        전월 동일 일자 기준으로 **{count_change}% {count_dir}**.

        - **총 매출액**은 **{df_yesterday["total"]:,}원**이며,  
        전월 동일 일자 기준으로 **{total_change}% {total_dir}**.

        - **ARPU**는 **{df_yesterday["ARPU"]:,}원**이며,  
        전월 동일 일자 기준으로 **{arpu_change}% {arpu_dir}**.
        """)

        st.markdown(summary)
        
        # OpenAI 클라이언트
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        st.markdown(" ")

        # 챗봇 제목
        st.subheader("🤖 매출 요약 챗봇")

        # 초기 분석 메시지를 한 번만 생성
        if "messages" not in st.session_state:
            st.session_state.messages = []

            with st.spinner("📊 데이터 분석 중입니다..."):
                # GPT에게 분석 요청
                system_prompt = "너는 데이터 분석 어시스턴트야. 아래 매출 요약을 보고, 간단한 분석과 인사이트를 대화 형식으로 설명해줘."
                user_prompt = f"다음은 매출 요약 데이터야:\n{summary}"

                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                summary_analysis = res.choices[0].message.content.strip()

            # 메시지 저장
            st.session_state.messages.append({"role": "assistant", "content": summary_analysis})

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 사용자 입력 처리
        if user_input := st.chat_input("무엇이 궁금하신가요?"):
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("분석 중입니다..."):
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "너는 데이터 분석 및 설명 전문가야. 사용자 질문에 친절히 답해줘."},
                            *st.session_state.messages  # 전체 대화 맥락 전달
                        ]
                    )
                    reply = response.choices[0].message.content.strip()
                    st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})

# 운영관리 메뉴 - 전략 기획
elif page == "전략 기획":

    st.title('📈 전략 기획')

    # Tableau 대시보드 URL
    viz_url = "https://public.tableau.com/views/__17458989623690/1"

    # HTML + JS 삽입
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"></script>
            <style>
                #tableauViz {{
                    width: 100%;
                    height: 800px;
                }}
            </style>
        </head>
        <body>
            <tableau-viz id="tableauViz" src="{viz_url}" toolbar="bottom" hide-tabs></tableau-viz>
        </body>
        </html>
        """,
        height=800,
        width=1500
    )

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

# 운영관리 메뉴 - 마케팅
elif page == "마케팅":

    st.header('🛒 마케팅')

    # Tableau 대시보드 URL
    viz_url = "https://public.tableau.com/views/CVRDashboard/CVRDashboard"

    # HTML + JS 삽입
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"></script>
            <style>
                #tableauViz {{
                    width: 100%;
                    height: 800px;
                }}
            </style>
        </head>
        <body>
            <tableau-viz id="tableauViz" src="{viz_url}" toolbar="bottom" hide-tabs></tableau-viz>
        </body>
        </html>
        """,
        height=800,
        width=1500
    )    

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

# 운영관리 메뉴 - 공급망 관리
elif page == "공급망 관리":

    st.header('📦 공급망 관리')

    # Tableau 대시보드 URL
    viz_url = "https://public.tableau.com/views/_SCM/_SCM"

    # HTML + JS 삽입
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"></script>
            <style>
                #tableauViz {{
                    width: 120%;
                    height: 1800px;
                }}
            </style>
        </head>
        <body>
            <tableau-viz id="tableauViz" src="{viz_url}" toolbar="bottom" hide-tabs></tableau-viz>
        </body>
        </html>
        """,
        height=1800,
        width=1500
    )    

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
# 사용자 메뉴
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

# 사용자메뉴 - 메인
if page == "메인":

    # 저장소에서 필요한 정보 꺼내오기
    df_product = st.session_state["df_product"]
    df_recipe = st.session_state["df_recipe"]
    df_preference = st.session_state["df_preference"]
    df_similarity = st.session_state["df_similarity"]
    user = st.session_state["user"]

    # 평상시 메인 화면
    st.header("AI.re MART")
    st.markdown("##### 🔍 재료나 음식을 검색해보세요!")
    
    # 검색 콜백 함수(엔터로 검색)
    def trigger_search():
        # 텍스트 입력값을 세션에 저장하고, 검색 모드로 전환
        st.session_state.search_query = st.session_state.main_search_input
        st.session_state.search_mode = True
    
    col1, col2 = st.columns([4, 1])

    # 검색창
    col_empty, col1, col2, col_rest = st.columns([0.3, 5, 2 ,5])
    with col1: 
        search_query = st.text_input(
        label="검색",
        placeholder="검색할 재료명을 입력하세요",
        key="main_search_input",
        on_change=trigger_search,
        label_visibility="collapsed"  
        )

    # 검색버튼
    with col2: 
        if st.button("검색"):
            st.session_state.search_query = search_query
            st.session_state.search_mode = True    
            st.rerun()

    # 검색 모드일 때: 검색 결과만 그리고 바로 종료
    if st.session_state.get("search_mode", False):
        st.markdown("---")
        query = st.session_state.search_query
        st.markdown(f"### 🔍 “{query}” 검색 결과")
       
        # 빈 문자열 처리
        if not query.strip():
            st.warning("검색 결과가 없습니다.")
            if st.button("← 돌아가기", key="back_empty"):
                st.session_state.search_mode = False
                st.rerun()
            st.stop()
       
        # 재료 및 레시피 검색
        query = st.session_state.search_query
        if (
            "cached_recipe_query" not in st.session_state
            or st.session_state["cached_recipe_query"] != query
        ):
            st.session_state["cached_recipe_query"] = query
            st.session_state["cached_recipe_results"] = search_similar_recipes_with_vectordb(query, model, df_recipe)
        if (
            "cached_product_query" not in st.session_state
            or st.session_state["cached_product_query"] != query
        ):
            st.session_state["cached_product_query"] = query
            st.session_state["cached_product_results"] = search_products(query, df_product)

        product_results = st.session_state["cached_product_results"]
        recipe_results = st.session_state["cached_recipe_results"]

        # 검색 결과가 아무것도 없을 때
        if product_results.empty and recipe_results.empty:
            st.warning("검색 결과가 없습니다.")
            if st.button("← 돌아가기", key="back_none"):
                st.session_state.search_mode = False
                st.rerun()
            st.stop()
        
        # 상품 결과 먼저 출력
        if not product_results.empty:
            st.markdown("#### 🥬 관련 상품")
            # 4개씩 슬라이스해서 그리드로 뿌리기
            for slice_start in range(0, len(product_results), 4):
                row_slice = product_results.iloc[slice_start : slice_start + 4]
                cols = st.columns(4, gap="small")
                
                # 여기서 i는 0,1,2,3 ... 각 슬라이스 내에서의 상대 인덱스
                for i, (_, r) in enumerate(row_slice.iterrows()):
                    with cols[i]:
                        st.image(r["image"], use_container_width=True)
                        
                        # 장바구니에 추가
                        left, right = st.columns([4,1])
                        with right:
                            add_key = f"main_add_{slice_start + i}_{r['name'].replace(' ', '_')}"
                            st.button(
                                "🛒",
                                key=add_key,
                                on_click=lambda r=r: add_to_cart(
                                    st.session_state["cart"],
                                    r['domain'],
                                    r['division'],
                                    r["category"],
                                    r["name"],
                                    r.get("brand", ""), 
                                    r.get("weight", 0),  
                                    r.get("unit", ""),
                                    r.get("price", 0),
                                    r.get("image", "")
                                )
                            )

                        # 상품 정보
                        name_html = (
                            f"<strong>{r['brand']}</strong> {r['name']}"
                            if r.get("brand") and r["brand"] != "없음"
                            else r["name"]
                        )
                        st.markdown(f"""
                            <div style="
                                line-height:1.4;
                                margin-top:8px;
                            ">
                            <p style="font-size:20px;font-weight:600;margin:0 0 4px 0;">{name_html}</p>
                            <p style="font-size:20px;font-weight:bold;margin:0 0 4px 0;">{r['price']:,}원</p>
                            <p style="font-size:14px;color:#555;margin:0;">⭐ {r.get('score','N/A')} ({r.get('reviewCnt',0)}개)</p>
                            </div>
                        """, unsafe_allow_html=True)

                st.markdown("##")        
        
        # 레시피 결과 출력
        if not recipe_results.empty:
            st.markdown("#### 🍽️ 관련 레시피")
            # 4개씩 슬라이스해서 그리드로 뿌리기
            for slice_start in range(0, len(recipe_results), 4):
                row_slice = recipe_results.iloc[slice_start : slice_start + 4]
                cols = st.columns(4, gap="small")


                for i, (_, r) in enumerate(row_slice.iterrows()):
                    with cols[i]:
                        
                        # 레시피 추가
                        st.markdown(f"""
                            <div style="width: 290px; height: 290px; overflow: hidden; border-radius: 8px; margin-bottom: 12px;"">
                                <img src="{r['imgUrl']}" style="width: 100%; height: 100%; object-fit: cover;"/>
                            </div>
                        """, unsafe_allow_html=True)

                        left, right = st.columns([4,1])
                        with right:
                            st.button("📜", 
                                    key=f"add_{r['id']}",
                                    on_click=lambda r=r: (
                                        st.session_state.recipe_cart.append(r['id']) 
                                        if r['id'] not in st.session_state.recipe_cart else None
                                    )
                                )
                        # 레시피 정보
                        parsed = parse_recipe(r['inputRecipe'])

                        # "재료 수량" 형태로 문자열 리스트 만들기
                        ingredients_text = ", ".join(
                            f"{item['ingredient']} {item['quantity']}" if item['quantity'] else item['ingredient']
                            for item in parsed
                        )

                        st.markdown(f"""
                            <div style="line-height: 1.6; margin-bottom: 18px;">
                                <p style="font-size:20px; font-weight:bold; margin: 0 0 6px 0;">
                                    {r['name']}
                                </p>
                                <p style="font-size:16px; color:#444; margin: 0 0 6px 0;">
                                    {r['style']} / {r['category']} / {r['ingredient']}
                                </p>
                                <p style="font-size:15px; margin: 0 0 6px 0;">
                                    [재료] {ingredients_text}
                                </p>
                                <p style="font-size:14px; color:#555; margin: 0;">
                                    {r['time']}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("##")

        # 돌아가기 버튼
        if st.button("← 돌아가기"):
            st.session_state["search_mode"] = False
            st.rerun()
        st.stop()

    # 오늘의 추천 메뉴 배너
    st.markdown("---")
    st.subheader("오늘의 추천 메뉴")
    st.write("")
    
    # Top3 추천
    top3 = recommend_recipes(
        cart_dict={},  # 장바구니 없이
        recipe_df=df_recipe,
        similarity_df=df_similarity,
        user_num=int(user['userNum']),
        mode="preference",
        selected_recipe=list(map(str, st.session_state.recipe_cart))
    )

    top3 = top3[:3]
    st.session_state["top3_recipes"] = top3

    if top3:
        st.markdown(f"##### 📢 **{user['name']}님! 오늘 {top3[0]['name']}은 어떠세요? 😊**")

    render_recipe_cards(top3)

    # 이름 + 버튼 출력 따로
    cols = st.columns(3)

    for i, row in enumerate(top3):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="text-align:center; font-weight:600; margin-top:5px;">
                {row['name']}
            </div>
            """, unsafe_allow_html=True)

            left, right = st.columns([2, 1])
            with right:
                # 담기 버튼 (중복 방지 포함)
                if st.button("📜", key=f"add_top3_{row['id']}", help="레시피 장바구니에 추가"):
                    if row["id"] not in st.session_state.recipe_cart:
                        st.session_state.recipe_cart.append(row["id"])
                        st.rerun()

    if "opened_logged" not in st.session_state:
        banner_name = [x["name"] for x in top3[:3]]
        seq = list(range(1, len(banner_name) + 1))
        log_event(
            user_num=int(user['userNum']),
            os_type="Chrome",
            log_type="websiteOpen",
            parameter={"이름": banner_name, "노출순서": seq}
        )
        st.session_state["opened_logged"] = True
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

# 사용자 메뉴 - AIre봇
elif page == "AIre봇":

    # 데이터 불러오기
    df_recipe = st.session_state["df_recipe"]

    # 클라이언트 객체 생성
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # 제목
    st.title("AIre봇 🍽️")

    # 세션 상태 관리
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 오늘은 어떤 요리를 만들어 보시겠어요?"}
        ]

    # 이전 메시지 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if user_input := st.chat_input("현재 기분이나 상황, 재료 보유 현황 등을 알려주세요! 당신에게 딱 맞는 요리를 추천해드려요."):
        # 사용자 메시지 출력
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("🍳 당신에게 딱 맞는 요리를 고르는 중이에요..."):
                # [1] intent 분류
                intent = classify_user_intent(user_input, client)

                # [2] GPT로 키워드 추출
                keywords = chatbot_recommendation(client, user_input, intent)


                # [3] ChromaDB에서 레시피 검색
                recipe_results = choramadb_search(keywords, model)

                # [4] GPT가 최종 3개 레시피 + 이유 추출
                final_response = gpt_select_recipe(client, user_input, recipe_results)

                # [5] 어시스턴트 응답 표시                   
                final_response = gpt_select_recipe(client, user_input, recipe_results)
                st.markdown(final_response)

        st.session_state.messages.append({"role": "assistant", "content": final_response})

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

# 사용자 메뉴 - 사용자 설정
elif page == "사용자 설정":

    # 데이터 불러오기
    df_recipe = st.session_state["df_recipe"]
    user = st.session_state["user"]

    # 세션 상태 초기화
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "selected_ids" not in st.session_state:
        st.session_state.selected_ids = []
    if "excluded" not in st.session_state:
        st.session_state.excluded = []

    # 1단계 : 마음에 드는 레시피 선택
    if st.session_state.step == 1:
        st.header("🍽️ 마음에 드는 식단을 최소 5개 이상 선택하세요 🍽️")

        # 줄간격 공백
        st.write(" ")

        # 레시피 20개 무작위 추출 
        sampled = df_recipe.sample(20, random_state=42)
        selected_ids = []

        # 4열 레이아웃 생성
        recipes = sampled.to_dict('records')  
        selected_ids = []

        cols = st.columns(min(4, len(recipes)))

        for i, recipe in enumerate(recipes):
            with cols[i % 4]:
                # 이미지 카드 출력
                st.markdown(f"""
                    <div style="width:180px; height:180px; margin:0 auto; overflow:hidden; border-radius:10px;">
                        <img src="{recipe['imgUrl']}" style="width:100%; height:100%; object-fit:cover;"/>
                    </div>
                """, unsafe_allow_html=True)

                # 체크박스 바로 아래 출력
                col1, col2 = st.columns([1, 4])  # col1은 여백, col2에 체크박스
                with col2:
                    if st.checkbox(f"{recipe['name']}", key=f"sel_{i}"):
                        selected_ids.append(recipe['id'])

        # 줄간격 공백
        st.write(" ")

        # 다음 단계로 넘어가기 버튼
        if st.button("다음 단계로 ➡︎"):
            if len(selected_ids) < 5:
                st.warning("최소 5개 이상 선택해 주세요.")
            else:
                # 선택 결과 저장하고 다음 단계 진행
                st.session_state.selected_ids = selected_ids
                st.session_state.step = 2
                st.rerun()

    # 2단계: 알러지 및 기피 식재료 선택
    elif st.session_state.step == 2:
        st.header("🍽️ 알러지나 기피 식재료를 선택해 주세요 🍽️")

        # 제외할 재료 카테고리 목록
        options = ["곡류", "밀가루", "과일류", "닭고기", "돼지고기", "소고기", "채소류", "해물류"]

        # 제외할 항목 다중 선택
        excluded = st.multiselect("🚫 제외할 재료", options)

        # 추천 결과 보기 버튼
        if st.button("완료"):
            st.markdown("##### 설문을 완료하셨습니다. 📢 지금부터 고객님의 선호가 반영된 레시피를 추천받으실 수 있습니다.")
            st.write("💌 선호도가 반영된 레시피는 메인 화면에서 확인해보세요! 💌")

    # 3단계: 추천 결과 생성 및 DB 저장
    elif st.session_state.step == 3:
        st.header("3️⃣ 추천 결과 및 DB 저장")

        # 선택 및 제외 항목 불러오기
        selected_ids = st.session_state.selected_ids
        excluded = st.session_state.excluded
        user_id = int(user['userNum'])

        # 유사도 테이블 및 인코딩 테이블 생성
        df_similarity, df_encoded = generate_similarity_table(df_recipe, selected_ids, excluded)
        df_preference = generate_preference_table(df_encoded, selected_ids)

        # 사용자 ID 추가
        df_similarity["userNum"] = user_id
        df_preference["userNum"] = user_id

        # 결과를 Mysql 임시 테이블에 저장
        df_similarity.to_sql("similarity_tmp", con=get_mysql_connection(), if_exists="append", index=False)
        df_preference.to_sql("preference_tmp", con=get_mysql_connection(), if_exists="append", index=False)

        st.success(f"✅ {user_id} 저장 완료")

        # 추천 결과 Top3 출력
        st.subheader("🎯 추천 결과 Top 3")
        top3 = df_similarity[df_similarity["exception"].isna()].sort_values(by="similarity", ascending=False).head(3)
        for row in top3.itertuples():
            st.markdown(f"🔹 **{row.id}** (유사도: {row.similarity:.3f})")

        # 다시 시작 버튼
        if st.button("🔁 다시 시작"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

# 사용자 메뉴 - 레시피 추천 및 장바구니
elif page == "레시피 추천 및 장바구니":

    df_product = st.session_state["df_product"]
    df_recipe = st.session_state["df_recipe"]
    user = st.session_state["user"]
    recipe_cart = st.session_state["recipe_cart"]
    df_similarity = st.session_state["df_similarity"]

    st.header("🍽️ 레시피 추천 시스템")

    # 레시피 재료명 데이터 전처리
    df_recipe['parsedRecipe'] = df_recipe['inputRecipe'].apply(parse_recipe)
    left_col, right_col = st.columns([5, 3])

    # 레시피 출력
    with left_col:
        # 레시피 중복 제거
        selected_ids = list(map(str, st.session_state.recipe_cart))

        # 선호도 기반 레시피 추천
        top3 = recommend_recipes(
            cart_dict={},
            recipe_df=df_recipe,
            similarity_df=df_similarity,
            user_num=int(user['userNum']),
            mode="preference",
            selected_recipe=selected_ids
        )

        # 3개만 출력
        top3 = top3[:3]

        # 화면 출력
        render_recipe_recommendation(top3, "😍 취향저격 레시피", "top", df_product)

        # 장바구니 기반 레시피 추천
        cart_based_recipes = recommend_recipes(
            cart_dict=st.session_state.cart,
            recipe_df=df_recipe,
            similarity_df=df_similarity,
            user_num=int(user['userNum']),
            mode="basic",
            selected_recipe=selected_ids
        )

        # 3개만 출력
        cart_based_recipes = cart_based_recipes[:3]

        # 화면 출력
        render_recipe_recommendation(cart_based_recipes, "🛒 지금 담은 재료로 만들 수 있는 레시피", "cart", df_product)

        # recipe_cart에 담겨있는 레시피 ID랑 df_recipe에 있는 ID랑 매치
        selected_recipes_df = df_recipe[df_recipe['id'].astype(str).isin(recipe_cart)]

        # 장바구니에 담아있는 상품이랑 recipe_cart에 담아있는 레시피에 있는 재료로 중량 계산해서 남는 재료 도출
        remain = get_remaining_cart(st.session_state.cart, selected_recipes_df)

        remain = {
            k: v for k, v in remain.items()
            if v.get("weight", 0) >= 100
        }

        # 남은 재료 기반 레시피 추천
        remain_recipes = recommend_recipes(
            cart_dict=remain,
            recipe_df=df_recipe,
            similarity_df=df_similarity,
            user_num=int(user['userNum']),
            mode="remain",
            selected_recipe=selected_ids
        )

        # 3개만 출력
        remain_recipes = remain_recipes[:3]

        # 화면 출력
        render_recipe_recommendation(remain_recipes, "🌱 남는 재료로 만들 수 있는 레시피", "remain", df_product)

                          
    # 장바구니
    with right_col:
        # 🛒 장바구니 타이틀 + 비우기 버튼 상단에 배치
        header_col, clear_col = st.columns([5, 1])
        
        with header_col:
            st.subheader("🛒 장바구니")

        with clear_col:
            if st.button("🧹", help="장바구니 비우기", key="clear_cart"):
                st.session_state.cart.clear()
                st.session_state.purchased_weight.clear()
                st.session_state.selected_recipes.clear()
                st.session_state.selected_recipe_sources.clear()
                st.session_state.recipe_cart.clear()
                st.rerun()

        updated_cart = {}
        if not st.session_state.cart:
            st.markdown("##### 😭 현재 장바구니가 비어있습니다.")

        else:
            # 기존대로 cart 항목 렌더링 & 수량 변경 처리
            for key, info in st.session_state.cart.items():
                cols = st.columns([6, 1])

                # [1] 재료 정보
                display_name = info["display_name"]
                image_url = apply_image_proxy(info.get("image", ""))
                qty = info["qty"]
                weight = info["weight"]
                unit = info["unit"]
                price = info["price"]
                total_w = weight * qty
                total_p = price * qty

                # [2] 좌측: 수량 조절 + 카드 렌더링
                with cols[0]:
                    new_qty = st.number_input(
                        label="",
                        min_value=0,
                        value=qty,
                        step=1,
                        key=f"cart_qty_{key}",
                        label_visibility="collapsed"
                    )
                    total_w = weight * new_qty
                    total_p = price * new_qty
                    st.session_state.purchased_weight[key] = total_w

                    st.markdown(f"""
                    <div style="border:1px solid #eee; border-radius:8px; padding:12px; display:flex; gap:12px; align-items:flex-start; justify-content:space-between;">
                        <img src="{image_url}" width="64" height="64" style="border-radius:6px;"/>
                        <div style="flex:1;">
                            <div style="font-weight:600;">{display_name}</div>
                            <div>수량: {new_qty}개 총량: {total_w}{unit}</div>
                            <div>금액: ₩{total_p:,}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    updated_cart[key] = {
                        "qty": new_qty,
                        "display_name": display_name,
                        "brand": info.get("brand",""),
                        "weight": weight,
                        "unit": unit,
                        "price": price,
                        "image": image_url,
                        "category": info.get("category", ""),
                        "division": info.get("division", ""),
                    }

                # [3] 우측: 제거 버튼 (정상 Streamlit 방식)
                with cols[1]:
                    remove_style = """
                    <style>
                    div[data-testid="stButton"] button {
                        color: #dc3545;
                        font-weight: bold;
                        font-size: 20px;
                    }
                    </style>
                    """
                    st.markdown(remove_style, unsafe_allow_html=True)

                    if st.button("✕", key=f"remove_cart_{key}"):
                        st.session_state.cart.pop(key, None)
                        st.session_state.purchased_weight.pop(key, None)
                        st.rerun()

            st.session_state.cart.update(updated_cart)

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
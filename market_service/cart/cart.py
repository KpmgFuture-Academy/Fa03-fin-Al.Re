import re
from collections import defaultdict
import pandas as pd

def get_remaining_cart(cart_dict, parsed_recipe_df):
    """
    장바구니(cart_dict)와 레시피 DataFrame(parsed_recipe_df)의 재료 정보를 바탕으로,
    사용 후 남은 재료(weight) 정보를 계산하여 반환하는 함수.

    Args:
        cart_dict (dict): 현재 장바구니 딕셔너리
        parsed_recipe_df (DataFrame): 'parsedRecipe' 열이 존재하는 레시피 DataFrame (recipe_cart 기준)

    Returns:
        dict: 재료별로 남은 무게(weight)가 반영된 cart_dict 형태의 딕셔너리
    """

    # 1. 전체 레시피의 parsedRecipe 컬럼에서 재료 리스트 평탄화 (list[dict] 형태)
    parsed_list = []
    for lst in parsed_recipe_df['parsedRecipe']:
        parsed_list.extend(lst)

    # 2. 사용된 재료 무게 누적 (단위는 g 기준)
    used = defaultdict(float)
    for item in parsed_list:
        name = item["ingredient"]
        match = re.match(r"([\d.]+)([a-zA-Z]+)", item["quantity"])  # 예: "300g" → ("300", "g")
        q = float(match.group(1)) if match else 0
        used[name] += q

    # 3. 남은 재료 계산
    remain = {}
    for key, info in cart_dict.items():
        name_in_cart = info.get("display_name", "")
        weight = info.get("weight", 0)
        qty = info.get("qty", 1)
        total = weight * qty  # 수량 고려한 총 보유량

        # 4. 사용된 재료 목록과 비교하여 남은 양 계산
        for ing_name, q in used.items():
            # display_name / category / division 중 하나라도 매치되면 차감
            if ing_name in name_in_cart or ing_name == info.get("category") or ing_name == info.get("division"):
                left = total - q
                if left > 0:
                    remain[key] = info.copy()
                    remain[key]["weight"] = left
                # 0g 이하이면 remain에 포함하지 않음
                break
        else:
            # 매치되지 않은 항목은 원본 그대로 유지
            remain[key] = info.copy()

    return remain

def parse_recipe(text):

    """
    레시피 재료 문자열을 전처리하고, 재료명과 수량을 분리하는 함수.

    Args:
        text: 원본 재료 문자열 (예: "감자300g|당근200g")

    Returns:s
        재료명과 수량을 분리한 딕셔너리 리스트
        예시: [{"ingredient": "감자", "quantity": "300g"}, {"ingredient": "당근", "quantity": "200g"}]
    """

    # 1. 대괄호 []와 소괄호 () 안의 내용을 제거 (예: [국산], (생략 가능))
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # 2. '|' 기호를 기준으로 재료 항목 분리
    items = text.split('|')
    
    parsed = []

    # 3. 각 재료 항목에 대해 처리
    for item in items:
        item = item.strip()  # 양쪽 공백 제거
        if not item:
            continue  # 빈 항목은 스킵

        # 4. 숫자가 처음 등장하는 위치를 찾아 재료명과 수량을 분리
        match = re.search(r'\d', item)
        if match:
            idx = match.start()
            ingredient = item[:idx].strip().replace(' ', '')  # 숫자 앞까지: 재료명
            quantity = item[idx:].strip()    # 숫자부터 끝까지: 수량
        else:
            ingredient = item.strip().replace(' ', '')  # 숫자가 없으면 전체를 재료명으로
            quantity = ''

        # 5. 파싱 결과 추가
        parsed.append({
            "ingredient": ingredient,
            "quantity": quantity
        })
    
    return parsed

def add_to_cart(cart: dict, domain: str, division: str, category: str,
                name: str, brand: str, weight: float, unit: str, price: int, image: str):
    """
    장바구니(cart)에 상품을 추가하거나, 이미 담긴 상품의 수량을 증가시키는 함수.

    Args:
        cart: 장바구니 정보를 담고 있는 딕셔너리 (session_state["cart"])
        domain: 상품 도메인 (예: 식품, 생활용품 등)
        division: 상품 대분류 (예: 채소, 음료)
        category: 상품 중분류 (예: 뿌리채소, 과일주스)
        name: 상품 이름 (예: 흙당근)
        brand: 브랜드명 (없으면 빈 문자열)
        weight: 상품 1개당 중량 (예: 300g → 300.0)
        unit: 중량 단위 (예: g, ml 등)
        price: 상품 가격
        image: 상품 이미지 URL (같은 이름의 상품이라도 이미지가 다르면 다른 상품으로 간주)

    Returns:
        None. cart 딕셔너리를 직접 수정함.
    """

    # 1. 고유한 key를 생성: 상품명 + 이미지 경로 조합
    key_name = f"{name.replace(' ', '_')}_{image}"

    # 2. 장바구니에 이미 존재하는 상품이면 수량만 1 증가
    if key_name in cart:
        cart[key_name]["qty"] += 1

    # 3. 장바구니에 없는 상품이면 새 항목으로 추가
    else:
        cart[key_name] = {
            "qty": 1,             
            "domain": domain,
            "division": division,
            "category": category,
            "display_name": name,   
            "brand": brand,
            "weight": weight,
            "unit": unit,
            "price": price,
            "image": image
        }

def recommend_recipes(cart_dict, recipe_df, similarity_df, user_num, mode="basic", selected_recipe=None):

    """
    추천 모드(basic / remain / preference)에 따라 선호도가 반영된 레시피를 추천하는 함수

    Args:
        cart_dict (dict): 장바구니 or 남은 재료 정보
        recipe_df (pd.DataFrame): 전체 레시피 데이터프레임 (parsedRecipe 포함)
        similarity_df (pd.DataFrame, optional): 사용자-레시피 유사도 및 예외 정보
        user_num (int, optional): 사용자 고유 번호
        mode (str): 추천 방식 ("basic", "remain", "preference")
        selected_recipe (list[str], optional): 중복 제거용 레시피 ID 리스트

    Returns:
        list: 추천 레시피 딕셔너리 리스트
    """
    
    # 예외처리 : 없는 모드를 입력했을 경우
    if mode not in ("basic", "remain", "preference"):
        raise ValueError(f"지원하지 않는 추천 모드입니다: {mode}")
    
    # 1. 유사도 필터링
    similarity_df["userNum"] = pd.to_numeric(similarity_df["userNum"], errors="coerce")
    user_similarity_df = similarity_df[
        (similarity_df["userNum"] == user_num) & 
        (similarity_df["exception"] != 1)
    ]

    user_similarity_df["id"] = user_similarity_df["id"].astype(str)
    recipe_df["id"] = recipe_df["id"].astype(str)

    # 2. 유사도 기반 추천 후보 ID 추출
    top_ids = user_similarity_df["id"].tolist()
    top_recipes = recipe_df[recipe_df["id"].isin(top_ids)].copy()

    # 3. similarity 점수 부여
    similarity_map = user_similarity_df.set_index("id")["similarity"].to_dict()
    top_recipes["similarity"] = top_recipes["id"].map(similarity_map)

    # 4. preference 모드: 유사도 순 정렬만
    if mode == "preference":
        top_recipes = top_recipes.sort_values(by="similarity", ascending=False) 
        return [
            {
                'id': row['id'],
                'name': row.get('name'),
                'similarity': row['similarity'],
                'imgUrl': row.get('imgUrl', ''),
                'parsedRecipe': [
                    item['ingredient'] for item in row.get('parsedRecipe', [])
                    if isinstance(item, dict) and 'ingredient' in item
                ],
                'portnum': row.get("portNum"),
            }
            for _, row in top_recipes.iterrows()
            if str(row['id']) not in selected_recipe
        ]

    # 5. 장바구니 기반 모드일 경우
    if not cart_dict:
        return []

    # 6. 장바구니 사전 정보 구성
    remaining_weights = {
        v.get("display_name", k): v.get("weight", 0)
        for k, v in cart_dict.items()
    }
    categories = set(
        sub_cat.strip()
        for info in cart_dict.values()
        if "category" in info and info["category"]
        for sub_cat in info["category"].split('/')
    )
    divisions = set(
        sub_div.strip()
        for info in cart_dict.values()
        if "division" in info and info["division"]
        for sub_div in info["division"].split('/')
    )
    category_priority = {cat: idx for idx, cat in enumerate(categories)}
    division_priority = {div: idx for idx, div in enumerate(divisions)}

    recommended = []

    # 7. top_recipes 순회하며 매칭 계산
    for _, row in top_recipes.iterrows():
        recipe_id = str(row.get("id"))
        if recipe_id in selected_recipe:
            continue

        parsed_recipe = row.get('parsedRecipe')
        if not isinstance(parsed_recipe, list):
            continue

        matched_ingredients = []
        matched_priorities = []
        matched_weights = []
        total_matched_weight = 0.0

        for item in parsed_recipe:
            ing = item.get("ingredient")
            quantity = item.get("quantity", "")

            if mode == "basic":
                cat_match = any(cat in ing or ing in cat for cat in categories)
                if cat_match:
                    matched_ingredients.append(ing)
                    matched_priorities.append(
                        min((category_priority.get(cat, 999) for cat in categories if cat in ing or ing in cat), default=999)
                    )
                    continue
                div_match = any(div == ing for div in divisions)
                if div_match:
                    matched_ingredients.append(ing)
                    matched_priorities.append(division_priority.get(ing, 999))

            elif mode == "remain":
                match = re.match(r"([\d.]+)([a-zA-Z]+)", quantity)
                weight = float(match.group(1)) if match else 0.0
                for key in remaining_weights:
                    if ing in key or key in ing:
                        matched_ingredients.append(ing)
                        matched_weights.append(weight)
                        total_matched_weight += weight
                        break

        if matched_ingredients:
            recommended.append({
                'id': recipe_id,
                'name': row.get("name"),
                'similarity': row.get("similarity", 0.0),
                'imgUrl': row.get('imgUrl', ''),
                'parsedRecipe': [item['ingredient'] for item in parsed_recipe if 'ingredient' in item],
                'matched': matched_ingredients,
                'matched_priorities': matched_priorities,
                'matched_weights': matched_weights,
                'total_matched_weight': total_matched_weight,
                'match_type': 'category' if mode == "basic" else 'weight',
                'portnum': row.get("portNum"),
            })

    # 8. 정렬
    if mode == "basic":
        recommended.sort(key=lambda r: (
            0 if r['match_type'] == 'category' else 1,
            -len(r['matched']),
            min(r['matched_priorities']) if r['matched_priorities'] else 999,
            -r['similarity']
        ))
    elif mode == "remain":
        recommended.sort(key=lambda r: (
            -r['total_matched_weight'],
            -len(r['matched']),
            -r['similarity']
        ))

    return recommended

def recipe_serving_price(cart_dict, parsed_recipe, port_num):
    """
    레시피 재료 목록과 장바구니(cart_dict)를 비교하여 1인분 가격을 계산 (category → division → name 순 매칭)
    """
    total_price = 0.0

    for item in parsed_recipe:
        name = item["ingredient"]
        quantity_str = item["quantity"]

        match = re.match(r"([\d.]+)([a-zA-Z가-힣]+)", quantity_str)
        if not match:
            continue

        qty = float(match.group(1))

        matched_key = None
        for key, info in cart_dict.items():
            cat_parts = (info.get("category") or "").split("/")
            div_match = name in (info.get("division") or "")
            name_match = name in (info.get("display_name") or "")

            # 1) category의 부분이 재료명에 포함되어 있으면 우선 매칭
            if name in cat_parts:
                matched_key = key
                break
            # 2) division과 포함 관계
            elif div_match:
                matched_key = key
                break
            # 3) 상품명 (display_name)과 포함 관계
            elif name_match:
                matched_key = key
                break

        if matched_key and cart_dict[matched_key]["weight"] > 0:
            unit_price = cart_dict[matched_key]["price"] / cart_dict[matched_key]["weight"]
            total_price += unit_price * qty

    return round(total_price / port_num) if port_num else 0
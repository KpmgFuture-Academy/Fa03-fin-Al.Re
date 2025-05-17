import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from chromadb import PersistentClient

def search_products(query, df):

    """
    상품 데이터프레임(df)에서 사용자의 쿼리(query)를 기반으로
    category, division, name 순서로 검색 결과를 반환하는 함수.

    Args:
        query: 사용자 입력 검색어 (str)
        df: 상품 데이터프레임 (DataFrame)

    Returns:
        검색 결과에 해당하는 데이터프레임 (DataFrame)
    """

    # category 검색: '/'로 분할해서 query와 '완전 일치'하는 항목 찾기
    mask_cat = df['category'].fillna('').apply(lambda s: query in s.split('/'))
    cat_df = df[mask_cat]

    # 쿼리가 한 글자인지 확인
    is_single_char = len(query) == 1

    # category 매칭 결과가 존재할 경우
    if not cat_df.empty:

        # 쿼리가 한 글자이면 name 검색은 생략
        if is_single_char:
            return cat_df.reset_index(drop=True)
        
        # 쿼리가 한 글자 이상이면 name 검색 / 이미 category로 매칭된 건 제외
        else:
            mask_name = (
                df['name'].str.contains(query, na=False)
                & ~mask_cat
            )
            name_df = df[mask_name]

            # category + name 매칭 결과 합치기
            return pd.concat([cat_df, name_df], ignore_index=True)

    # category 매칭 결과가 존재하지 않을 경우(division 검색)
    else:
        mask_div = df['division'].str.contains(query, na=False)
        div_df = df[mask_div]

        # 쿼리가 한 글자이면 name 검색은 생략
        if is_single_char:
            return div_df.reset_index(drop=True)
        
        # 쿼리가 한 글자 이상이면 name 검색 / 이미 division으로 매칭된 건 제외
        else:
            mask_name = (
                df['name'].str.contains(query, na=False)
                & ~mask_div
            )
            name_df = df[mask_name]

            # division + name 매칭 결과 합치기
            return pd.concat([div_df, name_df], ignore_index=True)
        


def search_similar_recipes(query, df, top_n=8):

    """
    사용자 쿼리(query)와 레시피 이름들(df['name']) 간의 의미적 유사도를 계산해
    가장 유사한 레시피 상위 N개를 반환하는 함수.

    Args:
        query (str): 사용자가 입력한 검색 문장 또는 키워드
        df (pd.DataFrame): 레시피 데이터프레임 (name 컬럼 필수)
        top_n (int, optional): 반환할 유사 레시피 개수 (기본값: 8)

    Returns:
        pd.DataFrame: 유사도 상위 N개의 레시피 + similarity 컬럼 포함
    """

    # 한국어 SBERT 임베딩 모델 로드
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device='cpu')

    # 레시피 제목 리스트 생성
    titles = df['name'].fillna("").tolist()

    # 문장 임베딩 생성
    embeddings = model.encode([query] + titles, convert_to_tensor=True)

    # 코사인 유사도 계산
    cos_sim = cosine_similarity(
        embeddings[0].unsqueeze(0).cpu().numpy(),
        embeddings[1:].cpu().numpy()
    ).flatten()

    # 유사도 기준 내림차순 상위 N개 인덱스 추출
    top_indices = cos_sim.argsort()[::-1][:top_n]

    # 상위 레시피 반환, 유사도 점수열 추가
    return df.iloc[top_indices].assign(similarity=cos_sim[top_indices])


def search_similar_recipes_with_vectordb(query, model, recipe_df, top_n=8):
    # ChromaDB 연결
    client = PersistentClient(path="C:/Users/Admin/workspace/market_service/vectordb/chroma_db")
    collection = client.get_collection(name="recipes_kr_sbert")

    # 쿼리 임베딩 생성
    query_embedding = model.encode([query]).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=top_n)

    # 메타데이터 + 유사도 DataFrame 생성
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    df = pd.DataFrame(metadatas)
    df["similarity"] = 1 - pd.Series(distances)

    df["id"] = df["id"].astype(int)
    recipe_df["id"] = recipe_df["id"].astype(int)

    # 이미지 URL 병합 (id 기준)
    df = df.merge(recipe_df[["id", "imgUrl", "inputRecipe", "time"]], on="id", how="left")

    return df

def generate_safe_key(recipe_name: str, product_name: str, ing_name: str, i: int, j: int):

    """
    Streamlit UI 컴포넌트의 고유 key 생성을 위한 함수.
    레시피 이름, 상품 이름, 재료 이름, 인덱스를 조합한 문자열을 MD5 해시하여 안전한 key를 만든다.

    Args:
        recipe_name: 레시피 이름 (str)
        product_name: 상품 이름 (str)
        ing_name: 재료 이름 (str)
        i: 반복문 외부 인덱스 (int)
        j: 반복문 내부 인덱스 (int)

    Returns:
        Streamlit 컴포넌트에 사용할 고유한 문자열 key
    """
    
    # 1. 레시피명, 재료명, 상품명, 인덱스(i, j)를 하나의 문자열로 합친다
    combined = f"{recipe_name}_{ing_name}_{product_name}_{i}_{j}"

    # 2. 이 combined 문자열을 MD5 해시로 변환하여 고유한 key를 생성한다
    return "card_select_" + hashlib.md5(combined.encode()).hexdigest()



import re
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from data import get_mysql_connection

# 사용자 번호 생성 (현재 사이트에서 유저 번호가 있기에 생성할 필요 없음)
def get_next_user_num():
    with get_mysql_connection().connect() as conn:
        result = conn.execute("SELECT MAX(userNum) FROM similarity").fetchone()[0]
        if result is None:
            return "user001"
        match = re.search(r"(\d+)$", result)
        if match:
            next_num = int(match.group(1)) + 1
            return f"user{next_num:03d}"
        else:
            return "user001"

# 유사도 테이블에 넣을 결과 생성
def generate_similarity_table(df, selected_ids, excluded):

    """
    사용자가 선택한 레시피 기반으로 레시피 유사도(similarity)를 계산하고,
    제외할 재료를 반영한 결과를 반환.

    Args:
        df: 전체 레시피 데이터프레임
        selected_ids: 사용자가 선택한 레시피 id 리스트
        excluded: 사용자가 제외하고 싶은 재료 리스트

    Returns:
        df_similarity: 유저-레시피별 유사도 결과
        df_encoded: 원-핫 인코딩된 레시피 데이터 (추후 preference 계산용)
    """
    today = datetime.today().strftime("%Y-%m-%d")

    # 필수 컬럼이 모두 채워진 데이터만 사용
    df_filtered = df.dropna(subset=["style", "instruction", "ingredient"])

    # style, instruction, ingredient 컬럼을 원-핫 인코딩 
    encoder = OneHotEncoder()
    feature_matrix = encoder.fit_transform(df_filtered[["style", "instruction", "ingredient"]]).toarray()

    # 인코딩된 데이터를 데이터프레임으로 변환
    df_encoded = pd.DataFrame(feature_matrix, columns=encoder.get_feature_names_out())
    df_encoded["id"] = df_filtered["id"].values
    df_encoded["name"] = df_filtered["name"].values
    df_encoded["ingredient"] = df_filtered["ingredient"].values

    # 선택된 레시피를 기반으로 유저 벡터 생성
    valid_ids = [rid for rid in selected_ids if rid in df_encoded["id"].values]
    user_vector = df_encoded[df_encoded["id"].isin(valid_ids)].drop(columns=["id", "name", "ingredient"]).mean().values.reshape(1, -1)

    # 전체 레시피와의 코사인 유사도 계산
    recipe_vectors = df_encoded.drop(columns=["id", "name", "ingredient"]).values
    similarities = cosine_similarity(user_vector, recipe_vectors).flatten()

    # 결과 테이블 구성
    df_similarity = pd.DataFrame({
        "id": df_filtered["id"],
        "similarity": similarities,
        "ingredient": df_filtered["ingredient"]
    })

    # 제외할 재료가 포함된 경우 '예외' 표시
    df_similarity["exception"] = df_similarity["ingredient"].apply(lambda x: "예외" if x in excluded else None)
    df_similarity.drop(columns=["ingredient"], inplace=True)
    df_similarity["partitionDate"] = today

    return df_similarity, df_encoded

# 선호도 테이블에 넣을 생성
def generate_preference_table(df_encoded, selected_ids):

    """
    사용자가 선택한 레시피를 기준으로 style, instruction, ingredient 별
    선호도(preference) 점수를 계산.

    Args:
        df_encoded: 원-핫 인코딩된 레시피 데이터프레임
        selected_ids: 사용자가 선택한 레시피 id 리스트

    Returns:
        df_preference: 유저-레시피별 선호도 점수 데이터프레임
    """

    # 컬럼 그룹 분리
    style_cols = [col for col in df_encoded.columns if col.startswith("style")]
    instr_cols = [col for col in df_encoded.columns if col.startswith("instruction")]
    ingre_cols = [col for col in df_encoded.columns if col.startswith("ingredient")]

    # 유저 벡터 생성 (평균값)
    user_style = df_encoded[df_encoded["id"].isin(selected_ids)][style_cols].mean().values
    user_instr = df_encoded[df_encoded["id"].isin(selected_ids)][instr_cols].mean().values
    user_ingre = df_encoded[df_encoded["id"].isin(selected_ids)][ingre_cols].mean().values

    # 코사인 유사도 점수 계산
    style_scores = cosine_similarity(user_style.reshape(1, -1), df_encoded[style_cols].values).flatten()
    instr_scores = cosine_similarity(user_instr.reshape(1, -1), df_encoded[instr_cols].values).flatten()
    ingre_scores = cosine_similarity(user_ingre.reshape(1, -1), df_encoded[ingre_cols].values).flatten()

    # 결과 테이블 구성
    df_preference = pd.DataFrame({
        "id": df_encoded["id"],
        "name": df_encoded["name"],
        "instruction": instr_scores,
        "ingredient": ingre_scores,
        "style": style_scores
    })

    return df_preference
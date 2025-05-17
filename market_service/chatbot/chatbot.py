from chromadb import PersistentClient


def classify_user_intent(user_input, client):
    prompt = f'''
    아래 문장을 읽고 사용자의 의도를 다음 중 하나로 분류해줘:
    [emotion_based, situation_based, ingredient_based, general]

    문장: "{user_input}"
    대답은 반드시 위 카테고리 중 하나로만 해줘.
    '''
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()

def chatbot_recommendation(client, user_input: str, intent: str) -> str:
    """
    사용자 입력과 intent를 기반으로 상황에 맞는 레시피 추천용 키워드를 반환합니다.
    intent에 따라 적절한 프롬프트 생성 함수를 선택하고, GPT 호출 결과에서 키워드만 추출합니다.
    """
    # intent → 프롬프트 함수 매핑
    prompt_map = {
        "emotion_based": emotion_prompt,
        "ingredient_based": ingredient_based,
        "situation_based": situation_based,
        "general": general_prompt
    }

    # 유효하지 않은 intent인 경우 대비 (default는 general_prompt 사용)
    prompt_func = prompt_map.get(intent, general_prompt)

    # 프롬프트 생성 및 GPT 호출
    messages = prompt_func(user_input)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    keyword = response.choices[0].message.content.strip()

    # 키워드만 반환
    return keyword

def emotion_prompt(user_input: str) -> list[dict]:
    """
    감정 기반 상황에서 GPT에게 전달할 메시지 리스트를 생성합니다.
    출력은 재료명 키워드만 GPT가 반환하게 유도합니다.
    """
    system = (
        "너는 사람들의 감정에 맞는 재료를 추천해주는 요리 어시스턴트야.\n"
        "사용자의 입력을 보고, 사용자의 감정을 분석한 다음 그 감정에 맞는 요리 재료를 2~3개만 골라줘.\n"
        "음식 이름, 조리법, 설명은 쓰지 말고, 재료 이름만 띄어쓰기로 구분해서 출력해줘.\n"
        "예: 두부 버섯 / 감자 양파 / 계란 파 참치"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input}
    ]

def ingredient_based(user_input: str) -> list[dict]:
    """
    재료 기반 질문에서 GPT에게 입력 문장 속 재료명만 추출하게 유도하는 프롬프트입니다.
    """
    system = (
        "너는 사용자의 문장에서 요리에 사용할 수 있는 재료명만 뽑아주는 역할이야.\n"
        "추가적인 감정 해석이나 요리 이름은 말하지 마.\n"
        "문장 안에 실제로 등장한 재료만 띄어쓰기로 나열해줘.\n"
        "예: 두부 버섯 / 감자 양파 / 계란 파 참치"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input}
    ]

def situation_based(user_input: str) -> list[dict]:
    """
    상황 기반 요리 추천을 위한 GPT 프롬프트 메시지를 생성합니다.
    GPT는 상황에 어울리는 재료 2~3개만 띄어쓰기로 출력해야 합니다.
    """
    system = (
        "너는 사용자의 상황에 맞는 요리 재료를 추천하는 역할이야.\n"
        "사용자의 문장을 읽고, 이 문장에서 어떤 상황이 일어나고 있는지 먼저 파악한 다음 그 상황에 어울리는 요리 재료를 2~3개만 골라줘.\n"
        "음식 이름, 조리법, 설명은 쓰지 말고, 재료 이름만 띄어쓰기로 구분해서 출력해줘.\n"
        "예: 두부 버섯 / 감자 양파 / 계란 파 참치"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input}
    ]

def general_prompt(user_input: str) -> list[dict]:
    """
    일반적인 요리 고민에 대해 GPT가 대중적이고 자주 쓰이는 재료 2~3개를 제시하도록 유도하는 프롬프트입니다.
    """
    system = (
        "너는 사용자의 일반적인 요리 고민에 대해 흔히 쓰이는 재료 2~3개를 추천하는 역할이야.\n"
        "감정, 상황, 재료 언급이 없는 질문일 경우에도 요리에 자주 쓰이는 재료를 제안해줘.\n"
        "재료 이름만 띄어쓰기로 출력하고, 음식명이나 설명은 절대 포함하지 마.\n"
        "예: 계란 양파 / 밥 김 참치 / 두부 김치"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input}
    ]

def choramadb_search(query, model):
    # 기존 ChromaDB 폴더에 연결
    client = PersistentClient(path="C:/Users/Admin/workspace/market_service/vectordb/chroma_db")

    # 컬렉션 불러오기
    collection = client.get_collection(name="recipes_kr_sbert")

    # 사용자 쿼리 → 임베딩
    query_embedding = model.encode([f'"{query}"']).tolist()

    # 유사 문서 검색
    result = collection.query(query_embeddings=query_embedding, n_results=10)

    # 레시피 출력
    return result["metadatas"][0]

def gpt_select_recipe(client, user_input: str, recipes: list[dict]) -> str:
    """
    사용자 입력과 Chroma에서 가져온 레시피 10개를 기반으로,
    GPT가 최종 추천 레시피 3개를 골라주고 그 이유를 설명합니다.
    """

    recipe_text_block = ""
    for i, recipe in enumerate(recipes, 1):
        recipe_text_block += f"{i}. 요리명: {recipe['name']}\n   재료: {recipe['inputrecipe']}\n\n"

    prompt = f"""
    다음은 사용자의 질문과 관련된 레시피 후보 10개입니다.

    [사용자 질문]
    {user_input}

    [후보 레시피 목록]
    {recipe_text_block}

    사용자의 질문을 고려해서 가장 잘 어울리는 레시피 3개를 골라줘.

    - 출력 형식은 아래와 같이 **줄마다 끊어서** 표현해줘. 한 줄로 이어 쓰지 마!
    1. 요리명 : <레시피 name>\n
       재료 : <레시피 inputrecipe>에 있는 재료를 그대로 사용하되, 재료 구분자는 쉼표(,)로 바꿔줘. (예: 두부 150g, 배추 100g ...)\n
       이 요리를 추천하는 이유는 ... 식으로 자연스럽게 풀어줘.

    - 반드시 위 형식처럼, 요리명/재료/설명 각각을 줄을 바꿔서 써줘.
    - 요리명과 재료는 반드시 위에 제시된 레시피 내용 그대로 사용하고, 한 글자도 바꾸면 안 돼.

    - (예외처리) 만약에 사용자가 요리와 관련된 아래 예시와 같은 답변을 해줬으면 좋겠어.
      예시 : 죄송하지만, 이순신의 탄생일에 대한 정보는 제공할 수 없습니다. 요리 재료 추천을 도와드릴 수 있습니다. 특정 상황이나 요리 재료가 필요하신가요?
    """

    messages = [
        {"role": "system", "content": "너는 사용자가 질문한 의도를 파악해서 요리를 추천해주는 전문가야. 응답할 때 꼭 출력형식을 지켜야 해."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    return response.choices[0].message.content.strip()
# 라이브러리 불러오기
from sentence_transformers import SentenceTransformer
import mysql.connector
import chromadb
from chromadb import PersistentClient
import time
from datetime import datetime
import shutil
import os

chroma_path = "./chroma_db"
if os.path.exists(chroma_path):
    shutil.rmtree(chroma_path)
    print("✅ 기존 ChromaDB 삭제 완료!")
else:
    print("✅ 초기화할 ChromaDB가 없습니다.")


# 임베딩 모델 로딩
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# MySQL 연결
conn = mysql.connector.connect(
    host="192.168.14.53",
    user="dongdong",
    password="20250517",
    database="ai_re",
    connection_timeout=3600     # 1시간까지 MySQL 연결 유지
)
cursor = conn.cursor()

# ChromaDB 연결 및 컬렉션 생성
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="recipes_kr_sbert")

# 중복 키 체크용 세트 (re_name + re_ingredient)
existing_data = collection.get()
existing_keys = set()
for doc, meta in zip(existing_data["documents"], existing_data["metadatas"]):
    key = f"{meta.get('name', '')}|{meta.get('ingredient', '')}"
    existing_keys.add(key)

# 배치 적재 파라미터 설정
batch_size = 500
offset = 0
total_inserted = 0

# 총 개수 파악 (예상 시간 계산용)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM recipe")
total_rows = cursor.fetchone()[0]
cursor.close()
estimated_batches = total_rows // batch_size + 1
print(f"\n🧲 전체 레시피 수: {total_rows}개 → 총 예상 루프 수: {estimated_batches}회")
start_time = time.time()

# 로그 파일 열기
log_file_path = "chroma_insert_log.txt"
with open(log_file_path, "a", encoding="utf-8") as log:

    # 루프 수행
    while True:
        cursor = conn.cursor()  # 하위에서 컨널팅 오류 바인드 체크
        query = f"""
            SELECT id, name, instruction, role, ingredient, category, inputrecipe, portnum, level, timenum, style
            FROM recipe
            LIMIT {batch_size} OFFSET {offset}
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        if not rows:
            print("\n✅ 모든 레시피 생입 완료!")
            log.write(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 모든 레시피 생입 완료!\n")
            break

        docs, metas, ids = [], [], []

        for row in rows:
            id, name, instruction, role, ingredient, category, inputrecipe, portnum, level, timenum, style = row      
            text = f"{id}: {name}, 재료: {inputrecipe}, 카테고리: {category}, 조리방법: {instruction}, 유형: {style}, 주재료: {ingredient}"
            key = f"{name}|{inputrecipe}"

            if key in existing_keys:
                continue  # 중복 건너떡

            meta = {
                "id": id,
                "name": name,
                "inputrecipe": inputrecipe,
                "category": category,
                "level": level,
                "cook_time": timenum,
                "portnum": portnum,
                "role": role, 
                "instructions": instruction,
                "style": style,
                "ingredient": ingredient,
            }

            docs.append(text)
            metas.append(meta)
            ids.append(f"rec_{id}")
            existing_keys.add(key)

        # 임벤드 및 생입
        if docs:
            embeddings = model.encode(docs).tolist()
            collection.add(
                documents=docs,
                metadatas=metas,
                embeddings=embeddings,
                ids=ids
            )
            total_inserted += len(docs)

            elapsed = time.time() - start_time
            eta = (elapsed / (offset // batch_size + 1)) * (estimated_batches - (offset // batch_size + 1)) / 60

            log_line = f"[{datetime.now().strftime('%H:%M:%S')}] 현재까지 생입된 레시피 수: {total_inserted} | 예상 남은 시간: {eta:.1f}분\n"
            print(log_line.strip())
            log.write(log_line)

            if total_inserted % 10000 < batch_size:
                log.write("[CHECKPOINT] ---- 저장 시점 도달 ----\n")

        offset += batch_size
        
# 연결 종료
cursor.close()
conn.close()


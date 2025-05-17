from sqlalchemy import create_engine
import pandas as pd
import os
from datetime import date, datetime, timedelta
import json
import pymysql

# 날짜 지정
yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
today_str = datetime.today().strftime('%Y-%m-%d')

# SQLAlchemy 엔진 연결
engine = create_engine("mysql+mysqlconnector://dongdong:20250517@192.168.14.47/ai_re")

# 테이블 로드
preference_df = pd.read_sql("SELECT * FROM preference", engine)
similarity_df_original = pd.read_sql("SELECT * FROM similarity", engine)
recipe_df = pd.read_sql("SELECT id, instruction, ingredient, style FROM recipe", engine)

# userNum을 int로 통일
preference_df['userNum'] = pd.to_numeric(preference_df['userNum'])
preference_df['userNum'] = preference_df['userNum'].astype(int)
similarity_df_original['userNum'] = similarity_df_original['userNum'].astype(int)

# 백업 저장
save_dir = "C:/Users/Admin/workspace/backups"
os.makedirs(save_dir, exist_ok=True)
pref_path = os.path.join(save_dir, f"mysql_preference_backup_before_update_{yesterday}.csv")
sim_path = os.path.join(save_dir, f"mysql_similarity_backup_before_update_{yesterday}.csv")
preference_df.to_csv(pref_path, index=False)
similarity_df_original.to_csv(sim_path, index=False)

# 구매 로그 불러오기
log_df = pd.read_sql(f"""
    SELECT userNum, parameter FROM user_logs 
    WHERE logType = 'cartPurchase' AND partitionDate = '{yesterday}'
""", engine)

# 기존 선호도 감쇠 적용
decay_factor = 0.95  # 하루 5%씩 점수 감소
preference_df[['instruction', 'ingredient', 'style']] *= decay_factor

# 가중치 보정
alpha = 0.2
for _, row in log_df.iterrows():
    user = int(row['userNum'])
    try:
        purchased = json.loads(row['parameter'])['레시피']
    except:
        continue
    for recipe in purchased:
        r_id = str(recipe['id'])
        r_attr = recipe_df[recipe_df['id'] == int(r_id)]
        if r_attr.empty:
            continue
        r_attr = r_attr.iloc[0]
        for attr in ['instruction', 'ingredient', 'style']:
            attr_value = r_attr[attr]
            targets = recipe_df[recipe_df[attr] == attr_value]['id'].tolist()
            for tid in targets:
                idx = (preference_df['userNum'] == user) & (preference_df['id'] == tid)
                if idx.any():
                    before = preference_df.loc[idx, attr].values[0]
                    updated = before * (1 - alpha) + 1.0 * alpha
                    preference_df.loc[idx, attr] = updated

# similarity 재계산
preference_df['similarity'] = preference_df[['instruction', 'ingredient', 'style']].mean(axis=1)

# 기존 name, exception과 결합
similarity_df = preference_df[['userNum', 'id', 'similarity']].merge(
    similarity_df_original[['userNum', 'id', 'name', 'exception']],
    on=['userNum', 'id'],
    how='left'
)
similarity_df['partitionDate'] = today_str

# 덮어쓰기
conn = pymysql.connect(
    host='192.168.14.47',
    user='dongdong',
    password='20250517',
    database='ai_re',
    charset='utf8'
)
cursor = conn.cursor()

cursor.execute("DELETE FROM preference")
for _, row in preference_df.iterrows():
    cursor.execute("""
        INSERT INTO preference (userNum, id, instruction, ingredient, style)
        VALUES (%s, %s, %s, %s, %s)
    """, (row['userNum'], row['id'], row['instruction'], row['ingredient'], row['style']))

cursor.execute("DELETE FROM similarity")
for _, row in similarity_df.iterrows():
    cursor.execute("""
        INSERT INTO similarity (userNum, id, name, similarity, exception, partitionDate)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        row['userNum'],
        row['id'],
        row['name'],
        row['similarity'],
        row['exception'],
        row['partitionDate']
    ))

conn.commit()
cursor.close()
conn.close()
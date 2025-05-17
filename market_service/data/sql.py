import mysql.connector
import pandas as pd

# SQL 연결
def get_mysql_connection():
    return mysql.connector.connect(
        host='192.168.14.47',
        user='dongdong',
        password='20250517',
        database='ai_re',
        connection_timeout=3600
    )

# 레시피 테이블 로드
def load_recipes():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    query = '''
        SELECT *
        FROM recipe
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)


# 상품 테이블 로드
def load_product():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    query = '''
        SELECT *
        FROM product
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)


# 선호도 테이블 로드
def load_preference():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    query = '''
        SELECT *
        FROM preference
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)

# 유사도 테이블 로드
def load_similarity():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    query = '''
        SELECT *
        FROM similarity
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)

# 유사도 테이블 로드
def load_similarity():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    query = '''
        SELECT *
        FROM similarity
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)

def load_total_revenues():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    query = '''
        SELECT *
        FROM planning_total_revenues
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)
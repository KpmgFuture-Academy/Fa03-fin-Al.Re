from data import get_mysql_connection
import bcrypt

def authenticate(login_id: str, password: str):
    
    """
    로그인 ID와 비밀번호를 기반으로 사용자를 인증하는 함수.

    Args:
        login_id: 사용자가 입력한 로그인 ID (str)
        password: 사용자가 입력한 비밀번호 (str)

    Returns:
        인증 성공 시: {'userNum': ..., 'userid': ..., 'username': ...} 딕셔너리 반환
        인증 실패 시: None 반환
    """
    
    # 관리자 계정 확인
    if login_id == "admin" and password == "admin1234":
        return {"id": "admin", "name": "관리자", "role": "admin"}

    # MySQL 연결
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)

    # 입력된 login_id에 해당하는 사용자 정보 조회
    cur.execute(
        "SELECT userNum, name, id, passwordhash FROM userinfo WHERE id = %s",
        (login_id,)
    )
    user = cur.fetchone()

    # 커서 및 연결 종료
    cur.close()
    conn.close()

    # 사용자 정보가 존재하고, 비밀번호가 일치하는지 확인
    if user and bcrypt.checkpw(password.encode('utf-8'), user['passwordhash'].encode('utf-8')):
        # 인증 성공 → 필요한 정보만 딕셔너리로 반환
        return {
            "userNum": user['userNum'],
            "userid": user['id'],
            "name": user['name']
        }
    
    # 인증 실패 → None 반환
    return None

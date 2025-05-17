from data import get_mysql_connection
from datetime import datetime, date
import json

def log_event(user_num: str, os_type: str, log_type: str, parameter: dict):

    """
    사용자 활동을 user_logs 테이블에 기록하는 함수.

    Args:
        user_num: 사용자 번호 (str)
        os_type: 사용자 디바이스 OS 타입 (str)
        log_type: 로그 이벤트 타입 (str)
        parameter: 추가 파라미터를 담은 딕셔너리 (dict)

    Returns:
        None
    """
    
    # MySQL 연결
    conn = get_mysql_connection()
    cursor = conn.cursor()

    # 현재 시간 및 날짜 생성
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    part_date = date.today().strftime("%Y-%m-%d")

    # user_logs 테이블에 이벤트 기록
    cursor.execute(
        """
        INSERT INTO user_logs
          (userNum, logType, timestamp, parameter, osType, partitionDate)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            user_num,
            log_type,
            now,
            json.dumps(parameter, ensure_ascii=False),
            os_type,
            part_date
        )
    )

    # 변경사항 커밋 및 연결 종료
    conn.commit()
    cursor.close()
    conn.close()

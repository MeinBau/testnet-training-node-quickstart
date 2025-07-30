import os
import requests

finnhub_api_key = os.environ["FINNHUB_API_KEY"]

def get_stock_quote(symbol: str) -> dict:
    """
    Finnhub API를 사용하여 주식 심볼(symbol)의 실시간 주가를 가져옵니다.
    :param symbol: 조회할 주식 심볼 (예: 'AAPL', 'TSLA')
    :return: API 응답(JSON) 딕셔너리
    """
    if not finnhub_api_key:
        raise EnvironmentError("환경변수 FINNHUB_API가 설정되지 않았습니다.")

    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={finnhub_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ConnectionError(f"API 호출 실패: {response.status_code} - {response.text}")


# Задаем значения констант
RANDOM_STATE = 300625
TEST_SIZE = 0.25
CV = 5

CONTRACT_LINK = 'https://code.s3.yandex.net/datasets/contract_new.csv'
PERSONAL_LINK = 'https://code.s3.yandex.net/datasets/personal_new.csv'
INTERNET_LINK = 'https://code.s3.yandex.net/datasets/internet_new.csv'
PHONE_LINK = 'https://code.s3.yandex.net/datasets/phone_new.csv'

TARGET_COLUMN = "subscriber_left"
INTERVAL_COLUMNS = ['monthly_charges', 'num_of_days']
COLUMNS_TO_DELETE = ['total_charges', 'gender', 'internet_service', 'streaming_tv', 'device_protection']

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
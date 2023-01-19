import requests
import kfp

HOST = 'http://localhost:8080/'
USERNAME = 'user@example.com'
PASSWORD = '12341234'
NAMESPACE = 'kubeflow-user-example-com'

def create_client():

    session = requests.Session()
    response = session.get(HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]

    client = kfp.Client(
        host=f"{HOST}/pipeline",
        cookies=f"authservice_session={session_cookie}",
        namespace=NAMESPACE,
    )
    print(client.list_pipelines())
    return client


import requests

query = "Siapa gubernur Jakarta?"
url = "http://localhost:12345/answer_query"
data = {'query': query}
response = requests.post(url, json=data)

print(response.json()['answer'])

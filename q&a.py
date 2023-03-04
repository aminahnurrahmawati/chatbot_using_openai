import requests

query = "Siapa gubernur Jakarta?"
url = "http://localhost:5000/answer_query"
data = {'query': query}
response = requests.post(url, json=data)

print(response.json()['answer'])
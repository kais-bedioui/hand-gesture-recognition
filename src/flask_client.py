import requests
import base64

# Encode an image in Base64
image_path = "../data/test_data/open_hand/open_hand_1.png"
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Send the POST request
url = "http://localhost:5000/predict"
payload = {"image": encoded_image}
response = requests.post(url, json=payload)

# Print the response
print(response.json())
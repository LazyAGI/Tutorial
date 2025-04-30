import requests
# 当前网页url
url = "https://blog.csdn.net/star_nwe/article/details/141174167"
headers = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/131.0.0.0 Safari/537.36")
}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    with open("webPage.html", "w", encoding='utf-8') as file:
        file.write(response.text)
    print("Webpage downloaded successfully!")
else:
    print(f"Failed to download webpage. Status code: {response.status_code}")

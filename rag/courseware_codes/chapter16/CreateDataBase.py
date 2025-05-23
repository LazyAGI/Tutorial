import re
import requests
import time
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import sqlite3

# 下载文件的函数
def spider(url):
    response = requests.get(url)

    # 检查请求是否成功
    res = {}
    if response.status_code == 200:
        # 解析网页内容
        soup = BeautifulSoup(response.text, 'html.parser')
        elements = soup.find_all(class_='title mathjax')
        for ele in elements:
            res['title'] = ele.get_text().replace("Title:", '')
        elements = soup.find_all(class_='authors')
        authors = []
        for ele in elements:
            links = ele.find_all('a')
            for link in links:
                authors.append(link.get_text())
        res['author'] = ';'.join(authors)
        elements = soup.find_all(class_='tablecell subjects')
        for ele in elements:
            res['subject'] = ele.get_text()
        return res

    else:
        print(f"网页加载失败，状态码：{response.status_code}")


def get_information():
    with open('2024.md')as f:
        texts = f.readlines()[2:102]

    reses = []
    for text in tqdm(texts):
        abs_url = re.findall(r'\[.*?\]\((https?://arxiv.*?)\)', text)
        if len(abs_url) > 0:
            res = spider(abs_url[0])
            reses.append(res)
            time.sleep(5)

    with open('test.txt', 'w')as f:
        json.dump({"data": reses}, f)

def write_into_table():
    with open('test.txt', 'r')as f:
        data = json.load(f)

    data = data['data']

    # 连接到 SQLite 数据库（如果数据库不存在，它会被创建）
    conn = sqlite3.connect('papers.db')
    cursor = conn.cursor()

    # 创建表格
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        author TEXT,
        subject TEXT
    )
    ''')

    # 插入数据
    for paper in data:
        cursor.execute('''
        INSERT INTO papers (title, author, subject)
        VALUES (?, ?, ?)
        ''', (paper['title'], paper['author'], paper['subject']))

    # 提交并关闭连接
    conn.commit()
    conn.close()

if __name__ == "__main__":
    get_information()
    write_into_table()

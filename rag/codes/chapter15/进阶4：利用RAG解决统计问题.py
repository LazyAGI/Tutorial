import sqlite3  # 导入sqlite3模块，用于操作SQLite数据库
import lazyllm

# SQL 概念及数据库构建

# 将LazyLLM路径加入环境变量
# import sys
# sys.path.append("/home/mnt/chenzhe1/Code/LazyLLM")
# # 设置环境变量
# import os
# os.environ["LAZYLLM_SENSENOVA_API_KEY"] = ""
# os.environ["LAZYLLM_SENSENOVA_SECRET_KEY"] = ""

# 1 构建 SQL 数据库
# 如果出现下面的错误表明数据库未清空，可执行下面代码实现清空数据库
# IntegrityError: UNIQUE constraint failed: orders.order_id
# cursor.execute("DELETE FROM orders")  # 删除表中的所有数据
# conn.commit()

# 连接到数据库
# 如果数据库文件'ecommerce.db'不存在，SQLite会自动创建一个新的数据库文件
conn = sqlite3.connect('ecommerce.db')
cursor = conn.cursor()  # 创建一个游标对象，用于执行SQL操作

# 创建 orders 表
# 使用CREATE TABLE语句创建一个新的表，表名为 orders
# IF NOT EXISTS 子句确保如果表已经存在，不会重复创建
cursor.execute('''
CREATE TABLE IF NOT EXISTS orders (
    order_id INT PRIMARY KEY,
    product_id INT,
    product_category TEXT,
    product_price DECIMAL(10, 2),
    quantity INT,
    cost_price DECIMAL(10, 2),
    order_date DATE
)
''')
# 订单ID，作为主键，确保每个订单有唯一标识
# 产品ID
# 产品类别（例如，手机、电脑、电视等）
# 产品价格，保留2位小数
# 购买数量
# 产品成本价格，保留2位小数
# 订单日期

# 插入数据
# 定义一个包含多个订单的列表，每个订单的相关信息（如订单ID、产品ID、价格等）
data = [
    [1, 101, "手机", 1000, 2, 600, "2025/1/1"],
    [2, 102, "手机", 1200, 1, 700, "2025/1/2"],
    [3, 103, "电脑", 5000, 1, 3500, "2025/1/3"],
    [4, 104, "电脑", 4500, 3, 3000, "2025/1/4"],
    [5, 105, "电视", 3000, 1, 1800, "2025/1/5"],
    [6, 106, "电视", 3500, 2, 2000, "2025/1/6"]
]

# 执行插入操作，将每一条数据插入到 'orders' 表中
# cursor.executemany() 方法用于执行多个INSERT语句，批量插入数据
cursor.executemany('''
INSERT INTO orders (order_id, product_id, product_category, product_price, quantity, cost_price, order_date)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', data)

# 提交更改
# 使用 conn.commit() 提交事务，将所有插入操作保存到数据库中
conn.commit()

# 关闭连接
# 在完成操作后，关闭数据库连接，释放资源
conn.close()

# 2.查看数据库中的内容
# 连接到数据库
conn = sqlite3.connect('ecommerce.db')
cursor = conn.cursor()  # 创建一个游标对象，用于执行SQL操作

# 执行查询操作，查看 orders 表中的所有数据
cursor.execute('SELECT * FROM orders')

# 获取所有查询结果
rows = cursor.fetchall()

# 打印查询结果
for row in rows:
    print(row)  # 打印每一行数据

# 关闭连接
conn.close()

# 3.Text2SQL
prompt = '''给定以下数据库表结构：
"CREATE TABLE IF NOT EXISTS orders (
order_id INT PRIMARY KEY,
product_id INT,
product_category TEXT,
product_price DECIMAL(10, 2),
quantity INT,
cost_price DECIMAL(10, 2),
order_date DATE );
请你根据用户的要求，编写相应的SQL查询语句'''
llm = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-5").prompt(lazyllm.ChatPrompter(instruction=prompt))
print(llm("请给我每个品类的盈利汇总，按照盈利从高到低排序"))

# 连接到数据库
conn = sqlite3.connect('ecommerce.db')  # 连接到刚创建的数据库文件
cursor = conn.cursor()  # 创建一个游标对象，用于执行SQL操作

# 执行给定的SQL查询语句
cursor.execute('''
SELECT
    product_category,
    SUM((product_price * quantity) - (cost_price * quantity)) AS total_profit
FROM
    orders
GROUP BY
    product_category
ORDER BY
    total_profit DESC;
''')

# 获取查询结果
rows = cursor.fetchall()

# 打印查询结果
for row in rows:
    print(row)  # 打印每一行数据

# 关闭连接
conn.close()

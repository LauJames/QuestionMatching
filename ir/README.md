# Indexing the questions (ID, content, primary_ID) using Elasticsearch
## Data process
- step1 配置es configuration: 修改config.py
- step2 为数据建立索引 python ir/index.py
- step3 设置检索策略并根据question检索question内容、主问题ID
- step4 利用主问题ID，依据主问题列表。找到主问题内容
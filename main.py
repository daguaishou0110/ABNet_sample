# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import WordCloud

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

data = pd.read_csv('all.csv',encoding='gbk')
column_data = data['Abstract']
text = ' '.join(column_data.astype(str))
stop_words = set(['the', 'and', 'is', 'in', 'to', 'of', 'it', 'with', 'as','one','two','this','paper','algorithm','both','method',
                  'proposed', 'a', 'an', 'the', 'and', 'but', 'or', 'so',
                  'for', 'of', 'to', 'in', 'on',
    'at', 'by', 'with', 'from', 'as', 'this', 'that', 'these', 'those', 'such',
    'be', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'can', 'could',
    'may', 'might', 'will', 'would', 'should', 'must', 'it', 'its', 'we', 'our',
    'us', 'you', 'your', 'they', 'their', 'them', 'he', 'him', 'his', 'she',
    'her', 'hers', 'itself', 'himself', 'herself', 'themselves', 'not', 'no',
    'nor', 'don', 'don\'t', 'doesn', 'doesn\'t', 'didn', 'didn\'t', 'isn',
    'isn\'t', 'aren', 'aren\'t', 'wasn', 'wasn\'t', 'weren', 'weren\'t', 'hasn',
    'hasn\'t', 'haven', 'haven\'t', 'hadn', 'hadn\'t', 'won', 'won\'t', 'wouldn',
    'wouldn\'t', 'can\'t', 'cannot', 'shouldn', 'shouldn\'t', 'mustn', 'mustn\'t',
    'about', 'above', 'after', 'again', 'against', 'all', 'am', 'any', 'because',
    'before', 'below', 'between', 'both', 'down', 'during', 'each', 'fewer',
    'further', 'here', 'how', 'more', 'most', 'much', 'now', 'other', 'over',
    'same', 'some', 'such', 'than', 'then', 'there', 'these', 'they', 'this',
    'those', 'through', 'under', 'until', 'very', 'when', 'where', 'which',
    'while', 'who', 'why', 'you', 'your', 'the', 'and', 'is', 'in', 'to', 'of', 'it', 'with', 'as', 'this', 'that',
    'for', 'we', 'are', 'on', 'by', 'an', 'be', 'can', 'which', 'has', 'been',
    'from', 'at', 'or', 'not', 'but', 'also', 'paper', 'proposed', 'method',
    'algorithm', 'results', 'show', 'our', 'based', 'using', 'two', 'one',
    'new', 'approach', 'problem', 'model', 'data', 'analysis', 'study', 'work',
    'result', 'experimental', 'performance', 'proposed', 'system', 'technique',
    'methodology', 'evaluation', 'proposed', 'design', 'implementation','various','instance',
                  'application','deep','However','set','methods','due','large','only','existing','state','novel','state','first',
                  'experiment','real word','sample','art','better','do','many'])
wordcloud = WordCloud(width=800, height=400, background_color='white',max_font_size=72,stopwords=stop_words).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()
wordcloud.to_file("few-shot 1223.png")

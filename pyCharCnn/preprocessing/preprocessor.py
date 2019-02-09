#encoding:utf-8
import re

class Preprocessor(object):
    def __init__(self,min_len=2,stopwords_path = None):
        self.min_len = min_len
        self.stopwords_path = stopwords_path
        self.reset()

    # 加载停用词
    def reset(self):
        if self.stopwords_path:
            with open(self.stopwords_path,'r') as fr:
                self.stopwords = {}
                for line in fr:
                    word = line.strip(' ').strip('\n')
                    self.stopwords[word] = 1
    # 大写转化为小写
    def text_lower(self,text):
        return text.lower()
    
    # 移除标签标志
    def remove_hashtags(self,text):
        clean_text = re.sub(r'#[A-Za-z0-9_]+', "", text)
        return clean_text
    
    # 移除url
    def remove_urls(self,text):
        clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        return clean_text
    
    # 去除长度小于min_len的文本
    def clean_length(self,text):
        if len([x for x in text]) >= self.min_len:
            return text

    # 全角转化为半角
    def full2half(self,text):
        ret_str = ''
        for i in text:
            if ord(i) >= 33 + 65248 and ord(i) <= 126 + 65248:
                ret_str += chr(ord(i) - 65248)
            else:
                ret_str += i
        return ret_str

    #去除停用词
    def remove_stopword(self,text):
        words = text.split()
        x = [word for word in words if word not in self.stopwords]
        return " ".join(x)

    # 提取中文
    def get_china(self,text):
        zhmodel = re.compile("[\u4e00-\u9fa5]")
        words = [x for x in text if zhmodel.search(x)]
        return ''.join(words)
    
    # 移除数字
    def remove_numbers(self,text):
        words = text.split()
        x = [re.sub('\d+','',word) for word in words]
        return ' '.join([w for w in x if w !=''])

    def remove_whitespace(self,text):
        x = ''.join([x for x in text if x !=' ' or x !='' or x!='  '])
        return x

    # 主函数
    def __call__(self, text):
        x = text.strip('\n')
        x = self.text_lower(x)
        x = self.remove_urls(x)
        return x


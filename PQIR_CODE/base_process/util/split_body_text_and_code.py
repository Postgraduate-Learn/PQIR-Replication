import re
import html

CODE_LENGTH_THRESHOLD = 50


def clean_html_with_links(raw_html):
    text = html.unescape(raw_html)
    # 处理<a>标签：变成“描述 (链接)”形式
    text = re.sub(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r'\2 (\1)', text)
    # 去掉其他HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 清理多余的空白
    text = re.sub(r'\n+', '\n', text).strip()
    return text


# 提取body和code
def extract_body_and_code(body_text):
    """从body中提取文本"""
    # 正则表达式匹配 <pre> 标签及其包裹的 <code> 内容
    code_pattern = r'(<pre(?:\s+[^>]*?)?><code>(.*?)</code></pre>)'
    matches = re.findall(code_pattern, body_text, re.DOTALL)

    long_codes = []

    for full_pre_code_block, code_content in matches:
        code_content = code_content.strip()
        if len(code_content) > CODE_LENGTH_THRESHOLD:
            long_codes.append(code_content)
            # 长代码整体删除
            body_text = body_text.replace(full_pre_code_block, '')
        else:
            # 短代码直接嵌回body
            body_text = body_text.replace(full_pre_code_block, f'\n{code_content}\n')

    # 对移除长代码后的 body_text 进行 HTML 清洗在
    cleaned_body = clean_html_with_links(body_text)

    # 拼接所有长代码
    full_code_text = '\n\n'.join(long_codes) if long_codes else ''

    return cleaned_body, full_code_text

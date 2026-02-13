import os
from lxml import etree
from conf import conf
from base_process.preprocess.post import Post
from base_process.util.writer import writerObjs2xml

patchsize = 50000

def getQuestions(filtered_questions_path):
    """筛选出Score>5 and ViewCount >500的数据"""
    # 用于记录记录问题的数量
    qc_dict = 0
    # 用于记录问题的id
    qids_dict = []
    # 用于记录每个问题的完整对象
    qs_dict = []

    context = etree.iterparse(conf.post_xml, events=('end',), tag='row')

    for event, row in context:
        if event == 'end' and row.tag == 'row':
            if row.get('PostTypeId') == '1' and int(row.get('Score')) > 5 and int(row.get('ViewCount')) > 500:
                qc_dict += 1
                qids_dict.append(row.get('Id'))
                qs_dict.append(Post(row))
                c = qc_dict

                if c % patchsize == 0:
                    print("Number of filtered questions:", c)
        row.clear()
        while row.getprevious() is not None:
            del row.getparent()[0]
    del context

    if len(qids_dict) > 0:
        q_xml = os.path.join(filtered_questions_path, 'questions.xml')
        writerObjs2xml(q_xml, qs_dict, 'Question')
    print("Number of filtered questions:", qc_dict)
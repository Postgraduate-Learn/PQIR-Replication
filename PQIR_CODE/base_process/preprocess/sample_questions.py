import os
import random
import xml.etree.ElementTree as ET
from base_process.util.writer import writerObjs2xml
from base_process.preprocess.post import Post
from base_process.util.writer import writer_file_excel

def Extract(xml_file, sample_xml_path, sample_excel_path, count, sample_size=384):
    """对筛选后的数据进行随机抽样"""
    samples_data = []
    xml_file = os.path.join(xml_file, 'Questions')

    files_list = os.listdir(xml_file)

    tree = ET.parse(os.path.join(xml_file, files_list[0]))
    root = tree.getroot()
    print(f'对{files_list[0]}进行抽样，抽取{sample_size}')
    samples_per_file = len(root)

    # 生成随机索引用于抽样和删除
    sample_indices = random.sample(range(samples_per_file), sample_size)
    sample_indices.sort(reverse = True)

    for sample_index in sample_indices:
        samples_data.append(Post(root[sample_index]))
        root.remove(root[sample_index])
    root.set('count', str(len(root)))
    tree.write(os.path.join(xml_file, files_list[0]), encoding = 'utf-8', xml_declaration = True)

    if not os.path.exists(sample_xml_path):
        os.makedirs(sample_xml_path)
    sample_xml_path = os.path.join(sample_xml_path, "sample_" + str(count) + ".xml")
    writerObjs2xml(sample_xml_path, samples_data, 'Samples')
    if not os.path.exists(sample_excel_path):
        os.makedirs(sample_excel_path)
    sample_excel_path = os.path.join(sample_excel_path, "sample_" + str(count) + ".xlsx")
    writer_file_excel(sample_excel_path, samples_data)
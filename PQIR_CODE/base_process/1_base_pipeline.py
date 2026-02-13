import time
import os
from base_process.preprocess.filter_high_quality_questions import getQuestions
from conf import conf
from base_process.preprocess.sample_questions import Extract

def base_pipeline():
    filtered_questions_path = os.path.join(conf.experiment_dir, "Questions")
    if not os.path.exists(filtered_questions_path):
        os.makedirs(filtered_questions_path)
    sample_path = os.path.join(conf.experiment_dir, "sample_xml")
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    sample_title_path = os.path.join(conf.experiment_dir, "sample_excel")
    if not os.path.exists(sample_title_path):
        os.makedirs(sample_title_path)

    print('====== filter questions ======\n')
    start_time = time.perf_counter()
    getQuestions(filtered_questions_path)
    end_time = time.perf_counter()
    print('Running time:\n', end_time - start_time)

    print('====== Start sampling ======\n')
    start_time = time.perf_counter()
    for index in range(3):
        print(f'Sampling {index + 1}')
        Extract(conf.experiment_dir, sample_path, sample_title_path, index + 1)
    end_time = time.perf_counter()
    print('Running time:\n', end_time - start_time)


if __name__ == '__main__':
    base_pipeline()



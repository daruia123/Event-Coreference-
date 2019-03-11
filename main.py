#!/usr/bin/env python
import logging
import os
import sys



# os.environ.get()得到某个环境变量的值
if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO


# os.pardir:指向上一级目录，os.path.abspath:或得当前文件的路径，os.path.dirname:或得当前路径的上级路径
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
#sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
# asctime: 日志发生的时间，levelname:日志级别， name:日志器名称， message：日志内容from allennlp.models.archival import archive_model, load_archive, Archive


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)


from allennlp.commands import main  # pylint: disable=wrong-import-position

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        #sys.argv = ['main.py', 'evaluate',  'debug/model.tar.gz', '--evaluation-data-file', "/data/rywu/data/coref_english/Kbp_test_english.txt", '--cuda-device', '1']
        sys.argv = ['main.py', 'train', 'training_config/coref_english.json', '-s', 'debug']
        #sys.argv = ['main.py', 'test', 'debug/model.tar.gz', "/data/rywu/data/coref_english/Kbp_test_english.txt", "--output-file", "output.txt", "--cuda-device", "2"]
    main()


## ChineseDiscourseParser

A Chinese discourse parser based on CDTB

##### Requirements:
- Python == 3.6
- pytorch >= 0.4
- sklearn == 0.20
- executable 'java' in system path
- ... (see `requirements.txt` for detail)

##### Install

```shell
git clone https://github.com/oisc/ChineseDiscourseParser --recursive
cd ChineseDiscourseParser
pip3 install -r requirements.txt
```

See [pyltp](https://github.com/HIT-SCIR/pyltp)'s documentation and download pyltp's models into `pub/pyltp_models`

##### Run Example:

```shell
python3 parse.py sample.txt sample.xml [--draw]
```

The script above take 3 parameters `source file`, `path to save parsing xml result` 
and an optional `--draw` which can draw the discourse tree if you are in GUI environment and have tkinter installed.
Every line in source file should be a paragraph and will be parsing into a CDT-styled discourse tree.

##### Issues:

Module `segmenter.svm` depends on berkeley parser which is employed to generate constituent parsing information.
However, berkeley parser may behave unstanble when handleing long sentences, resulting a timeout error.

from deepdoctection.analyzer import get_dd_analyzer
import os


if __name__=="__main__":
    path = "/Users/janismeyer/anr.pdf"
    analyzer= get_dd_analyzer(ocr=False)
    df =analyzer.analyze(path=path)
    for dp in df:
        pass
        #print(dp)



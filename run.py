from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from pprint import pprint
from pathlib import Path
from math import floor, ceil
import logging
import re

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from rouge import Rouge
from bert_score import score
import pandas as pd
import numpy as np
import torch

import nltk
nltk.download('punkt')

def preprocess_transcript(text):
    def match_speaker(t):
        if len(t) >= 1:
            if t[0] == "[":
                return t[1:(list(t).index("]"))]
            else:
                return None
        else:
            return None

    if not text.rstrip():
        return None, None

    # Remove "<text>"", e.g., <unrecognisable_speech>, <ehm>
    text = re.sub("\<\S*\>", "", text)
    # Remove "(??)""
    text = text.replace("(??)", "")
    # Replace &nbsp with spaces
    text = text.replace("&nbsp;", " ")
    speaker = match_speaker(text)
    if speaker:
        return speaker, text[len(speaker)+2:]
    else:
        return None, text

# Very badly written functions; change them if necessary
def process_minutes(text):
    def remove_symbols(t):
        return t.replace("•", "").replace("-", "").strip()
    text = list(map(remove_symbols, filter(lambda x: len(x) >1 and (x[0] == "•" or x[0] == "-"), text.split("\n"))))

    return text

def extract_number(n):
    return n.split("_")[0][-3:]

# KL is too slow; don't use that
summarizer_dict = {"LSA": LsaSummarizer, "Lex_rank": LexRankSummarizer, "SumBasic": SumBasicSummarizer, "Luhn": LuhnSummarizer, "Text_rank": TextRankSummarizer} # "KL": KLSummarizer,}
LANGUAGE = "english"

filepath = "/content/gdrive/My Drive/data/automin/"
file_names = [["transcript108_en", "minutes108_en"], ["transcript112_en", "minutes112_en"], ["transcript116_en", "minutes116_en"],]

for transcript_name, minutes_name in file_names:

    print(extract_number(transcript_name))
    #print("\hline")

    raw = Path(filepath+transcript_name).read_text()
    write_buffer = ""
    last_speaker = None

    for line in raw.split("\n"):
        speaker, proceeded = preprocess_transcript(line)
        if not proceeded:
            continue
        if speaker:
            write_buffer += ("[" + speaker + "] " + proceeded + "\n")
            last_speaker = speaker
        else:
            write_buffer += ("[" + str(last_speaker) + "] " + proceeded + "\n")
    
    with open(filepath + transcript_name + "_p", "w") as f:
        f.write(write_buffer)

    gt = process_minutes(Path(filepath+minutes_name).read_text())

    for summarizer_name, summarizer_instance in summarizer_dict.items():


        parser = PlaintextParser.from_file(filepath + transcript_name, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)

        summarizer = summarizer_instance(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)
        summary = []
        for sentence in summarizer(parser.document, len(gt)):
            summary.append(str(sentence))

        rouge = Rouge()
        rouge_score = rouge.get_scores([" ".join(summary)], [" ".join(gt)])
        P, R, F1 = score([" ".join(summary)], [" ".join(gt)], lang="en", verbose=True)
        print(summary)

        print("{} & {} & {} & {}".format(summarizer_name, rouge_score[0]['rouge-1']['f'], rouge_score[0]['rouge-2']['f'], rouge_score[0]['rouge-l']['f']))
        #print("\hline")
        print("{} & {} & {} & {}".format(summarizer_name, torch.mean(P), torch.mean(R), torch.mean(F1)))
        #print("\hline")
        #print("Summary score on transcript_{} using {} as summarizer:".format(extract_number(transcript_name), summarizer_name))
        #print(rouge_score[0])


from random import sample
for transcript_name, minutes_name in file_names:

    print(extract_number(transcript_name))

    transcript = Path(filepath+transcript_name).read_text().split("\n")
    gt = process_minutes(Path(filepath+minutes_name).read_text())

    summary = sample(transcript, 10)
    summarizer_name = "RandomSampler"

    rouge = Rouge()
    rouge_score = rouge.get_scores([" ".join(summary)], [" ".join(gt)])
    P, R, F1 = score([" ".join(summary)], [" ".join(gt)], lang="en", verbose=False)
    print(summary)
    print(gt)

    print("{} & {} & {} & {}".format(summarizer_name, rouge_score[0]['rouge-1']['f'], rouge_score[0]['rouge-2']['f'], rouge_score[0]['rouge-l']['f']))
    #print("\hline")
    print("{} & {} & {} & {}".format(summarizer_name, torch.mean(P), torch.mean(R), torch.mean(F1)))
    #print("\hline")
    #print("Summary score on transcript_{} using {} as summarizer:".format(extract_number(transcript_name), summarizer_name))
    #print(rouge_score[0])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

for transcript_name, minutes_name in file_names:

    print(extract_number(transcript_name))
    #print("\hline")

    raw = Path(filepath+transcript_name).read_text()
    write_buffer = ""
    corpus = []
    speakers = []
    last_speaker = None

    for line in raw.split("\n"):
        speaker, proceeded = preprocess_transcript(line)
        if not proceeded:
            continue
        if speaker:
            speakers.append(speaker)
            write_buffer += ("[" + speaker + "] " + proceeded + "\n")
            last_speaker = speaker
        else:
            speakers.append(last_speaker)
            write_buffer += ("[" + str(last_speaker) + "] " + proceeded + "\n")
        corpus.append(proceeded)

    with open(filepath + transcript_name + "_p", "w") as f:
        f.write(write_buffer)    
   
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    #print(X.toarray())
    #print(len(vectorizer.get_feature_names()))
    tfidf_df = pd.DataFrame(X, columns=vectorizer.get_feature_names())
    data_only = tfidf_df.copy(True)
    tfidf_df["_Speaker"] = speakers
    tfidf_df["_raw"] = corpus

    gt = process_minutes(Path(filepath+minutes_name).read_text())

    kmeans = KMeans(n_clusters=len(gt), random_state=0).fit(data_only)
    tfidf_df["_label"] = kmeans.labels_

    print(tfidf_df.head())
    
    for summarizer_name, summarizer_instance in summarizer_dict.items():


        parser = PlaintextParser.from_file(filepath + transcript_name + "_p", Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)

        summarizer = summarizer_instance(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)
        summary = []
        for sentence in summarizer(parser.document, len(gt)):
            summary.append(str(sentence))

        rouge = Rouge()
        rouge_score = rouge.get_scores([" ".join(summary)], [" ".join(gt)])
        print(" ".join(summary))
        P, R, F1 = score([" ".join(summary)], [" ".join(gt)], lang="en", verbose=True)
        print(summary)

        print("{} & {} & {} & {}".format(summarizer_name, rouge_score[0]['rouge-1']['f'], rouge_score[0]['rouge-2']['f'], rouge_score[0]['rouge-l']['f']))
        #print("\hline")
        print("{} & {} & {} & {}".format(summarizer_name, torch.mean(P), torch.mean(R), torch.mean(F1)))
        #print("\hline")
        #print("Summary score on transcript_{} using {} as summarizer:".format(extract_number(transcript_name), summarizer_name))
        #print(rouge_score[0])


def _round(x: float):
    if x - floor(x) >= 0.5:
        return ceil(x)
    else:
        return floor(x)

for transcript_name, minutes_name in file_names:

    print(extract_number(transcript_name))
    #print("\hline")

    raw = Path(filepath+transcript_name).read_text()
    write_buffer = ""
    corpus = []
    speakers = []
    last_speaker = None

    for line in raw.split("\n"):
        speaker, proceeded = preprocess_transcript(line)
        if not proceeded:
            continue
        if speaker:
            speakers.append(speaker)
            write_buffer += ("[" + speaker + "] " + proceeded + "\n")
            last_speaker = speaker
        else:
            speakers.append(last_speaker)
            write_buffer += ("[" + str(last_speaker) + "] " + proceeded + "\n")
        corpus.append(proceeded)

    with open(filepath + transcript_name + "_p", "w") as f:
        f.write(write_buffer)    
   
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    #print(X.toarray())
    #print(len(vectorizer.get_feature_names()))
    tfidf_df = pd.DataFrame(X, columns=vectorizer.get_feature_names())
    data_only = tfidf_df.copy(True)
    tfidf_df["_Speaker"] = speakers
    tfidf_df["_raw"] = corpus

    gt = process_minutes(Path(filepath+minutes_name).read_text())

    kmeans = KMeans(n_clusters=(len(gt)), random_state=0).fit(data_only)
    tfidf_df["_label"] = kmeans.labels_
    _labels = sorted(list(set(kmeans.labels_)))

    print(tfidf_df.head())


    n_summ_d = {}
    total_num = len(gt)

    for _id in _labels:
        tmp = tfidf_df[tfidf_df._label == _id]["_raw"].tolist()
        with open(filepath + transcript_name + "_p" + str(_id), "w") as f:
            f.write("\n".join(tmp))
        n_summ_d[_id] = min(total_num, _round(len(gt) * len(tmp) / len(tfidf_df)))
        total_num -= n_summ_d[_id]
    
    if total_num > 0:
        for _id in _labels:
            n_summ_d[_id] += 1
            total_num -= 1
            if total_num == 0:
                break
    
    print(total_num, sum([v for k, v in n_summ_d.items()]))

    for summarizer_name, summarizer_instance in summarizer_dict.items():

        summary = []

        for _id in _labels:

            parser = PlaintextParser.from_file(filepath + transcript_name + "_p" + str(_id), Tokenizer(LANGUAGE))
            stemmer = Stemmer(LANGUAGE)

            summarizer = summarizer_instance(stemmer)
            summarizer.stop_words = get_stop_words(LANGUAGE)


            for sentence in summarizer(parser.document, n_summ_d[_id]):
                summary.append(str(sentence))

        rouge = Rouge()
        print(len(summary), len(gt), n_summ_d)
        rouge_score = rouge.get_scores([" ".join(summary)], [" ".join(gt)])
        P, R, F1 = score([" ".join(summary)], [" ".join(gt)], lang="en", verbose=True)
        #P, R, F1 = score(" ".join(summary), " ".join(gt), lang="en", verbose=True)
        print(summary)

        print("{} & {} & {} & {}".format(summarizer_name, rouge_score[0]['rouge-1']['f'], rouge_score[0]['rouge-2']['f'], rouge_score[0]['rouge-l']['f']))
        #print("\hline")
        print("{} & {} & {} & {}".format(summarizer_name, torch.mean(P), torch.mean(R), torch.mean(F1)))
        #print("\hline")
        #print("Summary score on transcript_{} using {} as summarizer:".format(extract_number(transcript_name), summarizer_name))
        #print(rouge_score[0])
import numpy as np
from bert_serving.client import BertClient

prefix_q = '##### **Q:** '
topk = 5

with open('README.md') as fp:
    questions = [v.replace(prefix_q, '').strip() for v in fp if v.strip() and v.startswith(prefix_q)]
    print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))

bc = BertClient(port=4000, port_out=4001)
doc_vecs = bc.encode(questions)

while True:
    query = input('your question: ')
    query_vec = bc.encode([query])[0]
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], questions[idx]))
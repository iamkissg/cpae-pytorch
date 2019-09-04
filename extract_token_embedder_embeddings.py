import sys
import logging

from allennlp.models.archival import load_archive

from cpae.models import *

archive_path, saveto_path = sys.argv[1], sys.argv[2]
archive = load_archive(archive_path)


vocab = archive.model.vocab.get_token_to_index_vocabulary('definition')
embedding_matrix = archive.model.text_embedder.token_embedder_tokens.weight.data.cpu().numpy()

total_vec = len(vocab)
vector_size = embedding_matrix.shape[1]
logging.info("storing %sx%s projection weights into %s", total_vec,
             vector_size, saveto_path)

assert (len(vocab), vector_size) == embedding_matrix.shape, \
    f'{(len(vocab), vector_size)}, {embedding_matrix.shape}'

with open(saveto_path, 'w') as fout:
    # store in sorted order: most frequent words at the top
    for word, index in sorted(vocab.items(), key=lambda item: item[1]):
        row = embedding_matrix[index]
        fout.write("%s %s\n" % (word, ' '.join(repr(val) for val in row)))

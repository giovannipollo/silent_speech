import os
import sys
import json
import urllib.request
from tqdm import tqdm
from read_emg import EMGDataset
from data_utils import TextTransform
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('lm_directory', 'KenLM', 'directory for lexicon and LM files')


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_lexicon(vocab, output_file):
    with open(output_file, 'w', encoding='utf-8') as fout:
        for word in sorted(vocab):
            chars = list(word)
            fout.write(f"{word} " + " ".join(chars) + " |\n")


def extract_vocab(dataset):
    vocab = set()
    text_transform = TextTransform()
    for directory_info, idx in dataset.example_indices:
        with open(os.path.join(directory_info.directory, f'{idx}_info.json')) as f:
            info = json.load(f)
        text = text_transform.clean_text(info['text'])
        vocab.update(text.split())
    return vocab


if __name__ == '__main__':
    FLAGS(sys.argv)

    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    testset = EMGDataset(test=True)

    vocab = set()
    for ds in (trainset, devset, testset):
        vocab |= extract_vocab(ds)

    os.makedirs(FLAGS.lm_directory, exist_ok=True)
    lexicon_path = os.path.join(FLAGS.lm_directory, 'gaddy_lexicon.txt')
    get_lexicon(vocab, lexicon_path)
    print(f'Lexicon written with {len(vocab)} words to {lexicon_path}')

    url = "https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/lm.bin"
    lm_path = os.path.join(FLAGS.lm_directory, 'lm.bin')

    if not os.path.exists(lm_path):
        print(f'Downloading KenLM language model...')
        with DownloadProgressBar(unit='B', unit_scale=True, unit_divisor=1024,
                                 miniters=1, desc='lm.bin') as t:
            urllib.request.urlretrieve(url, lm_path, reporthook=t.update_to)
    
    print('KenLM files ready')
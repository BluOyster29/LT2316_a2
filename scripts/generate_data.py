from lxml import etree
import re
import nltk.data
import os
import csv
import argparse
from tqdm import tqdm

def get_debate_text(debate_path, filename, data_path):
    '''
    Filter the text of each speech and label pairwise the sentences with 'same' or 'change'.
    Write the new files to data_path.
    :param debate_path: path to debate files
    :param filename: filename of debate
    :param data_path: path where the new files should be stored
    '''

    # get data from debate
    with open(os.path.join(debate_path, filename), 'rb') as file:
        debate = file.read()

    # parser
    parser = etree.XMLParser(encoding='ISO-8859-1')

    # get root element
    root = etree.fromstring(debate, parser=parser)

    # get speech elements
    speech_elems = [child for child in root.getchildren() if child.tag == 'speech']

    # need this for stripping the tags out of the text
    clean = re.compile('<.*?>')

    # all sentences with speaker id as tupel (sentence, speaker_id)
    all_sent = []

    # sentence detector to separate text into sentences
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # get the whole speech text without any tags
    for elem in speech_elems:
        # get clean text
        elem.text = re.sub(clean, '', etree.tostring(elem).decode("ISO-8859-1"))

        # get speaker id
        speaker_id = elem.attrib['speakerid'] if 'speakerid' in elem.attrib else 'nospeaker'

        # list of sentences in this speech tag
        list_sent = sent_detector.tokenize(elem.text.strip())

        # put tuple of (sentence, speaker_id) in list
        for sentence in list_sent:
            all_sent.append((sentence, speaker_id))

    return all_sent

if __name__== "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--debate_path', type=str, help='Path to dabates xml files.', required=True)

    parser.add_argument('--data_path', type=str, help='Name of folder to which the new file should be written.', default='data')

    args = parser.parse_args()

    # get arguments
    DEBATE_PATH = args.debate_path
    DATA_PATH = args.data_path

    # create new folder for the instances files
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)


    same = 0
    change = 0

    # write sentences to file
    with open(os.path.join(DATA_PATH, 'debates_sents.csv'), 'w') as csv_file:
        csv_out = csv.writer(csv_file)

        print("Writing instances to file '%s'." % (DATA_PATH + "debates_sents.csv"))

        # write (sentence, next sentence, label) to file

        csv_out.writerow(['sent1', 'sent2', 'label', 'speaker_id'])

        # iterate over debate files
        for file in tqdm(os.listdir(DEBATE_PATH)):
            if file.endswith('.xml'):
                all_sent = get_debate_text(DEBATE_PATH, file, DATA_PATH)

                for i in range(len(all_sent) - 1):

                    sent = all_sent[i][0]
                    speaker_id = all_sent[i][1]

                    sent_next = all_sent[i + 1][0]
                    speaker_id_next = all_sent[i + 1][1]

                    # label data
                    # 0: same, 1: change
                    # check if there was a change in the speaker
                    if speaker_id == speaker_id_next:
                        label = 0
                        same += 1
                    else:
                        label = 1
                        change += 1

                    # write (sentence, next sentence, label) to file
                    csv_out.writerow((sent, sent_next, label, speaker_id_next))

    print('same: ', same)
    print('change: ', change)
    print('Wrote instances in files to %s.' % DATA_PATH)

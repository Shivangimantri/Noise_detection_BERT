import re
from collections import Counter
from string import punctuation

import nltk
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from bs4 import BeautifulSoup

nltk.download('punkt', quiet=True)
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def model_extract_junk(tickets, min_junk_frequency):
    """
    Split all tickets in chunks, detect the junk ones
    :param tickets:
    :param min_junk_frequency: min occurrences of a chunk in a partner's set of tickets to be classified as junk
    :return:
    """
    junk = {}  # distinct junks for each partner
    map_chunk_ticket = {}  # maps chunks to message_id that contain them for each partner

    for account_id in pd.unique(tickets["account_id"]):
        chunks_smb_all = []
        junk_smb = []
        map_chunk_ticket[account_id] = {}
        tickets_smb = tickets
        for index, row in tickets_smb.iterrows():
            chunks_in_row = chunkenizer(str(row["message_body_raw"]))
            chunks_smb_all += chunks_in_row
            for chunk in chunks_in_row:
                if chunk in map_chunk_ticket[account_id].keys():
                    map_chunk_ticket[account_id][chunk].append(
                        row.conversation_id)
                else:
                    map_chunk_ticket[account_id][chunk] = [row.conversation_id]
        chunks_frequency = Counter(chunks_smb_all)
        for chunk in chunks_frequency.keys():
            if chunks_frequency[chunk] >= min_junk_frequency:
                junk_smb.append(chunk)
        junk[account_id] = junk_smb

    return junk, map_chunk_ticket


def get_co_occurence_matrix(junk_smb, map_chunk_ticket_smb, tickets):
    """Compute matrix of co-occurence of junk phrases"""
    messages = list(tickets["conversation_id"])
    n_messages = len(messages)
    n_junk = len(junk_smb)

    occurence = np.zeros((n_junk, n_messages))

    for i, junk in enumerate(junk_smb):
        for message in map_chunk_ticket_smb[junk]:
            # At least one appearance of this junk in this ticket
            occurence[i, messages.index(message)] = 1.

    co_occurence = occurence.dot(occurence.T)
    return co_occurence, junk_smb


def chunkenizer(text):
    """
    Split a ticket text into chunks
    :return: list of chunks
    """
    chunks = []
    clean_text = BeautifulSoup(text, features="html.parser").get_text()
    lines = nltk.tokenize.blankline_tokenize(clean_text)

    for line in lines:
        # Hide emails and links with placeholder
        emails_and_links = re.findall(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
            line)
        for el in emails_and_links:
            line = line.replace(el, 'LINK_HIDDEN.')

        # Split at each punctuation
        separator = '000x000000e&l'
        line = line.strip()
        line = line.replace('\xa0', '')
        line = line.replace(';', ';' + separator)
        line = line.replace('...', '...' + separator)
        line = line.replace(':', ':' + separator)
        line = line.replace('?', '?' + separator)
        line = line.replace('!', '!' + separator)
        line = line.replace('\n', separator)
        line = line.replace('\r', separator)
        line = re.sub(r"[.]+(?=[^0-9])?", "." + separator, line)
        line_chunks = sent_tokenizer.tokenize(line)

        for i, chunk in enumerate(line_chunks):
            this_line_chunks = [c.strip() for c in chunk.split(
                '000x000000e&l')]  # Strip all chunks
            chunks.extend([c for c in this_line_chunks if len(c) > 0])

    return chunks


def get_new_junk(junk_new, junk_old):
    """Difference of old junk dictionary with new junk dictionary"""
    junk_diff = {}
    for account_id in junk_new.keys():  # /!\ Do we want partners that are not in the new but in old junk list?
        if account_id in junk_old.keys():
            junk_new_smb = set(junk_new[account_id])
            junk_old_smb = set(junk_old[account_id])
            junk_diff_smb = junk_new_smb - junk_old_smb
            junk_diff[account_id] = list(junk_diff_smb)
    return junk_diff


def display_junk(junk_dict, map_junk_ticket):
    """Display the junk phrases and how often they appeared"""
    for account_id in junk_dict.keys():
        print("")
        print(account_id)
        for junk in junk_dict[account_id]:
            print("  Appeared {} times: {}".format(
                len(map_junk_ticket[account_id][junk]), junk))


def get_account_junk(
        acc_df,
        account_id,
        min_junk_frequency=5,
        min_cooccurence_ratio=0.8,
        min_component_size=5,
        min_detection_percent=1):

    # select tickets for this smbname
    tickets = acc_df

    # junk (Frequent phrases) detected
    junk_new, map_chunk_ticket = model_extract_junk(
        tickets, min_junk_frequency)

    # Junk co-occurence matrix + connected components
    co_occurence, index = get_co_occurence_matrix(
        junk_new[account_id], map_chunk_ticket[account_id], tickets)
    mat = (co_occurence > min_cooccurence_ratio * np.array([np.diag(co_occurence)]).T) \
        & (co_occurence > min_cooccurence_ratio * np.array([np.diag(co_occurence)]))

    n_connected_components, connected_components_label = connected_components(
        mat)
    for i in range(n_connected_components):
        component_i_junk_idx = np.where(connected_components_label == i)[0]
        n_total = len(tickets)
        n_template_detected = np.median(
            [len(map_chunk_ticket[account_id][index[i]]) for i in component_i_junk_idx])
        rate_pct = np.round(100 * n_template_detected / n_total, 2)
        if len(component_i_junk_idx) >= min_component_size and rate_pct >= min_detection_percent:
            return [index[i] for i in component_i_junk_idx]
    return []


def main(min_junk_frequency=20, min_cooccurence_ratio=0.8,
         min_component_size=5, min_detection_percent=1):
    """
    Clean tickets using the current cleaners. Detect the remaining templates
    in tickets
    :param min_junk_frequency: min occurrences of a chunk to be considered
        junk
    :param min_cooccurence_ratio: 2 chunks will "co-occur" only when the
        co-occurence is at least this ratio of both chuck's occurrence
    :param min_component_size: min number of elements in a template
    :param min_detection_percent: min percentage of tickets having a
        template for this template to be flagged
    :return:
    """
    # read all tickets of all IDs
    tickets_all = pd.read_csv("data/capstone_full_data.csv")
    tickets_all = tickets_all.iloc[:10000]

    for account_id in pd.unique(tickets_all["account_id"]):
        # select tickets for this smbname
        tickets = tickets_all[tickets_all["account_id"] == account_id]
        print("----------", account_id, len(tickets), 'tickets', "----------")
        print(" ")

        # junk (Frequent phrases) detected
        junk_new, map_chunk_ticket = model_extract_junk(
            tickets, min_junk_frequency)

        # Junk co-occurence matrix + connected components
        co_occurence, index = get_co_occurence_matrix(
            junk_new[account_id], map_chunk_ticket[account_id], tickets)
        mat = (co_occurence > min_cooccurence_ratio * np.array([np.diag(co_occurence)]).T) \
            & (co_occurence > min_cooccurence_ratio * np.array([np.diag(co_occurence)]))
        n_connected_components, connected_components_label = connected_components(
            mat)

        # Summary of templates detected
        for i in range(n_connected_components):
            component_i_junk_idx = np.where(connected_components_label == i)[0]
            n_total = len(tickets)
            n_template_detected = np.median(
                [len(map_chunk_ticket[account_id][index[i]]) for i in component_i_junk_idx])
            rate_pct = np.round(100 * n_template_detected / n_total, 2)
            if len(component_i_junk_idx) >= min_component_size and rate_pct >= min_detection_percent:
                template_elements = [index[i] for i in component_i_junk_idx]
                sample_tickets = [np.random.choice(
                    map_chunk_ticket[account_id][index[i]]) for i in component_i_junk_idx]
                slack_message = "*`Detected in {}% of {} tickets`*\n" \
                                "•  *Template elements >* {}\n" \
                                "•  *Sample tickets >* {}\n\n".format(
                                    rate_pct, account_id, template_elements,
                                    sample_tickets)
                print(slack_message)


if __name__ == '__main__':
    main(min_junk_frequency=20,
         min_cooccurence_ratio=0.8,
         min_component_size=5,
         min_detection_percent=1)

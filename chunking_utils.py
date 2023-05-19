import pandas as pd
import random
import string
from tqdm import tqdm

from frequency_model import chunkenizer
from data import get_accounts_week

from flair.data import Sentence
from flair.models import SequenceTagger


def load_chunks_df(df, min_length=0):
    '''
    Receives a df with the full messages
    Returns a df with the messages chunked
    '''

    df_out = pd.DataFrame(columns=[
        'account_id',
        'conversation_id',
        'created_at',
        'message_rank',
        'chunk',
        'chunk_rank',
        'signal'
    ])

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        account_id = row.account_id
        conversation_id = row['conversation_id']
        created_at = row['created_at']
        message_rank = row['message_rank']
        message_body_raw = row['message_body_raw']
        message_body_clean = row['message_body_clean']

        chunks = chunkenizer(message_body_raw)

        for j, chunk in enumerate(chunks):
            if (len(chunk) >= min_length):
                df_out = df_out.append({
                    'account_id': account_id,
                    'conversation_id': conversation_id,
                    'created_at': created_at,
                    'message_rank': message_rank,
                    'chunk': chunk,
                    'chunk_rank': j + 1,
                    'message_body_clean': message_body_clean,
                    'signal': chunk in message_body_clean
                }, ignore_index=True)

    return df_out


def chunk_accounts_from_week(weekly_data_path, week_number):
    accounts = get_accounts_week(week_number, weekly_data_path)

    for account in accounts:
        df = pd.read_csv(weekly_data_path + account)
        df.dropna(subset=['message_body_raw',
                  'message_body_clean'], inplace=True)
        chunks_csv = load_chunks_df(df)
        if (chunks_csv.shape[0] > 0):
            chunks_csv.to_csv(weekly_data_path + 'Chunks/chunks_' + account)


def generateFlairString(raw, clean):
    counter = 0
    signal = []
    string = ''
    for word in raw.rstrip().split():
        if word == clean.split()[counter]:
            if counter + 1 < len(clean.split()):
                counter += 1
                signal.append(word)
            else:
                signal.append(word)
                counter = 0
                string += f"{signal[0]}\tB-Key\n"
                for s in signal[1:]:
                    string += f"{s}\tI-Key\n"
        else:
            string += f"{word}\tO\n"
    return string + '\n'


def generate_flair_txt(df, path):
    """
    Generates the text file necessary for training a flair model.

    Args:
        df: contains message_body_raw, message_body_clean columns
        path: the path in which to store the txt file

    Returns:
        None
    """

    f = open(path, 'w')
    f.write('')
    f = open(path, 'a')

    flair_strings = df.apply(lambda x: generateFlairString(
        x.message_body_raw, x.message_body_clean), axis=1)

    for string in flair_strings.values:
        f.write(string)
    f.close()


def is_chain(row):
    msg = row['message_body_raw']
    lines = msg.split("\n")

    for i, line in enumerate(lines):
        if "On" in line and "wrote:" in line:
            return True

    return False


def get_flair_clean_text(tagger, sentence_str_list):
    sentence_list = [Sentence(sentence_str)
                     for sentence_str in sentence_str_list]
    tagger.predict(sentence_list)

    return pd.Series([" ".join([e.text for e in sentence.get_spans('signal')]) for sentence in sentence_list])

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return random.randint(range_start, range_end)

def randomPhoneNumber():
    format = 1
    phone = str(random_with_N_digits(10))

    if format == 0:
        return phone
    if format == 1:
        return f'{phone[:3]}-{phone[3:-4]}-{phone[-4:]}'
    if format == 2:
        return f'({phone[:3]}) {phone[3:-4]}-{phone[-4:]}'

PRODUCT_NAMES = ["shoes", "shoe", "socks", "insole", "heels", "boots", "belt", "jacket", "pants", "sweater", "sweatshirt", "bag", "hat", "watch", "sunglasses", "sweatpants", "burger", "fries", "chicken nuggets", "face wash", "moisturizer", "lip balm", "lip gloss", "hair cream", "face mask"]
PARTNERS = ["Mack Weldon", "Steve Madden", "ANINE BING", "Popeyes", "Keen Footwear", "ShipStation", "YETI", "Happiest Baby", "Allbirds", "Tim Hortons", "Burger King", "Calendly", "Tecovas", "Paula's Choice Skincare", "Decathlon", "Asurion", "Vans", "Revolve", "The North Face"]

def randomProductName():
    return random.choice(PRODUCT_NAMES)

def randomEmail():
	extensions = ['com','net','org','gov']
	domains = ['gmail','yahoo','comcast','verizon','charter','hotmail','outlook','frontier']

	winext = extensions[random.randint(0,len(extensions)-1)]
	windom = domains[random.randint(0,len(domains)-1)]

	acclen = random.randint(1,20)

	winacc = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(acclen))

	finale = winacc + "@" + windom + "." + winext
	return finale

def randomTrackingNumber():
    format = random.randint(0, 2)
    
    if format == 0:
        return str(random_with_N_digits(10))
    if format == 1:
        return str(random_with_N_digits(11))
    if format == 2:
        return str(random_with_N_digits(12))

def randomOrderNumber():
    format = random.randint(0, 2)
    
    if format == 0:
        return str(random_with_N_digits(6))
    if format == 1:
        return str(random_with_N_digits(7))
    if format == 2:
        return str(random_with_N_digits(8))

def randomAddress(df):
    return df.sample(1).iloc[0].address

def randomName(df):
    return df.sample(1).iloc[0]['name']

def randomPartner():
    return random.choice(PARTNERS)

def noise1(message_raw):
    def isword(part):
        partlist = part.split()
        partlist = filter(lambda x: x, partlist)
        return len(list(partlist)) == 1

    message_dict = {}
    rows = message_raw.split("\n")
    key, val = "", ""
    for r in rows:
        rsplit = r.split(":")
        if len(rsplit) > 1 and isword(rsplit[0]):
            key = rsplit[0].strip().lower()
            val = ":".join(rsplit[1:])
            message_dict[key] = val
        elif val:
            val += ("\n" + r)
            message_dict[key] = val
    return bool(message_dict)
    
def noise2(message_raw):
    if "//" in message_raw:
        return True
    return False

def noise3(message_raw):
    if "sent from" in message_raw.lower():
        return True
    return False
    
def noise4(message_raw):
    mask1 = "We've been contacted by a customer regarding the order identified below" in message_raw
    mask2 = '----- Message:  ----' in message_raw
    mask3 = 'You have received a message' in message_raw
    mask4 = 'You received a new message' in message_raw
    return mask1 or mask2 or mask3 or mask4

NOISE_DICT = {
    "noise1": noise1,
    "noise2": noise2,
    "noise3": noise3,
    "noise4": noise4
}

def label_noise(message_raw):
    labels = []

    for noise_type, noise_func in NOISE_DICT.items():
        if noise_func(message_raw): 
            labels.append(noise_type)
    
    return labels

def label_noise_multicolumn(df, message_raw = "message_body_raw_replaced"):
    for noise_type, noise_func in NOISE_DICT.items():
        df[noise_type] = df[message_raw].apply(noise_func)

    return df


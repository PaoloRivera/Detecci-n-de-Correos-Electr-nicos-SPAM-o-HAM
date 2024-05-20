import os
import re
import email
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import warnings
from bs4 import MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

import re
from bs4 import BeautifulSoup
import html

def preprocess_text(text):
    text = html.unescape(text)
    
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    
    text = re.sub(r'\S*@\S*\s?', ' email ', text)
    
    text = re.sub(r'[^a-zA-Z0-9\s,.!?-]', '', text)
    
    text = text.lower()
    
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_eml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    if msg.is_multipart():
        parts = [part.get_payload(decode=True) for part in msg.walk() if part.get_content_type() == 'text/plain']
        text = ''.join(part.decode(errors='ignore') for part in parts)
    else:
        text = msg.get_payload(decode=True).decode(errors='ignore')

    return preprocess_text(text)

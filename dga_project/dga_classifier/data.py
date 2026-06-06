"""
data.py — Carrega Alexa Top 1M e Bambenek DGA.
Retorna lista de tuplas (label_binaria, dominio_sem_tld, familia).
"""
import os
import random
import csv

# Mapeamento de normalização: nome no CSV → nome canônico usado no código
FAMILY_NAME_MAP = {
    'cryptolocker':         'cryptolocker',
    'p2p':                  'p2pgameoverzeus',
    'post':                 'posttovargoz',
    'volatile':             'volatilecedar',
    'explosive':            'volatilecedar',
    'urlzone':              'shiotob',
    'bebloh':               'shiotob',
    'shiotob/urlzone/bebloh': 'shiotob',
    'banjori':              'banjori',
    'bedep':                'bedep',
    'beebone':              'beebone',
    'corebot':              'corebot',
    'cryptowall':           'cryptowall',
    'dircrypt':             'dircrypt',
    'dyre':                 'dyre',
    'fobber':               'fobber',
    'geodo':                'geodo',
    'hesperbot':            'hesperbot',
    'matsnu':               'matsnu',
    'murofet':              'murofet',
    'necurs':               'necurs',
    'nymaim':               'nymaim',
    'pushdo':               'pushdo',
    'pykspa':               'pykspa',
    'qakbot':               'qakbot',
    'ramnit':               'ramnit',
    'ranbyus':              'ranbyus',
    'shifu':                'shifu',
    'simda':                'simda',
    'suppobox':             'suppobox',
    'symmi':                'symmi',
    'tempedreve':           'tempedreve',
    'tinba':                'tinba',
}

# Mapeamento Tabela V do artigo (Superfamílias) centralizado
SUPERFAMILY_MAP = {
    'dyre':           1,
    'beebone':        2,
    'volatilecedar':  3,
    'shiotob':        4,
    'banjori':        5, 'cryptowall': 5, 'matsnu': 5, 'suppobox': 5,
    'murofet':        6, 'tinba': 6, 'shifu': 6, 'geodo': 6, 'necurs': 6,
    'cryptolocker':   6, 'ramnit': 6, 'ranbyus': 6, 'bedep': 6,
    'hesperbot':      6, 'tempedreve': 6, 'fobber': 6, 'nymaim': 6,
    'qakbot':         6, 'p2pgameoverzeus': 6, 'dircrypt': 6,
    'pykspa':         7,
    'pushdo':         8, 'simda': 8,
    'posttovargoz':   9,
    'corebot':        10,
    'symmi':          11,
}

# As 3 maiores famílias do dataset (para treino dos HMMs DGA)
TOP3_DGA_FAMILIES = ['posttovargoz', 'banjori', 'ramnit']

# As 10 menores famílias (experimento Leave-Class-Out)
LEAVE_OUT_FAMILIES = [
    'bedep', 'beebone', 'corebot', 'cryptowall', 'dircrypt',
    'fobber', 'hesperbot', 'matsnu', 'symmi', 'tempedreve'
]

def _strip_tld(domain):
    """Remove o TLD e retorna apenas o SLD em lowercase."""
    domain = str(domain).lower().strip()
    parts = domain.split('.')
    if len(parts) == 1:
        return parts[0]
    return '.'.join(parts[:-1])


def get_data(dga_path='datasets/bambenek_dga_domain_30.csv',
             alexa_path='datasets/top-1m.csv',
             seed=42):
    """
    Carrega os datasets respeitando o volume total (~750k DGA) e a proporção de classes reais.
    """
    indata = []

    print("[data] Lendo DGA (mantendo distribuicao real das classes)...")
    try:
        with open(dga_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_family = str(row.get('DGA_family', '')).lower().strip()
                family = FAMILY_NAME_MAP.get(raw_family, raw_family)
                domain = _strip_tld(row.get('Domain', ''))
                if domain:
                    indata.append(('dga', domain, family))
    except IOError:
        print("[data] Erro: Arquivo DGA nao encontrado em", dga_path)

    n_dga = len(indata)
    print("[data] Dominios DGA carregados: %d" % n_dga)

    print("[data] Lendo Alexa Top 1M...")
    alexa_data = []
    try:
        with open(alexa_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 1:
                    domain = _strip_tld(row[1])
                    if domain:
                        alexa_data.append(('benign', domain, 'benign'))
    except IOError:
        print("[data] Erro: Arquivo Alexa nao encontrado em", alexa_path)

    # Pareamento: Usa o mesmo número de amostras benignas que a soma dos DGAs
    random.seed(seed)
    random.shuffle(alexa_data)
    
    sample_size = min(n_dga, len(alexa_data))
    indata.extend(alexa_data[:sample_size])

    print("[data] Dominios Benignos amostrados: %d" % sample_size)

    random.seed(seed)
    random.shuffle(indata)

    print("[data] Dataset carregado. Total de amostras: %d" % len(indata))
    return indata
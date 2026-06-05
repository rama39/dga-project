"""
data.py — carrega Alexa Top 1M e Bambenek DGA (30 famílias).
Retorna lista de tuplas (label_binaria, dominio_sem_tld, familia).
  label_binaria: 'benign' ou 'dga'
  dominio_sem_tld: string lowercase sem TLD
  familia: nome normalizado da família (ex: 'banjori', 'posttovargoz', 'benign')

Compatível com lstm.py, bigram.py, randomforest.py e hmm.py.
"""
import pandas as pd
import os
import random


# Mapeamento de normalização: nome no CSV → nome canônico usado no código
FAMILY_NAME_MAP = {
    'cryptolocker':         'cryptolocker',
    'p2p':                  'p2pgameoverzeus',
    'post':                 'posttovargoz',
    'volatile':             'volatilecedar',
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
    'shiotob/urlzone/bebloh': 'shiotob',
    'simda':                'simda',
    'suppobox':             'suppobox',
    'symmi':                'symmi',
    'tempedreve':           'tempedreve',
    'tinba':                'tinba',
}

# As 3 maiores famílias do dataset (para treino dos HMMs DGA)
# Banjori (~439k), Post/posttovargoz (~66k), Ramnit (~56k)
TOP3_DGA_FAMILIES = ['banjori', 'posttovargoz', 'ramnit']

# As 10 menores famílias (experimento Leave-Class-Out)
LEAVE_OUT_FAMILIES = [
    'bedep', 'beebone', 'corebot', 'cryptowall', 'dircrypt',
    'fobber', 'hesperbot', 'matsnu', 'symmi', 'tempedreve'
]


def _strip_tld(domain: str) -> str:
    """Remove o TLD e retorna apenas o SLD em lowercase."""
    domain = domain.lower().strip()
    parts = domain.split('.')
    # Mantém subdomínios mas remove o TLD (último elemento)
    return parts[0] if len(parts) == 1 else '.'.join(parts[:-1])


def get_data(
    dga_path: str = os.path.join('datasets', 'bambenek_dga_domain_30.csv'),
    alexa_path: str = os.path.join('datasets', 'top-1m.csv'),
    samples_per_family: int = 2000,
    seed: int = 42,
):
    """
    Carrega e balanceia os datasets.

    Parâmetros
    ----------
    dga_path : caminho para o CSV do Bambenek
    alexa_path : caminho para o CSV do Alexa Top 1M
    samples_per_family : máximo de domínios por família DGA (reduz uso de RAM)
    seed : semente aleatória para reprodutibilidade

    Retorna
    -------
    list[tuple[str, str, str]]
        Cada elemento: (label_binaria, dominio_sem_tld, familia_normalizada)
    """
    indata = []

    # ------------------------------------------------------------------
    # 1. Dataset DGA (Bambenek)
    # ------------------------------------------------------------------
    print(f"[data] Lendo DGA de: {dga_path}")
    df_dga = pd.read_csv(dga_path)

    # Amostragem estratificada por família
    df_dga_sample = (
        df_dga
        .groupby('DGA_family', group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), samples_per_family), random_state=seed))
        .reset_index(drop=True)
    )

    for _, row in df_dga_sample.iterrows():
        raw_family = str(row['DGA_family']).lower().strip()
        family = FAMILY_NAME_MAP.get(raw_family, raw_family)
        domain = _strip_tld(str(row['Domain']))
        if domain:
            indata.append(('dga', domain, family))

    n_dga = len(indata)
    print(f"[data] DGA carregados: {n_dga}")

    # ------------------------------------------------------------------
    # 2. Dataset benigno (Alexa Top 1M)
    # ------------------------------------------------------------------
    print(f"[data] Lendo Alexa de: {alexa_path}")
    df_alexa = pd.read_csv(alexa_path, header=None, names=['rank', 'domain'])
    df_alexa = df_alexa.dropna(subset=['domain'])
    df_alexa_sample = df_alexa.sample(n=min(n_dga, len(df_alexa)), random_state=seed)

    for domain_raw in df_alexa_sample['domain']:
        domain = _strip_tld(str(domain_raw))
        if domain:
            indata.append(('benign', domain, 'benign'))

    print(f"[data] Benignos carregados: {len(indata) - n_dga}")

    # ------------------------------------------------------------------
    # 3. Embaralhar
    # ------------------------------------------------------------------
    random.seed(seed)
    random.shuffle(indata)

    print(f"[data] Total de amostras: {len(indata)}")
    return indata
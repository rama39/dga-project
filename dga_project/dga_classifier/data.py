"""
data.py — Carrega Alexa Top 1M e Bambenek DGA.
Retorna lista de tuplas (label_binaria, dominio_sem_tld, familia).

"""
import os
import random
import csv
from collections import Counter, defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DEFAULT_DATA_DIR = os.environ.get('DGA_DATA_DIR', os.path.join(_PROJECT_ROOT, 'datasets'))


def _resolve_path(path, default_name):
    """Resolve caminho de dataset de forma robusta para script local/Colab."""
    if path and os.path.isfile(path):
        return path
    candidate = os.path.join(_DEFAULT_DATA_DIR, os.path.basename(path or default_name))
    if os.path.isfile(candidate):
        return candidate
    return path or candidate


FAMILY_NAME_MAP = {
    'cryptolocker': 'cryptolocker',
    'p2p': 'p2pgameoverzeus',
    'post': 'posttovargoz',
    'volatile': 'volatilecedar',
    'explosive': 'volatilecedar',
    'urlzone': 'shiotob',
    'bebloh': 'shiotob',
    'shiotob/urlzone/bebloh': 'shiotob',
    'banjori': 'banjori',
    'bedep': 'bedep',
    'beebone': 'beebone',
    'corebot': 'corebot',
    'cryptowall': 'cryptowall',
    'dircrypt': 'dircrypt',
    'dyre': 'dyre',
    'fobber': 'fobber',
    'geodo': 'geodo',
    'hesperbot': 'hesperbot',
    'matsnu': 'matsnu',
    'murofet': 'murofet',
    'necurs': 'necurs',
    'nymaim': 'nymaim',
    'pushdo': 'pushdo',
    'pykspa': 'pykspa',
    'qakbot': 'qakbot',
    'ramnit': 'ramnit',
    'ranbyus': 'ranbyus',
    'shifu': 'shifu',
    'simda': 'simda',
    'suppobox': 'suppobox',
    'symmi': 'symmi',
    'tempedreve': 'tempedreve',
    'tinba': 'tinba',
}

SUPERFAMILY_MAP = {
    'dyre': 1,
    'beebone': 2,
    'volatilecedar': 3,
    'shiotob': 4,
    'banjori': 5, 'cryptowall': 5, 'matsnu': 5, 'suppobox': 5,
    'murofet': 6, 'tinba': 6, 'shifu': 6, 'geodo': 6, 'necurs': 6,
    'cryptolocker': 6, 'ramnit': 6, 'ranbyus': 6, 'bedep': 6,
    'hesperbot': 6, 'tempedreve': 6, 'fobber': 6, 'nymaim': 6,
    'qakbot': 6, 'p2pgameoverzeus': 6, 'dircrypt': 6,
    'pykspa': 7,
    'pushdo': 8, 'simda': 8,
    'posttovargoz': 9,
    'corebot': 10,
    'symmi': 11,
}

TOP3_DGA_FAMILIES = ['posttovargoz', 'banjori', 'ramnit']

LEAVE_OUT_FAMILIES = [
    'bedep', 'beebone', 'corebot', 'cryptowall', 'dircrypt',
    'fobber', 'hesperbot', 'matsnu', 'symmi', 'tempedreve'
]


def _strip_tld(domain):
    """Remove TLD e retorna apenas o domínio sem TLD em lowercase."""
    domain = str(domain).lower().strip()
    if not domain:
        return ''
    parts = domain.split('.')
    return parts[0] if len(parts) == 1 else '.'.join(parts[:-1])


def _env_int(name, default=None):
    v = os.environ.get(name)
    if v in (None, '', 'None'):
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v in (None, '', 'None'):
        return default
    return str(v).lower() in ('1', 'true', 'yes', 'y', 'sim')


def _sample_by_family(records, seed=42, max_per_family=None, max_total=None):
    """Aplica teto por família e/ou teto total preservando minorias sempre que possível."""
    rng = random.Random(seed)
    by_fam = defaultdict(list)
    for rec in records:
        by_fam[rec[2]].append(rec)

    capped = []
    for fam in sorted(by_fam):
        recs = list(by_fam[fam])
        rng.shuffle(recs)
        if max_per_family is not None:
            recs = recs[:max_per_family]
        capped.extend(recs)

    if max_total is not None and len(capped) > max_total:
        # Amostragem estratificada proporcional após o teto por família.
        by_fam = defaultdict(list)
        for rec in capped:
            by_fam[rec[2]].append(rec)

        total = sum(len(v) for v in by_fam.values())
        selected = []
        quotas = {}
        for fam, recs in by_fam.items():
            quotas[fam] = max(1, int(round(max_total * len(recs) / total)))
        # Corrige arredondamento para bater no máximo pedido.
        while sum(quotas.values()) > max_total:
            fam = max(quotas, key=lambda f: quotas[f])
            quotas[fam] -= 1
        while sum(quotas.values()) < max_total:
            fam = max(by_fam, key=lambda f: len(by_fam[f]) - quotas.get(f, 0))
            quotas[fam] += 1

        for fam in sorted(by_fam):
            recs = list(by_fam[fam])
            rng.shuffle(recs)
            selected.extend(recs[:quotas[fam]])
        capped = selected

    rng.shuffle(capped)
    return capped


def get_data(dga_path='datasets/bambenek_dga_domain_30.csv',
             alexa_path='datasets/top-1m.csv',
             seed=42,
             balance=None,
             max_benign=100000,
             max_dga=None,
             max_per_family=None,
             verbose=True):
    """
    Carrega os datasets com controles opcionais de volume.

    Variáveis úteis no Colab:
      DGA_DATA_DIR=/content/seu_repo/datasets
      DGA_BALANCE=1                 -> deixa benign ~= dga
      DGA_MAX_BENIGN=100000         -> limite de benignos usados
      DGA_MAX_SAMPLES=50000         -> limite total de DGA após teto por família
      DGA_MAX_PER_FAMILY=5000       -> teto por família; use None/0 para desligar
    """
    dga_path = _resolve_path(dga_path, 'bambenek_dga_domain_30.csv')
    alexa_path = _resolve_path(alexa_path, 'top-1m.csv')

    if balance is None:
        balance = _env_bool('DGA_BALANCE', default=False)
    if max_dga is None:
        max_dga = _env_int('DGA_MAX_SAMPLES', default=None)
    if max_benign is None:
        max_benign = _env_int('DGA_MAX_BENIGN', default=None)
    if max_per_family is None:
        max_per_family = _env_int('DGA_MAX_PER_FAMILY', default=5000)
    if max_per_family in (0, -1):
        max_per_family = None

    rng = random.Random(seed)

    if verbose:
        print(f"[data] DGA path: {dga_path}")
        print(f"[data] Alexa path: {alexa_path}")

    dga_records = []
    try:
        with open(dga_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_family = str(row.get('DGA_family', '')).lower().strip()
                family = FAMILY_NAME_MAP.get(raw_family, raw_family)
                domain = _strip_tld(row.get('Domain', ''))
                if domain:
                    dga_records.append(('dga', domain, family))
    except IOError:
        print("[data] ERRO: DGA nao encontrado em", dga_path)

    original_dga = len(dga_records)
    dga_records = _sample_by_family(
        dga_records,
        seed=seed,
        max_per_family=max_per_family,
        max_total=max_dga,
    )

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
        print("[data] ERRO: Alexa nao encontrado em", alexa_path)

    rng.shuffle(alexa_data)
    n_dga = len(dga_records)
    sample_size = min(n_dga, len(alexa_data)) if balance else len(alexa_data)
    if max_benign is not None:
        sample_size = min(sample_size, max_benign)

    indata = list(dga_records) + alexa_data[:sample_size]
    rng.shuffle(indata)

    if verbose:
        fam_counts = Counter(f for _, _, f in dga_records)
        print(f"[data] DGA original: {original_dga}")
        print(f"[data] DGA usado: {len(dga_records)} | max_per_family={max_per_family} | max_dga={max_dga}")
        print(f"[data] Benignos usados: {sample_size} | balance={balance} | max_benign={max_benign}")
        print(f"[data] Total: {len(indata)}")
        print("[data] Top famílias DGA:", fam_counts.most_common(10))

    return indata

from pathlib import Path
import re
import unicodedata
import pdfplumber

BASE = Path(r'C:\Users\BQ6757\chronos2_v1\tmp\cv_these_75968')
SOURCE = Path(r'C:\Users\BQ6757\Downloads\Memoire_Yoan_Kesraoui.pdf')
raw = (BASE / 'memoire.txt').read_text(encoding='utf-8')
raw = raw[raw.index('=== PAGE 6 ==='):raw.index('=== PAGE 27 ===')]
raw = re.sub(r'=== PAGE \d+ ===\n\d+\s*\n', '', raw)

# Recover paragraph boundaries from the body font and baseline spacing. The
# canonical content still comes from pypdf, whose reading order preserves notes.
def key(s):
    return ''.join(c.lower() for c in unicodedata.normalize('NFKC', s) if c.isalnum())

para_keys = set()
with pdfplumber.open(SOURCE) as doc:
    for page in doc.pages[5:26]:
        body = page.filter(lambda x: x.get('object_type') == 'char'
                           and 'TimesNewRoman' in x.get('fontname', '')
                           and abs(x.get('size', 0) - 12) < 0.1)
        previous_top = None
        for line in body.extract_text_lines():
            top = line['top']
            if previous_top is not None and top - previous_top > 17.2:
                k = key(line['text'])[:22]
                if len(k) >= 15:
                    para_keys.add(k)
            previous_top = top

equations = [
    ('𝑥𝑥𝑖𝑖 =', 'Le vecteur numérique', [
        r'x_i = E[v_i] + p_i \qquad \text{ou} \qquad x_i = u_i W_E + b_E + p_i \tag{1}']),
    ('𝑄𝑄 =', 'Q et K ont', [
        r'Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V \tag{2}']),
    ('𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝐴𝑜𝑜𝐴𝐴', 'Le softmax s’applique', [
        r'\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V \tag{3}']),
    ('𝑠𝑠𝑖𝑖𝑖𝑖 =', 'Forme sans masque', [
        r's_{ij}=\frac{q_i k_j^{\mathsf T}}{\sqrt{d_k}}, \qquad a_{ij}=\frac{\exp(s_{ij})}{\sum_{\ell=1}^{n}\exp(s_{i\ell})}, \qquad y_i=\sum_{j=1}^{n}a_{ij}v_j \tag{4}']),
    ('𝑀𝑀𝑀𝑀𝐴𝐴(𝑋𝑋)', 'La projection WO', [
        r'\operatorname{MHA}(X)=\operatorname{Concat}(Y_1,\ldots,Y_h)W_O \tag{5a}',
        r'Y_r=\operatorname{Attention}\!\left(XW_Q^{(r)},XW_K^{(r)},XW_V^{(r)}\right) \tag{5b}']),
    ('𝑃𝑃𝐸𝐸(𝑝𝑝', 'Pour dmodel pair', [
        r'\operatorname{PE}(p,2r)=\sin\!\left(\frac{p}{10000^{2r/d_{\mathrm{model}}}}\right) \tag{6a}',
        r'\operatorname{PE}(p,2r+1)=\cos\!\left(\frac{p}{10000^{2r/d_{\mathrm{model}}}}\right) \tag{6b}']),
    ('𝑀𝑀𝑖𝑖𝑖𝑖 =', 'M a la taille', [
        r'M_{ij}=\begin{cases}0,&j\leq i,\\-\infty,&j>i.\end{cases} \tag{7a}',
        r'Y=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V \tag{7b}']),
    ('𝐹𝐹𝐹𝐹𝐹𝐹(𝑥𝑥)', 'W₁ projette', [
        r'\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2 \tag{8}']),
    ('𝑅𝑅𝐹𝐹(𝑧𝑧)', 'γ et β sont appris', [
        r'\operatorname{LN}(z)=\gamma\odot\frac{z-\mu(z)}{\sqrt{\sigma^2(z)+\varepsilon}}+\beta \tag{9a}',
        r'\mu(z)=\frac{1}{d_{\mathrm{model}}}\sum_{j=1}^{d_{\mathrm{model}}}z_j, \qquad \sigma^2(z)=\frac{1}{d_{\mathrm{model}}}\sum_{j=1}^{d_{\mathrm{model}}}\bigl(z_j-\mu(z)\bigr)^2 \tag{9b}']),
    ('𝑍𝑍 =', 'Bloc post-normalisation', [
        r'Z=\operatorname{LN}\!\left(X+\operatorname{MHA}(X)\right), \qquad X^{\prime}=\operatorname{LN}\!\left(Z+\operatorname{FFN}(Z)\right) \tag{10}']),
    ('ŷ\n', 'Dans cette écriture, y représente', [
        r'\widehat{y}_{t+1:t+H}=f_\theta\!\left(y_{t-L+1:t},x_{t-L+1:t},z_{t+1:t+H},s\right) \tag{11}']),
    ('𝐹𝐹 =', 'Pc a la taille', [
        r'N=\left\lfloor\frac{L-P}{S}\right\rfloor+2 \tag{12a}',
        r'E_c=P_cW_p+E_{\mathrm{pos}} \tag{12b}']),
    ('𝐴𝐴\n', 'Z⁽⁰⁾ réunit', [
        r'e_c^{(0)}=E(U_{:,c}) \tag{13a}',
        r'Z^{(\ell+1)}=B_\ell\!\left(Z^{(\ell)}\right), \qquad \ell=0,\ldots,K-1 \tag{13b}',
        r'\widehat{Y}_{:,c}=G\!\left(Z_{c,:}^{(K)}\right) \tag{13c}']),
    ('ỹ𝑟𝑟,𝑐𝑐', 'r parcourt le contexte', [
        r'\widetilde{y}_{r,c}=\frac{y_{r,c}-\mu_c}{s_c} \tag{14a}',
        r'\widehat{y}_{t+h,c}=s_c\widehat{\widetilde{y}}_{t+h,c}+\mu_c \tag{14b}']),
    ('𝑀𝑀𝐴𝐴𝐸𝐸 =', 'La MASE rapporte', [
        r'\operatorname{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\widehat{y}_i|, \qquad \operatorname{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\widehat{y}_i)^2} \tag{15}']),
    ('𝑀𝑀𝐴𝐴𝑆𝑆𝐸𝐸 =', 'Un dénominateur nul', [
        r'\operatorname{MASE}=\frac{\displaystyle\frac{1}{H}\sum_{h=1}^{H}|y_{T+h}-\widehat{y}_{T+h}|}{\displaystyle\frac{1}{T-m}\sum_{t=m+1}^{T}|y_t-y_{t-m}|} \tag{16}']),
    ('𝜌𝜌\n', 'Le CRPS intègre', [
        r'\rho_\tau(y-q)=(y-q)\bigl(\tau-\mathbf{1}_{\{y<q\}}\bigr), \qquad 0<\tau<1 \tag{17}']),
    ('𝐶𝐶𝑅𝑅𝑃𝑃𝑆𝑆', 'Ces scores évaluent', [
        r'\operatorname{CRPS}(F,y)=\int_{-\infty}^{+\infty}\left[F(z)-\mathbf{1}_{\{y\leq z\}}\right]^2\,\mathrm{d}z \tag{18}']),
    ('𝑅𝑅\n', 'K est ici le nombre', [
        r'\mathcal{L}_{\mathrm{CE}}(\theta)=-\sum_{k=1}^{K}\log p_\theta\!\left(z_k\mid c,z_{1:k-1}\right) \tag{19}']),
    ('𝑅𝑅𝐹𝐹𝐹𝐹', 'τ est uniforme', [
        r'\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{\tau,\varepsilon,y,h}\!\left[\left\|v_\theta(x_\tau,\tau,h)-(y-\varepsilon)\right\|_2^2\right] \tag{20}']),
]

math_blocks = {}
caption_starts = []
for family, (start, end, formulas) in enumerate(equations, 1):
    a = raw.index(start)
    b = raw.index(end, a)
    token = f'@@EQUATION_{family}@@'
    raw = raw[:a] + token + '\n' + raw[b:]
    math_blocks[token] = '\n\n'.join('$$\n' + f + '\n$$' for f in formulas)
    caption_starts.append(end)

# pypdf puts a few subscripts or hyperlink fragments on their own physical line.
raw = raw.replace('d’indice v\ni,', 'd’indice vᵢ,')
raw = re.sub(r'\n\s*\n+', '\n', raw)
raw = raw.replace('Con­', 'Con')
forced_starts = [
    'Le Transformer transforme', 'On note n le nombre', 'Pour un token discret',
    'Les représentations sont regroupées', 'L’auto-attention commence',
    'La requête qi décrit', 'Dans l’auto-attention', 'Le produit QKᵀ',
    'Le calcul peut être détaillé', 'Prenons un exemple simple', 'La division par',
    'Les coefficients d’attention', 'Une attention à plusieurs têtes',
    'Le choix fréquent', 'Les paramètres de chaque tête',
    'L’attention calculée uniquement', 'Le Transformer original utilise',
    'Cette construction fournit', 'Pour une série temporelle',
    'Pendant l’apprentissage d’un décodeur', 'Le masque a la taille',
    'En prévision, un encodeur', 'L’attention fait circuler',
    'Une connexion résiduelle', 'La normalisation par couche',
    'LayerNorm ne normalise', 'L’encodeur empile', 'Pendant l’entraînement',
    'Pendant l’apprentissage autorégressif', 'Le coût de l’attention dense',
    'La prévision consiste', 'Dans cette écriture, y représente',
    'L’apprentissage se construit', 'Pour comparer les méthodes',
    'PatchTST normalise', 'N compte les segments', 'Dans un exemple pédagogique',
    'iTransformer change', 'E transforme un historique', 'Cette attention coûte',
    'Informer conserve', 'Entre certains blocs', 'La complexité annoncée',
    'Autoformer introduit', 'Le mécanisme Auto', 'La sortie additionne',
    'Le Temporal Fusion', 'Des couches LSTM', 'La tête de sortie fournit',
    'RevIN peut entourer', 'Les statistiques proviennent', 'DLinear permet',
    'DeepAR fournit', 'Les architectures décrites', 'Le préentraînement ajoute',
    'En zero', 'LoRA conserve', 'Sur N observations', 'La MASE rapporte',
    'Un dénominateur nul', 'Pour un quantile de niveau', 'Le CRPS intègre',
    'Ces scores évaluent', 'Les modèles de fondation temporels cherchent',
    'La littérature retenue', 'Chronos transforme', 'Dans cette écriture, c désigne',
    'Cette formulation pose', 'TimesFM conserve', 'L’apprentissage masque',
    'TimesFM 2.5', 'Moirai 1 traite', 'La projection de sortie fournit',
    'Moirai 2.0 modifie', 'Cette simplification', 'Chronos-2 rompt',
    'Les tâches multivariées', "L'annonce de TimesFM", 'Une prévision probabiliste peut',
    'Le point intermédiaire', 'Sundial applique', 'Le décodage comporte',
    "Dans un mélange d'experts", 'Time-MoE représente', "À l'inférence, l'organisation",
    'Moirai-MoE utilise', "L'interprétation demande", 'Timer-S1 déplace',
    "Le préentraînement est suivi", 'TiRex apporte', 'Toto 2.0 reprend',
    'La famille est entraînée', 'Ces méthodes conduisent', 'Les résultats présentés',
    'Le zero-shot désigne', 'GIFT-Eval sépare', 'TSFMAudit propose',
    'TIME rassemble', "Impermanent s'appuie", 'On retient de ces travaux',
    'fev-bench comprend', 'Guibert et al.', 'Laglil et al.',
    "L'apprentissage continu", 'Enfin, Zhang et al.', 'Perez-Diaz et al.',
    'Wan et al.',
]
forced_starts += caption_starts

lines = [line.strip() for line in raw.splitlines() if line.strip()]
blocks = []
current = []
def flush():
    if current:
        blocks.append(' '.join(current))
        current.clear()

for line in lines:
    if line in math_blocks:
        flush()
        blocks.append(math_blocks[line])
        continue
    if re.match(r'^[2-5](?:\.\d+)?\s+[A-Za-zÉÀ]', line):
        flush()
        blocks.append(('# ' if re.match(r'^[2-5]\s', line) else '## ') + line)
        continue
    if line.startswith('Figure '):
        flush()
        fig = re.match(r'Figure (\d+)', line).group(1)
        blocks.append(f'[[FIGURE_{fig}]]')
        current.append(line)
        continue
    new = any(line.startswith(s) for s in forced_starts)
    if not new and len(key(line)) >= 22:
        new = key(line)[:22] in para_keys
    if new:
        flush()
    current.append(line)
flush()

def cleanup(p):
    if p.startswith('$$'):
        return p
    p = re.sub(r'\s+', ' ', p)
    p = re.sub(r'\s+([,.)])', r'\1', p)
    p = re.sub(r'\(\s+', '(', p)
    p = re.sub(r'(?<=\w)\s*-\s*(?=\w)', '-', p)
    p = p.replace('apprennent -ils', 'apprennent-ils')
    p = p.replace('d model', 'dmodel').replace('d k', 'dk')
    # Restore inline subscripts where extraction lost their typography.
    subs = {
        'dmodel': 'd_model', 'dk': 'd_k', 'dv': 'd_v', 'dff': 'd_ff',
        'qi': 'qᵢ', 'kj': 'kⱼ', 'vj': 'vⱼ', 'pi': 'pᵢ',
        's ij': 'sᵢⱼ', 'aij': 'aᵢⱼ', 'WE': 'W_E', 'WO': 'W_O',
        'Pc': 'P_c', 'Wp': 'W_p', 'Ec': 'E_c', 'Epos': 'E_pos',
        'sc': 's_c',
    }
    for src, dst in subs.items():
        p = re.sub(r'(?<!\w)' + re.escape(src) + r'(?!\w)', lambda m: dst, p)
    p = p.replace('u i,', 'uᵢ,')
    p = p.replace('μ c', 'μ_c')
    p = p.replace('2.2 Construire les requêtes les clés et les valeurs',
                  '2.2 Construire les requêtes, les clés et les valeurs')
    p = p.replace('transformées par arcsinus hyperbolique',
                  'transformées par la fonction sinus hyperbolique inverse (arsinh)')
    p = re.sub(r'\. (\([^()]*\b(?:19|20)\d{2}[^()]*\))$', r' \1.', p)
    p = p.replace('attention point par point', 'attention point par point')
    return p.strip()

blocks = [cleanup(p) for p in blocks]

# Five validated scientific clarifications. Keep bibliographic citations already
# present in the source; no unverified practical experience is added.
def add_after(anchor, text):
    hits = [i for i, p in enumerate(blocks) if anchor in p]
    assert len(hits) == 1, (anchor, hits)
    blocks.insert(hits[0] + 1, text)

add_after('Fournir davantage d’historique',
    'Un modèle de fondation peut rester gelé alors qu’une régression auxiliaire, '
    'une correction résiduelle ou des poids d’ensemble sont ajustés sur la tâche cible. '
    'Il faut alors distinguer l’absence d’adaptation du réseau préentraîné de '
    'l’apprentissage réalisé par le système complet. La qualification zero-shot doit '
    'préciser le composant auquel elle s’applique.')

add_after('\\tag{18}',
    'Lorsqu’un modèle fournit seulement une grille finie de quantiles, le score calculé '
    'à partir de cette grille constitue une approximation du CRPS ou une perte quantile '
    'agrégée, selon la convention retenue. Les niveaux, pondérations, normalisations et '
    'règles d’interpolation doivent être identiques ou explicitement documentés pour '
    'comparer les modèles (Gneiting et Raftery, 2007 ; Shchur et al., 2025).')

add_after('Chronos-2 rompt',
    'Les covariables catégorielles sont converties en valeurs numériques avant '
    'l’entrée dans le réseau : target encoding pour une cible univariée, encodage '
    'ordinal dans le cas multivarié décrit par l’article. Cette prise en charge '
    'repose sur un encodage numérique des catégories, sans traitement natif '
    'de covariables textuelles (Ansari et al., 2025).')

add_after('repose sur un encodage numérique des catégories',
    'Le partage d’information entre variables modifie le conditionnement des '
    'prévisions. Il ne suffit pas à définir la dépendance des erreurs futures : '
    'une tête de quantiles fournit encore des quantiles marginaux, par variable '
    'et par horizon. Cette distinction sera reprise dans la discussion des '
    'trajectoires probabilistes en section 5.5 (Ansari et al., 2025 ; '
    'Perez-Diaz et al., 2025).')

old = "Ses scores de gain et ses intervalles bootstrap renseignent l'ampleur et l'incertitude des écarts entre modèles."
new = ("Ses scores de gain renseignent l’ampleur des écarts entre modèles. Ses intervalles "
       "reposent sur un bootstrap apparié des tâches et renseignent la stabilité des "
       "comparaisons lorsque la composition du benchmark varie. Ils ne quantifient "
       "pas, à eux seuls, toute l’incertitude liée aux observations futures ou aux "
       "répétitions d’entraînement.")
assert sum(old in p for p in blocks) == 1
blocks = [p.replace(old, new) for p in blocks]

result = '\n\n'.join(blocks).strip() + '\n'
assert not any(ord(c) > 0xffff for c in result), 'Unrepaired mathematical glyphs'
assert '\ufffd' not in result
assert result.count('[[FIGURE_') == 3
assert len(re.findall(r'^# ', result, re.M)) == 4
assert len(re.findall(r'^## ', result, re.M)) == 28, re.findall(r'^## .+$', result, re.M)
assert all(re.search(r'\\tag\{' + str(i) + r'(?:[abc])?\}', result) for i in range(1,21))
(BASE / 'memoire_ch2_5.md').write_text(result, encoding='utf-8')
print(f'Wrote {len(result)} characters, {len(blocks)} blocks, '
      f'{len(re.findall(chr(92) + chr(92) + "tag", result))} equations/subequations.')

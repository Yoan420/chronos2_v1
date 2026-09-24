# 2 Comprendre le fonctionnement du Transformer

## 2.1 Représenter une séquence

Le Transformer transforme une suite d’éléments en une suite de vecteurs contextualisés. Pour une position donnée, la représentation obtenue dépend de son contenu et des autres éléments accessibles. Cette mise en relation est assurée par l’attention. Le réseau comporte également des transformations non linéaires, des connexions résiduelles et des normalisations. Ces opérations sont répétées dans plusieurs couches (Vaswani et al., 2017).

On note n le nombre d’éléments de la séquence, d_model la dimension de leur représentation et h le nombre de têtes d’attention. Il faut d’abord choisir ce que représente un élément, appelé token. Dans un texte, ce peut être un mot ou une partie de mot. Dans une série temporelle, ce peut être une mesure, un vecteur de variables ou un segment de plusieurs mesures. Ce choix détermine les informations que le réseau pourra directement mettre en relation.

Pour un token discret d’indice vᵢ, l’embedding consiste à sélectionner une ligne dans une matrice E apprise. L’indice sert uniquement à retrouver un vecteur ; sa valeur numérique ne mesure pas une proximité entre tokens. Pour une entrée numérique uᵢ, une projection affine permet de construire directement le vecteur. On peut ensuite ajouter un vecteur de position pᵢ.

$$
x_i = E[v_i] + p_i \qquad \text{ou} \qquad x_i = u_i W_E + b_E + p_i \tag{1}
$$

Le vecteur numérique uᵢ est projeté par W_E ; pᵢ représente sa position. Les biais et les positions ont d_model coordonnées.

Les représentations sont regroupées dans une matrice X de taille n × d_model. Chaque ligne correspond à un token. Dans la suite, les formules utilisent cette convention de vecteurs lignes. Les dimensions sont importantes : elles permettent de vérifier quelles opérations sont possibles et sur quel axe l’information circule.

## 2.2 Construire les requêtes, les clés et les valeurs

L’auto-attention commence par trois projections linéaires de X. Elles produisent les requêtes Q, les clés K et les valeurs V. Les matrices de projection sont des paramètres appris avec le reste du réseau.

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V \tag{2}
$$

Q et K ont n × d_k coordonnées ; V en a n × d_v. Les biais sont omis.

La requête qᵢ décrit ce qui sert à chercher de l’information pour la position i. Elle est comparée à chaque clé kⱼ. La valeur vⱼ contient, quant à elle, le vecteur transmis lorsque l’élément j reçoit un poids important. Cette explication aide à distinguer les rôles, mais il ne faut pas imaginer que le réseau formule des questions en langage naturel.

Dans l’auto-attention, Q, K et V proviennent de la même séquence. Elles restent différentes puisque leurs matrices de projection sont distinctes. Pour une attention croisée, on utilise deux séquences : les requêtes sont construites à partir de la première, les clés et les valeurs à partir de la seconde. Ce mécanisme permet, par exemple, à un décodeur de consulter une entrée déjà encodée.

[[FIGURE_1]]

Figure 1 Calcul d’une tête d’auto-attention. Schéma original d’après Vaswani et al. (2017).

## 2.3 Calculer les poids et agréger les valeurs

Le produit QKᵀ mesure toutes les compatibilités entre les requêtes et les clés. Dans le cas d’une auto-attention sur n éléments, il produit une matrice n × n. Le score de la ligne i et de la colonne j est ensuite divisé par la racine carrée de d_k, la dimension des requêtes et des clés. La fonction softmax est appliquée sur les colonnes de chaque ligne. Elle transforme les scores en coefficients positifs dont la somme vaut un.

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V \tag{3}
$$

Le softmax s’applique aux colonnes de chaque ligne. La sortie a la taille n × d_v.

Le calcul peut être détaillé pour une position. On obtient d’abord le score sᵢⱼ, puis le coefficient aᵢⱼ, et enfin la somme pondérée des valeurs. La formule suivante suppose qu’aucune position n’est masquée.

$$
s_{ij}=\frac{q_i k_j^{\mathsf T}}{\sqrt{d_k}}, \qquad a_{ij}=\frac{\exp(s_{ij})}{\sum_{\ell=1}^{n}\exp(s_{i\ell})}, \qquad y_i=\sum_{j=1}^{n}a_{ij}v_j \tag{4}
$$

Forme sans masque ; la somme des aᵢⱼ vaut un pour chaque requête i.

Prenons un exemple simple : si les poids d’une requête sont 0,7, 0,2 et 0,1, sa sortie vaut 0,7v₁ + 0,2v₂ + 0,1v₃. Il s’agit d’une moyenne pondérée de vecteurs, pas nécessairement d’une moyenne des observations brutes. Ces poids changent avec l’entrée, alors que les matrices de projection restent fixes pendant une inférence ordinaire. Cette distinction explique comment le même réseau peut exploiter différemment deux historiques.

La division par √ d_k évite que les scores deviennent trop grands quand leur dimension augmente. Si l’on suppose, pour comprendre ce point, des composantes indépendantes, centrées et de variance un, la variance du produit scalaire augmente avec d_k. Des scores de grande amplitude peuvent concentrer fortement le softmax et réduire ses gradients. L’argument motive le changement d’échelle ; il n’impose pas que les composantes apprises restent indépendantes.

Les coefficients d’attention décrivent une opération du réseau. Ils ne suffisent pas à établir une relation causale entre variables ni à expliquer toute la prédiction. La sortie dépend aussi des valeurs projetées et des couches qui suivent.

## 2.4 Utiliser plusieurs têtes d’attention

Une attention à plusieurs têtes répète ce calcul avec des projections différentes. Chaque tête produit une représentation ; les résultats sont concaténés puis projetés dans la dimension d_model. Le réseau peut ainsi calculer plusieurs types de relations à partir de la même entrée.

$$
\operatorname{MHA}(X)=\operatorname{Concat}(Y_1,\ldots,Y_h)W_O \tag{5a}
$$

$$
Y_r=\operatorname{Attention}\!\left(XW_Q^{(r)},XW_K^{(r)},XW_V^{(r)}\right) \tag{5b}
$$

La projection W_O transforme h × d_v coordonnées en d_model coordonnées.

Le choix fréquent d_k = d_v = d_model/h conserve une dimension totale constante pour l’ensemble des têtes. Avec d_model = 128 et h = 8, chaque tête peut par exemple produire 16 coordonnées. Leur concaténation en contient alors 128. Ajouter des têtes ne signifie donc pas nécessairement multiplier la dimension finale.

[[FIGURE_2]]

Figure 2 Attention à plusieurs têtes. Schéma original d’après Vaswani et al. (2017). Chaque tête utilise ses propres projections.

Les paramètres de chaque tête sont appris. On peut étudier leur spécialisation après l’entraînement, mais rien ne garantit qu’une tête représentera exclusivement une saisonnalité et une autre une tendance. Pour montrer qu’une tête joue un rôle particulier, il faut examiner son comportement ou mesurer l’effet de son retrait.

## 2.5 Donner une information sur l’ordre

L’attention calculée uniquement à partir du contenu ne distingue pas spontanément les positions. Sans encodage positionnel ni masque, une permutation des lignes de X entraîne la même permutation des lignes de sortie : le calcul est équivariant aux permutations. Pour utiliser l’ordre, le modèle doit donc recevoir une information supplémentaire.

Le Transformer original utilise notamment un encodage sinusoïdal ajouté aux embeddings. Les coordonnées paires et impaires représentent des sinus et des cosinus de fréquences différentes.

$$
\operatorname{PE}(p,2r)=\sin\!\left(\frac{p}{10000^{2r/d_{\mathrm{model}}}}\right) \tag{6a}
$$

$$
\operatorname{PE}(p,2r+1)=\cos\!\left(\frac{p}{10000^{2r/d_{\mathrm{model}}}}\right) \tag{6b}
$$

Pour d_model pair, r = 0, …, d_model/2 − 1 ; p est une position entière à partir de zéro.

Cette construction fournit un vecteur pour chaque position p. Certaines coordonnées varient rapidement, d’autres plus lentement. La formule reste calculable pour des positions plus grandes que celles rencontrées pendant l’apprentissage, sans garantir que le modèle saura exploiter un contexte beaucoup plus long.

Pour une série temporelle, la position dans une fenêtre doit aussi être distinguée de la date. Le dixième point peut correspondre à dix heures ou à dix jours selon l’échantillonnage. Si les mesures sont irrégulières, deux positions consécutives ne représentent même pas toujours la même durée. Des variables calendaires ou des écarts de temps peuvent donc compléter l’encodage des positions.

## 2.6 Empêcher l’accès au futur

Pendant l’apprentissage d’un décodeur autorégressif, les entrées sont décalées par rapport aux cibles. La position i utilise un élément déjà connu pour prédire le suivant. Un masque causal lui permet de consulter les positions j ≤ i et interdit les suivantes. Il est ajouté aux scores avant le softmax.

$$
M_{ij}=\begin{cases}0,&j\leq i,\\-\infty,&j>i.\end{cases} \tag{7a}
$$

$$
Y=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V \tag{7b}
$$

M a la taille n × n. Cette convention suppose des entrées du décodeur décalées.

Le masque a la taille n × n et porte sur les relations entre positions. Avec la convention décalée, la diagonale est autorisée. La valeur −∞ annule le poids d’une position interdite après le softmax. Il faut conserver au moins une clé accessible par requête pour que ce calcul reste défini. Un masque de remplissage peut être ajouté pour ignorer les emplacements artificiels utilisés dans un lot de séquences de longueurs différentes.

En prévision, un encodeur qui reçoit uniquement le passé peut consulter toute cette fenêtre sans fuite. La règle essentielle est de ne pas lui fournir une information indisponible à l’origine de la prévision. Un masque triangulaire dans toutes les couches n’est donc pas une condition nécessaire pour tous les modèles temporels.

## 2.7 Transformer et normaliser les représentations

L’attention fait circuler l’information entre positions. Le réseau feed-forward, ou FFN, transforme ensuite chaque vecteur séparément avec les mêmes paramètres à toutes les positions d’une couche. Une forme classique utilise deux projections affines et une activation ReLU.

$$
\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2 \tag{8}
$$

W₁ projette de d_model vers d_ff ; W₂ revient de d_ff vers d_model.

Une connexion résiduelle ajoute l’entrée d’un sous-bloc à sa sortie. Le sous-bloc apprend ainsi une modification de la représentation existante. Cette voie directe aide à transmettre l’information et les gradients lorsque plusieurs couches sont empilées. Elle impose aussi que les deux termes additionnés aient la même taille.

La normalisation par couche, appelée LayerNorm, utilise la moyenne et la variance des coordonnées d’un token. Elle recentre le vecteur, le remet à l’échelle et applique des coefficients appris γ et β. Le terme ε évite une division instable lorsque la variance est très faible (Ba et al., 2016).

$$
\operatorname{LN}(z)=\gamma\odot\frac{z-\mu(z)}{\sqrt{\sigma^2(z)+\varepsilon}}+\beta \tag{9a}
$$

$$
\mu(z)=\frac{1}{d_{\mathrm{model}}}\sum_{j=1}^{d_{\mathrm{model}}}z_j, \qquad \sigma^2(z)=\frac{1}{d_{\mathrm{model}}}\sum_{j=1}^{d_{\mathrm{model}}}\bigl(z_j-\mu(z)\bigr)^2 \tag{9b}
$$

γ et β sont appris ; ε > 0. Les statistiques portent sur les coordonnées d’un token.

LayerNorm ne normalise pas la série brute sur l’axe du temps. Elle agit sur les coordonnées d’une représentation, à une position donnée. Dans le bloc original, elle suit l’addition résiduelle. D’autres architectures la placent avant le sous-bloc. Les deux organisations ne doivent pas être confondues.

$$
Z=\operatorname{LN}\!\left(X+\operatorname{MHA}(X)\right), \qquad X^{\prime}=\operatorname{LN}\!\left(Z+\operatorname{FFN}(Z)\right) \tag{10}
$$

Bloc post-normalisation ; dropout omis. FFN et LN s’appliquent ligne par ligne.

## 2.8 Entraîner le réseau et produire une sortie

L’encodeur empile des blocs pour représenter l’entrée. Dans l’architecture encodeur-décodeur, le décodeur possède une auto-attention masquée et une attention croisée vers les sorties de l’encodeur. Un modèle à encodeur seul peut recevoir une tête de prévision ; un modèle à décodeur seul peut prolonger un contexte de façon causale. Le Transformer décrit donc une organisation du calcul, à laquelle il faut ajouter un objectif d’apprentissage et une sortie adaptée.

Pendant l’entraînement, une fonction de perte compare la sortie aux cibles. La rétropropagation calcule les gradients par rapport aux embeddings, aux projections d’attention, aux FFN et à la tête de sortie. Un optimiseur met à jour ces paramètres. Les coefficients d’attention sont recalculés à chaque passage : ils ne forment pas une matrice de paramètres conservée pour toutes les séquences.

Pendant l’apprentissage autorégressif, le décodeur reçoit les tokens réels décalés. À l’inférence, il utilise progressivement ses propres sorties, ce qui peut propager les erreurs. Une tête qui produit directement tous les horizons suit une autre procédure.

Le coût de l’attention dense croît quadratiquement avec n à dimensions fixées. La version standard matérialise h matrices de scores n × n ; les projections et les FFN ajoutent leur propre coût. Enfin, pouvoir entraîner plusieurs positions en parallèle ne supprime pas les étapes successives d’une génération autorégressive. Les adaptations temporelles cherchent justement à réduire le nombre de tokens, à modifier les relations calculées ou à produire plusieurs horizons dans un même passage.

# 3 Adapter les Transformers aux séries temporelles

## 3.1 Définir la tâche avant de choisir une architecture

La prévision consiste à exploiter un historique pour estimer des observations qui ne sont pas encore disponibles. On note L la longueur de cet historique, H l’horizon et C le nombre de variables cibles. À une date t, les données d’entrée forment donc une matrice de L lignes et C colonnes. Ces dimensions ne sont pas interchangeables : augmenter L étend la mémoire temporelle, tandis qu’augmenter C ajoute des variables susceptibles d’apporter une information complémentaire.

$$
\widehat{y}_{t+1:t+H}=f_\theta\!\left(y_{t-L+1:t},x_{t-L+1:t},z_{t+1:t+H},s\right) \tag{11}
$$

Dans cette écriture, y représente les cibles, x les covariables historiques, z les covariables futures connues à la date t et s les caractéristiques statiques. Les paramètres θ sont appris. Un calendrier futur peut être fourni au modèle ; une mesure réalisée après t ne peut pas l’être. De même, lorsqu’une covariable provient d’une autre prévision, il faut utiliser la version disponible à l’origine de prévision. Sans cette précaution, une architecture peut paraître performante simplement parce qu’elle reçoit une information inaccessible en pratique.

L’apprentissage se construit en faisant glisser des fenêtres de longueur L + H sur les séries. Les L premières observations servent d’entrée et les H suivantes fournissent la cible. On calcule une fonction de perte, puis la rétropropagation ajuste les paramètres. La séparation entre apprentissage, validation et test doit respecter la chronologie ; les fenêtres réservées au test ne servent ni à choisir les hyperparamètres ni à estimer les transformations globales des données.

Pour comparer les méthodes, on retient le même fil conducteur : comment les données sont préparées, quels éléments deviennent des tokens, quelles informations échangent ces tokens, puis comment le réseau produit la prévision. Cette lecture permet de comprendre ce que modifie réellement chaque architecture.

## 3.2 PatchTST et la représentation par segments

PatchTST normalise chaque canal, puis découpe son historique en segments de longueur P, espacés d’un pas S. Une projection transforme les segments en vecteurs de dimension d ; un encodage conserve leurs positions. La version supervisée ajoute S répétitions de la dernière valeur (Nie et al., 2023).

$$
N=\left\lfloor\frac{L-P}{S}\right\rfloor+2 \tag{12a}
$$

$$
E_c=P_cW_p+E_{\mathrm{pos}} \tag{12b}
$$

P_c a la taille N × P ; W_p a la taille P × d ; E_c et E_pos ont la taille N × d. La formule inclut le remplissage final de S valeurs.

N compte les segments, P_c est leur matrice pour le canal c, W_p la projection et E_pos l’encodage de position. L’attention relie les segments d’un même canal. Les paramètres sont partagés entre canaux, sans échange direct entre leurs représentations. Après l’attention, les réseaux feed-forward et les normalisations BatchNorm, la représentation aplatie est projetée vers H valeurs. L’entraînement minimise une erreur quadratique ; l’inférence produit l’horizon directement.

Dans un exemple pédagogique avec L = 96, P = 16 et S = 8, on obtient 12 tokens par canal. Une tête construit alors une matrice d’attention de 12 × 12. Le choix de P détermine les détails locaux regroupés avant l’attention ; celui de S règle le recouvrement et le nombre de tokens.

## 3.3 iTransformer et l’attention entre variables

iTransformer change l’axe sur lequel l’attention opère. L’historique complet d’une variable est projeté vers un token de dimension d. Les C tokens obtenus traversent un encodeur dont l’attention combine les représentations des variables. Le réseau feed-forward, partagé entre tokens, transforme ensuite chaque représentation. Une projection finale produit les H valeurs futures de chaque variable, et l’apprentissage supervisé utilise l’erreur quadratique (Liu et al., 2024).

$$
e_c^{(0)}=E(U_{:,c}) \tag{13a}
$$

$$
Z^{(\ell+1)}=B_\ell\!\left(Z^{(\ell)}\right), \qquad \ell=0,\ldots,K-1 \tag{13b}
$$

$$
\widehat{Y}_{:,c}=G\!\left(Z_{c,:}^{(K)}\right) \tag{13c}
$$

Z⁽⁰⁾ réunit les C tokens initiaux. Les matrices Z ont la taille C × d et Ŷ la taille H × C. La restitution de l’échelle est omise dans ce schéma latent.

E transforme un historique de longueur L en un vecteur ; Bℓ désigne le bloc numéro ℓ et G la projection vers l’horizon. Le nombre de blocs est K. Dans un exemple pédagogique avec C = 8 et d = 64, l’encodeur reçoit une matrice de 8 × 64, même si chaque variable possède 96 observations. L’attention compare alors huit historiques représentés par huit tokens, à travers une matrice de 8 × 8.

Cette attention coûte quadratiquement en C ; l’encodage dépend toujours de L. Les positions dans l’historique projeté conservent l’ordre temporel. Les relations entre variables sont explicites, sans comparaison de toutes les paires d’instants.

[[FIGURE_3]]

Figure 3 Deux façons de construire les tokens. Schéma original d’après Nie et al. (2023) et Liu et al. (2024). Le découpage est illustratif ; le remplissage en bord de fenêtre est omis.

## 3.4 Informer et la sélection des requêtes

Informer conserve une organisation encodeur-décodeur, mais réduit le calcul d’attention avec ProbSparse. Son principe est de consacrer le calcul détaillé aux requêtes dont les scores se distinguent le plus d’une distribution uniforme. Une approximation fondée sur un échantillonnage de clés estime cette propriété ; seules les requêtes retenues calculent ensuite une attention complète sur les clés. Dans l’encodeur, les autres positions conservent un contexte initial fondé sur les valeurs (Zhou et al., 2021).

Entre certains blocs, une convolution temporelle, une activation et un max-pooling réduisent la longueur de la représentation. Cette opération, appelée distillation, diminue le nombre de positions à traiter aux étages suivants. Le décodeur reçoit une portion récente de l’historique et des emplacements nuls pour les valeurs futures, accompagnés des informations temporelles disponibles. Une passe produit tout l’horizon ; l’apprentissage minimise l’erreur quadratique.

La complexité annoncée en O(L log L) concerne le mécanisme proposé. La méthode repose donc sur deux réductions différentes : sélectionner les requêtes et raccourcir les représentations. Pour comprendre une variation de performance, il faudrait les retirer séparément. Une sélection trop forte peut supprimer une dépendance utile, tandis que le pooling peut atténuer un événement bref.

## 3.5 Autoformer et les dépendances par décalage

Autoformer introduit la décomposition à l’intérieur du réseau. Un lissage par moyenne mobile extrait une composante lente ; la différence avec l’entrée constitue un résidu appelé composante saisonnière. Cette appellation décrit le rôle du bloc : elle ne garantit pas que le résidu soit une saisonnalité pure. Les décompositions sont répétées sur les représentations intermédiaires, et le décodeur accumule progressivement les contributions de tendance (Wu et al., 2021).

Le mécanisme Auto-Correlation remplace l’attention point par point. Les projections Q et K servent à mesurer des similarités pour plusieurs décalages temporels. Le calcul utilise des transformées de Fourier rapides. Le modèle retient les décalages les plus pertinents, décale circulairement les représentations V, puis additionne ces versions avec des poids normalisés par softmax. Il rapproche ainsi des sous-séries alignées selon un retard commun.

La sortie additionne les composantes prédites sur l’horizon, avec un apprentissage par erreur quadratique. L’hypothèse structurante est que des motifs décalés apportent une information réutilisable. Elle donne un sens temporel aux interactions apprises, mais invite à tester la sensibilité aux changements de période. La largeur du lissage et le nombre de retards retenus constituent également des choix méthodologiques à documenter.

## 3.6 Temporal Fusion Transformer et les covariables

Le Temporal Fusion Transformer, ou TFT, organise explicitement les variables statiques, historiques et futures connues. Les variables catégorielles passent par des embeddings et les variables continues par des projections. Des réseaux de sélection calculent des poids normalisés par softmax, puis combinent les représentations des variables. Ce choix dépend des entrées et, pour les variables temporelles, du contexte statique (Lim et al., 2021).

Des couches LSTM traitent ensuite les dépendances locales sur l’historique et les entrées futures disponibles. Les représentations sont enrichies par les informations statiques, puis une attention temporelle masquée combine les positions admissibles. Des portes apprises règlent la contribution des transformations non linéaires dans les connexions résiduelles. L’attention modifiée partage la projection des valeurs entre les têtes et agrège leurs contributions, afin de faciliter leur lecture.

La tête de sortie fournit plusieurs quantiles pour chaque horizon. La fonction de perte est donc la perte quantile, présentée dans la section sur les métriques, plutôt qu’une simple erreur quadratique. Le modèle produit ces quantiles directement à partir des entrées disponibles. Les poids de sélection renseignent sur son fonctionnement, mais leur amplitude ne suffit pas à établir une relation causale entre une covariable et la cible.

## 3.7 Normaliser les fenêtres et conserver des comparateurs simples

RevIN peut entourer un modèle de prévision. Pour chaque fenêtre et chaque canal, on calcule une moyenne μ_c et une échelle s_c, définie à partir de la variance et d’une petite constante ε. La fenêtre est centrée et réduite ; après la prévision, les mêmes statistiques restaurent les unités initiales. Dans la forme sans transformation affine apprise, on obtient les relations suivantes (Kim et al., 2022).

$$
\widetilde{y}_{r,c}=\frac{y_{r,c}-\mu_c}{s_c} \tag{14a}
$$

$$
\widehat{y}_{t+h,c}=s_c\widehat{\widetilde{y}}_{t+h,c}+\mu_c \tag{14b}
$$

r parcourt le contexte ; h = 1, …, H. s_c est la racine de la variance du contexte augmentée de ε > 0. La forme affichée omet les paramètres affines appris.

Les statistiques proviennent uniquement du contexte. Une version affine ajoute une échelle et un décalage appris, inversés à la sortie. Cette normalisation facilite la comparaison de fenêtres de niveaux différents. Elle n’annonce toutefois pas une rupture future : elle replace les sorties à l’échelle du passé observé. Une expérience doit donc préciser si RevIN est présent dans tous les modèles comparés.

DLinear permet de vérifier l’utilité des blocs complexes. Le modèle calcule une tendance lissée et un résidu, applique à chacun une projection linéaire de L vers H, puis additionne les sorties. Il apprend ces projections en minimisant une erreur de prévision. Il n’y a ni tokens ni attention. Les résultats de Zeng et al. (2023) montrent l’intérêt de cette référence dans les configurations étudiées. Une comparaison utile conserve le même horizon et examine le réglage du contexte, car une différence de longueur d’entrée peut masquer l’effet propre de l’architecture.

DeepAR fournit un autre point de comparaison : un réseau récurrent partagé entre séries produit les paramètres d’une distribution conditionnelle. L’apprentissage maximise la vraisemblance des observations. À l’inférence, des valeurs sont échantillonnées successivement et réinjectées pour construire des trajectoires (Salinas et al., 2020). Cette procédure se distingue des sorties directes précédentes. Elle montre aussi que le partage de connaissances entre séries ne commence pas avec les Transformers.

## 3.8 Du modèle supervisé au modèle préentraîné

Les architectures décrites jusqu’ici répondent à des problèmes distincts. Modifier les tokens change les interactions accessibles ; sélectionner des requêtes change le calcul ; introduire une décomposition change les hypothèses sur le signal. Il est donc préférable d’étudier ces choix séparément, avec des ablations, avant d’attribuer un gain à la seule présence d’attention.

Le préentraînement ajoute une autre dimension : le modèle apprend sur un ensemble de séries avant son utilisation sur une tâche cible. Un objectif autorégressif apprend une continuation ; un objectif par masquage apprend à reconstruire des observations retirées. Les cibles sont construites à partir des séries elles-mêmes. Ces procédures restent des apprentissages avec une fonction de perte, même lorsqu’elles ne demandent aucune annotation manuelle.

En zero-shot, les paramètres restent fixes sur la tâche cible. En fine-tuning, une optimisation modifie tout ou partie de ces paramètres. Fournir davantage d’historique au modèle ne constitue donc pas un ajustement fin. Enfin, l’absence d’adaptation ne prouve pas que le modèle n’a jamais rencontré les données cibles pendant son préentraînement. Cette différence devient centrale lorsque l’on passe des architectures supervisées aux modèles de fondation.

Un modèle de fondation peut rester gelé alors qu’une régression auxiliaire, une correction résiduelle ou des poids d’ensemble sont ajustés sur la tâche cible. Il faut alors distinguer l’absence d’adaptation du réseau préentraîné de l’apprentissage réalisé par le système complet. La qualification zero-shot doit préciser le composant auquel elle s’applique.

LoRA conserve les poids préentraînés W₀ et apprend une correction de faible rang ΔW = BA. Pour une matrice de taille d × k, seules deux matrices totalisant r(d + k) paramètres sont entraînées, avec r très inférieur aux dimensions initiales (Hu et al., 2022). Cette économie facilite l’adaptation ; celle-ci reste un apprentissage sur les données cibles, même lorsque W₀ demeure figée.

## 3.9 Mesurer la qualité des prévisions

Sur N observations, la MAE moyenne les erreurs absolues ; la RMSE prend la racine de la moyenne des erreurs quadratiques. Les deux conservent l’unité de la cible, mais la RMSE pénalise davantage les grandes erreurs. La perte absolue cible une médiane conditionnelle, la perte quadratique une moyenne : le score doit donc être lu avec la sortie évaluée (Hyndman et Athanasopoulos, 2021).

$$
\operatorname{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\widehat{y}_i|, \qquad \operatorname{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\widehat{y}_i)^2} \tag{15}
$$

La MASE rapporte la MAE de test à une erreur naïve calculée sur les T observations d’entraînement. Les différences sont espacées de m pas, avec m égal à la période saisonnière, ou à 1 sans saisonnalité ; T doit dépasser m (Hyndman et Koehler, 2006 ; Hyndman et Athanasopoulos, 2021). Une MASE inférieure à 1 signifie un gain face à cette référence historique, sans garantir un gain face à la naïve sur le test.

$$
\operatorname{MASE}=\frac{\displaystyle\frac{1}{H}\sum_{h=1}^{H}|y_{T+h}-\widehat{y}_{T+h}|}{\displaystyle\frac{1}{T-m}\sum_{t=m+1}^{T}|y_t-y_{t-m}|} \tag{16}
$$

Un dénominateur nul ne donne pas de MASE finie exploitable, et 0/0 reste indéfini. Une valeur presque nulle rend aussi le score instable. Ces cas et leur traitement doivent être signalés ; ajouter une constante change la mesure.

Pour un quantile de niveau τ, la perte pinball pondère une sous-estimation par τ et une surestimation par 1 − τ. Son espérance est minimisée par un quantile de la cible. Les niveaux évalués et leur agrégation doivent être précisés (Gneiting et Raftery, 2007).

$$
\rho_\tau(y-q)=(y-q)\bigl(\tau-\mathbf{1}_{\{y<q\}}\bigr), \qquad 0<\tau<1 \tag{17}
$$

Le CRPS intègre l’écart quadratique entre la probabilité F(z) et l’indicatrice de y ≤ z. Plus il est faible, meilleur il est. Pour les distributions de premier moment fini, il est strictement propre : annoncer la distribution réelle minimise le score espéré. Une prévision déterministe ramène le CRPS à l’erreur absolue (Gneiting et Raftery, 2007).

$$
\operatorname{CRPS}(F,y)=\int_{-\infty}^{+\infty}\left[F(z)-\mathbf{1}_{\{y\leq z\}}\right]^2\,\mathrm{d}z \tag{18}
$$

Lorsqu’un modèle fournit seulement une grille finie de quantiles, le score calculé à partir de cette grille constitue une approximation du CRPS ou une perte quantile agrégée, selon la convention retenue. Les niveaux, pondérations, normalisations et règles d’interpolation doivent être identiques ou explicitement documentés pour comparer les modèles (Gneiting et Raftery, 2007 ; Shchur et al., 2025).

Ces scores évaluent les marges, sans vérifier la dépendance entre dates. Des scénarios complets demandent également une évaluation de leur loi jointe.

# 4 Méthodologies des modèles de fondation temporels

## 4.1 Du préentraînement à une nouvelle prévision

Les modèles de fondation temporels cherchent à réutiliser un apprentissage réalisé sur de nombreuses séries. Une fenêtre d'historique devient l'entrée du modèle ; les observations suivantes fournissent la cible d'apprentissage. En répétant cette opération sur des domaines et des fréquences variés, le réseau apprend une fonction de prévision partageable. Lorsqu'il reçoit une nouvelle série, son contexte change, mais ses paramètres peuvent rester fixes : c'est le fonctionnement en zero-shot. Une adaptation par gradient constitue une étape différente, qui doit être annoncée séparément.

Pour comparer les méthodes, on retient quatre questions : comment les nombres sont-ils représentés, quelles informations peuvent interagir, quelle perte est minimisée et comment le futur est-il construit ? La sortie mérite une attention particulière. Une valeur centrale, neuf quantiles et plusieurs trajectoires ne décrivent pas exactement le même objet statistique. Le choix de l'architecture doit donc être lu avec celui de l'objectif et du décodage.

La littérature retenue s'arrête au 9 septembre 2026. Les méthodes fondatrices de TimesFM et Moirai sont publiées à ICML 2024 ; Time-MoE à ICLR 2025 ; Sundial et Moirai-MoE à ICML 2025 ; TiRex à NeurIPS 2025. Chronos-2, Moirai 2.0, Timer-S1 et Toto 2.0 sont étudiés dans les versions de leurs rapports techniques référencées en bibliographie. Pour TimesFM-3, la source utilisée est une annonce officielle datée du 31 août 2026. Ce dernier cas permet de discuter la méthode annoncée, avec un recul expérimental plus limité.

## 4.2 Chronos et TimesFM face à la représentation des valeurs

Chronos transforme la prévision en classification séquentielle. Chaque série est divisée par la moyenne des valeurs absolues de son contexte, puis les nombres normalisés sont associés à des classes d'un vocabulaire fini. Ces identifiants alimentent notamment une architecture T5 encodeur–décodeur, entraînée sur des séries temporelles depuis une initialisation aléatoire. Le modèle ne réutilise donc pas nécessairement des connaissances apprises sur du texte. Son objectif est l'entropie croisée des tokens futurs (Ansari et al., 2024).

$$
\mathcal{L}_{\mathrm{CE}}(\theta)=-\sum_{k=1}^{K}\log p_\theta\!\left(z_k\mid c,z_{1:k-1}\right) \tag{19}
$$

K est ici le nombre de tokens futurs. La somme porte sur une séquence cible ; l’entraînement agrège cette perte sur les fenêtres du lot.

Dans cette écriture, c désigne le contexte tokenisé, K le nombre de tokens à prévoir, zₖ le token futur observé et θ les paramètres. À l'entraînement, les tokens précédents corrects sont fournis au décodeur. À l'inférence, les nouveaux tokens sont échantillonnés puis réinjectés pour poursuivre la trajectoire. La conversion inverse vers des nombres et le rétablissement de l'échelle donnent une prévision. Plusieurs répétitions fournissent des scénarios et leurs quantiles. Cette méthode reste limitée par la finesse des classes, leur plage de valeurs et le coût des générations successives (Ansari et al., 2024).

Cette formulation pose une question de représentation : deux classes voisines correspondent à des nombres proches, mais l'objectif reste catégoriel. Le vocabulaire doit donc être étudié avec la remise à l'échelle.

TimesFM conserve au contraire des entrées continues. Des observations consécutives forment des patches, transformés en vecteurs par un petit réseau résiduel. Un codage de position leur est ajouté, puis un Transformer à attention causale calcule leurs représentations. Une seconde projection transforme chaque représentation en bloc de valeurs futures. La taille du bloc de sortie peut dépasser celle du patch d'entrée : une unité interne peut donc apprendre à prévoir plusieurs unités futures. L'article initial optimise une erreur quadratique et se concentre sur la prévision ponctuelle (Das et al., 2024).

L'apprentissage masque aléatoirement le début du premier patch pour exposer le réseau à différentes longueurs de contexte, y compris celles qui ne sont pas des multiples de la taille des patches. À l'inférence, les blocs sont produits successivement si l'horizon demandé dépasse celui de la tête de sortie. Le regroupement réduit le nombre de tokens et d'appels, mais le décodage prolongé utilise encore des prévisions antérieures comme entrées. La robustesse à cette différence entre apprentissage et inférence reste donc à évaluer (Das et al., 2024).

TimesFM 2.5, diffusé en septembre 2025, compte 200 millions de paramètres et accepte jusqu'à 16 000 observations de contexte. La documentation décrit une tête optionnelle de 30 millions de paramètres pour prévoir des quantiles jusqu'à un horizon de 1 000 points. Les covariables sont prises en charge par XReg, un complément de régression au modèle temporel. Ces éléments décrivent une version logicielle et ses capacités documentées ; ils ne doivent pas être attribués sans distinction au protocole de l'article de 2024 (Google Research, 2026).

## 4.3 Moirai et Chronos-2 pour structurer les variables

Moirai 1 traite la prévision comme le remplissage d'une partie masquée. Les séries sont normalisées, découpées en patches et réunies dans une même séquence de tokens. Les patches futurs des cibles sont remplacés par une représentation apprise du masque ; les covariables effectivement connues restent disponibles. Un encodeur traite cet ensemble. L'attention ajoute un biais différent selon que deux tokens appartiennent à la même variable ou à deux variables distinctes, en complément du repérage temporel. Le modèle peut ainsi recevoir un nombre variable de composantes, sans associer chaque position de variable à une signification fixe (Woo et al., 2024).

La projection de sortie fournit les paramètres d'un mélange de distributions : Student, binomiale négative, lognormale et normale de faible variance. Le préentraînement minimise la log-vraisemblance négative des observations futures et fait varier contexte et horizon. Plusieurs tailles de patches sont proposées selon les fréquences. L'aplatissement des variables augmente toutefois le nombre total de tokens : avec C variables et N patches chacune, l'attention complète considère une séquence de longueur CN. Cette flexibilité a donc un coût lorsque C devient grand (Woo et al., 2024).

Moirai 2.0 modifie cette démarche. Les patches continus et leurs indicateurs de présence alimentent désormais un décodeur causal. La tête produit plusieurs patches futurs et neuf quantiles, entraînés avec la perte pinball. Le masquage aléatoire de patches cherche à rendre le modèle moins dépendant d'un contexte intégralement observé. Pour éviter une fuite par la normalisation pendant le préentraînement causal, les statistiques sont calculées sur les premiers 30 % de la séquence, les 70 % restants servant à la prédiction. Pour prolonger l'horizon, le rapport propose un décodage récursif à partir des quantiles (C. Liu et al., 2025).

Cette simplification s'accompagne d'une restriction : les variables sont traitées indépendamment et le support des covariables est retiré. Moirai 2.0 ne représente donc pas une extension de toutes les capacités de Moirai 1. Le changement de méthode privilégie la prévision univariée probabiliste et l'efficacité. Une grille de quantiles renseigne les marges, mais ne définit pas à elle seule une distribution jointe de toutes les dates ; la procédure utilisée pour prolonger les sorties devient déterminante (C. Liu et al., 2025).

Chronos-2 rompt aussi avec la représentation de sa première génération. Les valeurs sont standardisées, transformées par la fonction sinus hyperbolique inverse (arsinh), puis regroupées en patches continus avec un indice temporel et un masque d'observation. Les futures cibles sont indiquées comme manquantes. L'architecture est un encodeur seul qui alterne attention temporelle au sein d'une série et attention entre les séries d'un même groupe, au même indice de patch. Un groupe peut réunir plusieurs cibles et leurs covariables. La tête fournit directement les quantiles des patches futurs, et la perte pinball porte uniquement sur les cibles observées de l'apprentissage (Ansari et al., 2025).

Les covariables catégorielles sont converties en valeurs numériques avant l’entrée dans le réseau : target encoding pour une cible univariée, encodage ordinal dans le cas multivarié décrit par l’article. Cette prise en charge repose sur un encodage numérique des catégories, sans traitement natif de covariables textuelles (Ansari et al., 2025).

Le partage d’information entre variables modifie le conditionnement des prévisions. Il ne suffit pas à définir la dépendance des erreurs futures : une tête de quantiles fournit encore des quantiles marginaux, par variable et par horizon. Cette distinction sera reprise dans la discussion des trajectoires probabilistes en section 5.5 (Ansari et al., 2025 ; Perez-Diaz et al., 2025).

Les tâches multivariées du préentraînement sont construites synthétiquement en imposant des dépendances entre séries. L'enjeu est d'apprendre à exploiter une relation visible dans le contexte sans réentraîner le réseau sur chaque tâche. Le masque précise quelles valeurs futures sont connues ; l'identifiant de groupe précise quelles séries peuvent échanger de l'information. La présence de ces mécanismes ne garantit cependant pas que les dépendances du corpus synthétique couvrent celles d'un nouveau domaine (Ansari et al., 2025).

L'annonce de TimesFM-3 décrit une autre combinaison : patches de 32 points, décodeur, attention temporelle causale et attention entre variables alternées. Les tokens des covariables connues dans le futur incluent des patches à venir. Le modèle reçoit aussi des patches futurs masqués et fournit neuf quantiles par cible. Son préentraînement est présenté comme nativement multivarié, sur des données réelles et synthétiques, avec un décodage de l'horizon en une passe. La publication consultée décrit ces choix, sans documenter une recette d'optimisation aussi complète qu'un article méthodologique. On retient donc cette évolution comme une piste récente, sans reprendre ses classements comme preuve indépendante de supériorité (Jain et Sen, 2026).

## 4.4 Sundial et la génération continue du futur

Une prévision probabiliste peut aussi être obtenue en apprenant une transformation entre du bruit et des observations. Le flow matching entraîne un réseau à prédire un champ de vitesse : à partir d'un point intermédiaire, il indique dans quelle direction le déplacer. Pour comprendre le principe, on considère un chemin linéaire entre un bruit gaussien ε et un bloc réel y. Le paramètre τ appartient à l'intervalle [0, 1] ; il mesure l'avancement de cette transformation et ne représente pas une date de la série (Lipman et al., 2023).

$$
\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{\tau,\varepsilon,y,h}\!\left[\left\|v_\theta(x_\tau,\tau,h)-(y-\varepsilon)\right\|_2^2\right] \tag{20}
$$

τ est uniforme sur [0, 1] et ε est un bruit gaussien N(0, I). h désigne ici la représentation du passé. La norme euclidienne est mise au carré.

Le point intermédiaire xτ vaut (1 − τ)ε + τy. La vitesse cible le long de ce chemin est y − ε. Le réseau vθ reçoit xτ, τ et, dans le cas conditionnel, une représentation h du passé. La perte compare sa vitesse à la vitesse cible. Lors de la génération, y est inconnu : on part d'un nouveau bruit et l'on intègre numériquement le champ appris. Changer le bruit initial produit un autre échantillon, pour un même contexte. Le préentraînement n'exige pas de simuler toute l'intégration à chaque mise à jour (Lipman et al., 2023).

Sundial applique ce principe à la prévision temporelle avec la perte TimeFlow. Il normalise la série, encode ses patches continus et calcule h avec un Transformer causal. Un petit réseau de flow matching, entraîné conjointement, transforme ensuite un bruit en bloc futur conditionné par h. Le même historique encodé est réutilisé pour générer plusieurs échantillons. L'entraînement porte sur TimeBench, un corpus comprenant des observations réelles et synthétiques. Les trajectoires permettent d'estimer médiane, quantiles et autres statistiques (Y. Liu et al., 2025).

Le décodage comporte ainsi deux progressions : les étapes numériques qui produisent un bloc, puis la prolongation temporelle si plusieurs blocs sont nécessaires. L'intérêt est de modéliser des sorties continues sans quantification ni mélange fixé de lois simples. En contrepartie, il faut choisir le nombre d'étapes d'intégration et d'échantillons. Ces choix influencent le coût et l'approximation obtenue. Sundial suit par ailleurs un préentraînement univarié : une génération probabiliste souple ne signifie pas automatiquement que les dépendances entre variables sont apprises (Y. Liu et al., 2025).

## 4.5 Les mélanges d'experts et la spécialisation des calculs

Dans un mélange d'experts, ou MoE, le réseau ne mobilise qu'une partie de ses sous-réseaux pour chaque token. Le routeur calcule des scores, sélectionne quelques experts et combine leurs sorties. La capacité stockée peut ainsi être élevée alors que le calcul par token reste limité. Cette séparation concerne surtout les transformations internes ; elle ne dispense pas de calculer les interactions temporelles de l'attention.

Time-MoE représente chaque point par un vecteur continu et utilise un décodeur causal. Dans ses couches, un ensemble d'experts remplace le réseau feed-forward classique. Un expert partagé est complété par des experts sélectionnés selon leurs scores. Plusieurs têtes prédisent des horizons différents ; l'apprentissage agrège leurs pertes de Huber, quadratiques près de zéro et linéaires pour les grandes erreurs, avec une pénalisation destinée à équilibrer l'utilisation des experts. Le corpus Time-300B contient plus de 300 milliards de points (Shi et al., 2025).

À l'inférence, l'organisation des différentes têtes permet de construire l'horizon demandé. Il s'agit principalement de prévisions ponctuelles, et les variables sont traitées indépendamment. Le routage peut se concentrer excessivement sur certains experts, d'où la perte auxiliaire. Par ailleurs, un token par observation conserve une séquence plus longue qu'une représentation par patches. Le nombre de paramètres activés ne résume donc ni la mémoire nécessaire ni le coût complet de la prévision (Shi et al., 2025).

Moirai-MoE utilise au contraire des patches et une projection d'entrée commune aux fréquences. Son décodeur causal apprend à prédire la distribution du patch suivant. Des couches MoE remplacent les feed-forward et orientent les tokens vers des experts spécialisés. Le travail étudie notamment un routage guidé par des centres de groupes obtenus à partir de représentations préentraînées. La tête produit les paramètres d'un mélange de distributions, entraîné par log-vraisemblance négative. La spécialisation se fait ainsi au niveau des motifs représentés, au lieu de dépendre uniquement d'une catégorie de fréquence (X. Liu et al., 2025).

L'interprétation demande néanmoins de la prudence. Un expert sollicité sur certains patches n'est pas forcément un détecteur d'une composante bien identifiée, comme la tendance. Pour étayer cette lecture, il faudrait vérifier que les affectations se maintiennent après un changement d'échelle, un rééchantillonnage ou l'ajout de bruit. Ce type d'analyse aide à distinguer une spécialisation utile d'un simple découpage accidentel du corpus.

## 4.6 La construction des horizons longs

Timer-S1 déplace une partie de la prévision successive dans la profondeur du modèle. Après normalisation et projection de patches univariés, des blocs TimeMoE calculent les représentations. Des blocs TimeSTP les reprennent ensuite en consultant aussi les entrées initiales ; chaque étape ajoute un décalage de prévision. Une tête partagée fournit des quantiles. L'horizon long bénéficie ainsi de transformations sérielles supplémentaires à l'intérieur d'un appel, sans réintroduire systématiquement chaque valeur prévue dans le contexte brut (Y. Liu et al., 2026).

Le préentraînement est suivi d'une poursuite de l'apprentissage avec une pondération privilégiant les horizons proches, puis d'une extension du contexte. Le rapport présente 8,3 milliards de paramètres au total, dont 0,75 milliard activé par token. Cette capacité reste orientée vers des séries traitées indépendamment. La méthode permet surtout d'étudier où placer le calcul : dans plusieurs appels, plusieurs sorties parallèles ou plusieurs transformations successives. Une seule passe ne signifie donc pas un calcul sans dépendances internes (Y. Liu et al., 2026).

TiRex apporte un contrepoint avec une architecture xLSTM. Des patches, accompagnés d'indicateurs de valeurs manquantes, alimentent un réseau récurrent ; une tête apprend neuf quantiles avec la perte pinball. Le préentraînement masque des blocs consécutifs. À l'inférence, les patches futurs restent indiqués comme manquants et l'état interne propage l'information nécessaire à la prévision, au lieu de recevoir systématiquement la médiane prédite comme une observation. La mémoire récurrente remplit donc une fonction différente de la mémoire accessible par attention (Auer et al., 2025).

Toto 2.0 reprend le masquage contigu dans un Transformer par patches dont les attentions temporelle et intervariables alternent. La normalisation robuste et les projections résiduelles préparent les entrées ; une tête produit neuf quantiles avec une perte pinball, remplaçant le mélange de Student de la génération précédente. Pendant l'apprentissage, des portions consécutives sont masquées ; à l'inférence, elles correspondent à l'horizon à prévoir. Les sorties peuvent être calculées en une passe ou par blocs successifs pour les horizons longs (Khwaja et al., 2026).

La famille est entraînée sur des séries d'observabilité et des données synthétiques, avec une procédure de transfert d'hyperparamètres entre tailles. Les grands modèles sont entraînés plus longtemps que les deux plus petits : leurs résultats ne constituent donc pas une comparaison à calcul constant. Le choix du masquage, de l'optimisation et du décodage participe aux effets observés ; attribuer toute amélioration à la seule taille du Transformer serait insuffisant (Khwaja et al., 2026).

Ces méthodes conduisent à une même exigence pour l'évaluation : définir ce qui est tenu constant. Une comparaison d'architectures doit conserver autant que possible les données, l'information accessible, la cible statistique et le budget. Une comparaison de systèmes complets peut accepter leurs différences, mais doit les documenter. Cette distinction sera centrale pour interpréter les limites et les perspectives présentées dans les chapitres suivants.

# 5 Limites des modèles et de leur évaluation

Les résultats présentés jusqu'ici montrent l'intérêt des modèles de fondation, mais leur interprétation demande quelques précautions. Un bon score peut dépendre des données déjà rencontrées, des informations fournies au modèle ou de la métrique retenue. Ce chapitre examine ces difficultés à partir des travaux disponibles au 9 septembre 2026.

## 5.1 Vérifier ce que le modèle a déjà rencontré

Le zero-shot désigne généralement une prévision sans mise à jour des paramètres sur la tâche cible. Il ne garantit pas que les données étaient absentes du préentraînement. Meyer et al. distinguent le recouvrement d'échantillons et le chevauchement temporel entre séries corrélées. Deux séries différentes peuvent, par exemple, décrire le même événement exceptionnel. Contrôler les noms des jeux de données ne suffit donc pas : il faut aussi examiner leurs périodes et leurs relations. Cette analyse identifie un risque d'estimation optimiste ; elle ne signifie pas que tous les modèles sont contaminés (Meyer et al., 2025).

GIFT-Eval sépare l'exposition au segment d'entraînement de l'exposition au test. Dans sa documentation, l'étiquette zero-shot exclut les deux, tandis qu'un indicateur distinct signale la fuite vers le test. Ces catégories précisent la lecture du classement. Le dépôt fournit également un corpus de préentraînement conçu pour ne pas recouvrir son test. Il reste néanmoins nécessaire de distinguer la provenance déclarée des données d'une vérification indépendante (Salesforce AI Research, 2026).

TSFMAudit propose justement une méthode complémentaire. Les auteurs réalisent une courte adaptation exploratoire et observent conjointement la baisse de la perte et la modification du modèle. Une amélioration rapide avec peu de changements peut signaler une exposition antérieure. L'étude porte sur six modèles et 187 jeux de données. Ce signal reste à interpréter avec prudence : une tâche facile ou proche du domaine appris peut également être assimilée rapidement. Il permet d'orienter une vérification, sans établir à lui seul l'origine des données (Li et al., 2026).

## 5.2 Évaluer plusieurs tâches et des observations nouvelles

TIME rassemble 50 jeux de données et 98 tâches. Qiao et al. cherchent à renouveler les sources, contrôler leur qualité et analyser les résultats selon les propriétés temporelles. Cette démarche aide à repérer une difficulté sur les signaux irréguliers, par exemple, que masquerait une moyenne par domaine. Cependant, un jeu nouveau aujourd'hui peut intégrer le préentraînement d'un modèle ultérieur (Qiao et al., 2026).

Impermanent s'appuie sur des flux actualisés d'activité GitHub. Les prévisions sont évaluées au fil des nouvelles observations. Il devient ainsi possible de suivre la stabilité des performances et les changements de distribution. Le domaine reste particulier : une bonne généralisation sur l'activité de projets logiciels ne suffit pas à conclure pour toutes les séries temporelles (Garza et al., 2026).

On retient de ces travaux qu'il faut distinguer le transfert entre domaines sur des archives et la prévision d'observations réellement futures. Une dégradation sur un nouveau flux peut venir d'un changement du phénomène, sans révéler une contamination antérieure. Des références simples, évaluées sur les mêmes périodes, permettent de mieux interpréter cette dégradation.

## 5.3 Comparer les informations disponibles et les coûts

fev-bench comprend 100 tâches issues de sept domaines, dont 46 avec covariables. Ses scores de gain renseignent l’ampleur des écarts entre modèles. Ses intervalles reposent sur un bootstrap apparié des tâches et renseignent la stabilité des comparaisons lorsque la composition du benchmark varie. Ils ne quantifient pas, à eux seuls, toute l’incertitude liée aux observations futures ou aux répétitions d’entraînement. La comparaison suppose aussi des entrées comparables. Si un modèle reçoit une covariable future connue à l'origine et l'autre uniquement la cible, l'écart mesure l'ensemble du système, sans isoler l'effet de l'architecture (Shchur et al., 2025).

Guibert et al. comparent dix modèles sur deux ensembles de données en mesurant précision, durée et énergie, à contexte et horizon fixés. Les compromis observés encouragent à présenter le coût en regard de la qualité. Une mesure reste toutefois liée au matériel, au traitement par lots et aux opérations comptabilisées. Le coût d'une inférence ne couvre pas celui du préentraînement, de l'adaptation ou des essais d'hyperparamètres (Guibert et al., 2026).

## 5.4 Distinguer adaptation, oubli et robustesse

Laglil et al. comparent Chronos-2 et TTM-R3-PT gelés à un ajustement complet et à des adaptations LoRA, avec au maximum 4 000 étapes. Ils rapportent des améliorations après adaptation, tout en précisant que certains segments d'entraînement ont pu être rencontrés pendant le préentraînement. Comparer chaque modèle à sa propre version adaptée limite ce déséquilibre, mais n'établit pas un transfert vers un domaine entièrement inédit. Les conclusions restent liées aux deux modèles étudiés (Laglil et al., 2026).

L'apprentissage continu pose une autre difficulté. Karaouli et al. montrent qu'une amélioration sur une nouvelle tâche peut s'accompagner d'un oubli des précédentes. Pour évaluer des adaptations successives, il faut donc conserver des tâches anciennes et mesurer leur évolution. Une amélioration locale ne suffit pas à décrire la qualité du modèle après adaptation (Karaouli et al., 2025).

Enfin, Zhang et al. étudient six modèles soumis à des perturbations adversariales. Leur sensibilité varie notamment selon la position des observations modifiées et la longueur du contexte ; un ajustement adversarial peut la réduire dans le cadre étudié. Ces attaques intentionnelles ne reproduisent pas directement les valeurs manquantes ordinaires ou une dérive naturelle. Elles testent une fragilité précise, sans en donner la fréquence dans des conditions courantes (Zhang et al., 2025).

## 5.5 Tenir compte de la forme des prévisions

Perez-Diaz et al. rappellent que des distributions marginales ne déterminent pas la dépendance entre horizons. Deux variables de Bernoulli de probabilité un demi illustrent cette différence : elles valent simultanément un avec probabilité un quart si elles sont indépendantes, contre un demi si elles sont identiques. Les marges sont pourtant les mêmes. Une bande de quantiles ne permet donc pas, à elle seule, de construire des scénarios cohérents pour un maximum, une somme ou un dépassement persistant (Perez-Diaz et al., 2025).

Wan et al. examinent des prévisions ponctuelles presque plates, qui préservent mal certaines relations entre séries. Leur analyse relie ce comportement à la prévisibilité de la cible et à l'objectif d'apprentissage. Lorsque le signal prévisible est faible, une sortie lisse peut rester compatible avec la minimisation de l'erreur quadratique. Son apparence ne suffit donc pas à juger le modèle : il faut vérifier les propriétés requises par la tâche, comme le classement entre séries (Wan et al., 2026).

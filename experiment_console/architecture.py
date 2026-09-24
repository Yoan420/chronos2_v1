"""Read-only, results-oriented explanation of the configured NuclearKalman path.

Only an allowlist of public model settings is returned. No model, data source,
checkpoint tensor or scientific runtime is loaded by this module.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ZONES = ("BE", "DE", "FR", "NL")


def build_architecture(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    warnings: list[str] = []

    def load_config(relative) -> dict:
        if not isinstance(relative, str):
            return {}
        try:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 2_000_000:
                raise ValueError("Configuration indisponible")
            data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, yaml.YAMLError):
            warnings.append("Une configuration du parcours est indisponible ; ses paramètres sont indiqués comme inconnus.")
            return {}

    def evidence(relative: str, marker: str, label: str) -> dict:
        line = None
        try:
            text = (root / relative).read_text(encoding="utf-8-sig")
            line = next((index for index, text_line in enumerate(text.splitlines(), 1) if marker in text_line), None)
        except OSError:
            pass
        return {"path": relative, "line": line, "label": label}

    def nested(mapping, *keys):
        value = mapping
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    def common(values):
        return values[0] if values and values[0] is not None and all(value == values[0] for value in values) else None

    def numeric(value):
        return value if isinstance(value, (float, int)) and not isinstance(value, bool) and abs(value) < 10**9 else None

    settings = load_config("config/nuclear_forecast.yaml")
    activation = load_config(settings.get("lora_activation_config"))
    kalman = load_config(settings.get("kalman_config"))
    configured_zones = settings.get("zone_configs", {})
    zone_configs = {zone: load_config(configured_zones.get(zone)) for zone in ZONES} if isinstance(configured_zones, dict) else {zone: {} for zone in ZONES}
    contexts = {zone: numeric(nested(config, "model", "context_length")) for zone, config in zone_configs.items()}
    residual_enabled = {zone: nested(config, "hourly", "residual_correction", "enabled") is True if nested(config, "hourly", "residual_correction", "enabled") is not None else None for zone, config in zone_configs.items()}
    modes = {zone: nested(activation, "zones", zone, "enabled_modes") for zone in ZONES}
    lora_by_zone = {zone: bool(set(value) & {"autonomous", "kalman"}) if isinstance(value, list) and all(isinstance(item, str) for item in value) else None for zone, value in modes.items()}
    lora_active = True if any(value is True for value in lora_by_zone.values()) else False if all(value is False for value in lora_by_zone.values()) else None
    model_ids = [nested(config, "model", "model_id") for config in zone_configs.values()]
    model_id = "amazon/chronos-2" if common(model_ids) == "amazon/chronos-2" else None
    backend = common([nested(config, "hourly", "residual_correction", "backend") for config in zone_configs.values()])
    residual_backend = "CatBoost" if backend == "catboost" else None
    trees = common([numeric(nested(config, "hourly", "residual_correction", "iterations")) for config in zone_configs.values()])
    depth = common([numeric(nested(config, "hourly", "residual_correction", "depth")) for config in zone_configs.values()])
    context = common(list(contexts.values()))
    mode = settings.get("computation_mode")
    mode = mode if isinstance(mode, str) and mode in {"incremental", "full"} else None
    if lora_active:
        warnings.append("Une activation LoRA est renseignée : le lanceur nucléaire refuse cette combinaison. Le parcours nécessite une configuration compatible.")
    if model_id is None:
        warnings.append("Le modèle de base commun aux quatre zones n'est pas confirmé comme amazon/chronos-2.")

    transformer_verified = False
    try:
        spec = importlib.util.find_spec("chronos")
        if spec and spec.origin:
            model_path = Path(spec.origin).parent / "chronos2" / "model.py"
            implementation = model_path.read_text(encoding="utf-8")
            transformer_verified = all(marker in implementation for marker in (
                "class Chronos2EncoderBlock", "TimeSelfAttention(config)", "GroupSelfAttention(config)",
                "FeedForward(config)", "self.input_patch_embedding", "self.output_patch_embedding",
            ))
    except (OSError, ImportError, ValueError):
        pass
    if not transformer_verified:
        warnings.append("Le détail interne du Transformer n'a pas pu être vérifié dans le paquet Chronos de cet environnement.")

    def node(identifier, title, kicker, description, details, tags, provenance):
        return {"id": identifier, "title": title, "kicker": kicker, "description": description,
                "details": details, "tags": tags, "evidence": provenance}

    common_context = f"{int(context):,}".replace(",", " ") + " heures" if context is not None else "selon la zone"
    residual_recipe = f"{int(trees)} arbres · profondeur {int(depth)}" if trees is not None and depth is not None else "Recette propre à chaque zone"
    nodes = [
        node("sources", "Les signaux du marché", "ENTRÉES", "Prix historiques, charges résiduelles prévues et production nucléaire française prévue.",
             ["Les charges résiduelles couvrent FR, DE, BE, NL et ES.", "Le signal nucléaire est toujours français, même pour une prévision BE, DE ou NL.", "La production nucléaire prévue entre dans le modèle ; les observations de la livraison ne sont pas des entrées."],
             ["Prix", "5 charges résiduelles", "Nucléaire FR"],
             [evidence("run_nuclear_forecast.py", "RESIDUAL_ALIASES =", "Charges résiduelles"), evidence("run_nuclear_forecast.py", "covariates[NUCLEAR_ALIAS] =", "Signal nucléaire prévu")]),
        node("pit", "Les données au bon instant", "POINT-IN-TIME", "Chaque journée utilise les informations disponibles à son origine de prévision.",
             ["Les prévisions sources sont sélectionnées au cutoff D−1 à 08:00, heure de Paris.", "Les instantanés isolés figent les entrées et gardent leurs références.", "La livraison respecte ses heures physiques : 23, 24 ou 25 heures selon le changement d'heure."],
             ["D−1 · 08:00 Paris", "Vintages figés"],
             [evidence("run_nuclear_forecast.py", "cutoff = (day -", "Origine de prévision"), evidence("chronos2_hourly/nuclear_forecast.py", "Raw forecast origins must equal", "Contrôle causal"), evidence("chronos2_hourly/model_storm_data.py", "expected = local_delivery_day_index", "Heures physiques")]),
        node("transformer", "Chronos-2", "TRANSFORMER", "Le modèle de fondation transforme les séries temporelles et les covariables en prévisions quantiles.",
             [f"Modèle de base confirmé : {model_id or 'non disponible'}. Contexte commun : {common_context}.", "Le nucléaire entre à la fois dans le contexte historique et dans les covariables futures connues.", "Les poids sont chargés localement. Le parcours nucléaire utilise Chronos-2 sans adaptateur LoRA."],
             ["Transformer", "Contexte " + common_context, "Quantiles"],
             [evidence("chronos2_modular/forecasting.py", "from chronos import Chronos2Pipeline", "Chargement de Chronos-2"), evidence("chronos2_hourly_fr_residual_v1.yaml", "context_length:", "Contexte configuré"), evidence("chronos2_hourly/nuclear_forecast.py", "Nuclear must enter both", "Contexte et futur nucléaire"), evidence("run_nuclear_forecast.py", "def check_lora_inactive", "Contrôle LoRA")]),
        node("residual", "Le correcteur résiduel", "CATBOOST", "Il apprend les erreurs passées de Chronos-2 et ajuste sa prévision.",
             [f"Recette commune confirmée : {residual_backend or 'non disponible'}. {residual_recipe}.", "Le correcteur est réajusté chaque jour sur les 365 jours civils strictement antérieurs.", "Ses variables comprennent le calendrier, les profils journaliers et la prévision nucléaire."],
             [residual_backend or "Recette à vérifier", "365 jours", "Refit quotidien"],
             [evidence("chronos2_hourly_fr_residual_v1.yaml", "  residual_correction:", "Recette du correcteur"), evidence("chronos2_hourly/nuclear_forecast.py", "def causal_residual_replay", "Replay causal"), evidence("chronos2_hourly/nuclear_forecast.py", '"residual_fit_rule"', "Fenêtre du refit")]),
        node("kalman", "L'ajustement Kalman", "FILTRE GOUVERNÉ", "Le filtre adapte la prévision corrigée aux erreurs récentes et aux variables de marché.",
             ["Il reçoit residual_corrected et produit residual_kalman, sans changer l'ordre de la chaîne.", "Les candidats couvrent biais, structure horaire, variables de marché et échelle ; leur utilisation est gouvernée.", "Le nucléaire est aussi une variable du filtre. Les refits utilisent 365 jours antérieurs."],
             ["Biais · marché · échelle", "365 jours", "Nucléaire inclus"],
             [evidence("chronos2_hourly/nuclear_forecast.py", "def nuclear_kalman_covariate_config", "Covariable nucléaire du filtre"), evidence("chronos2_hourly/nuclear_forecast.py", 'upstream_model="residual_corrected"', "Chaîne résiduel → Kalman"), evidence("config/kalman_operational.yaml", "  candidate_kinds:", "Candidats gouvernés")]),
        node("forecast", "La prévision finale", "NUCLEAR KALMAN", "Une trajectoire centrale et son intervalle de prévision, heure par heure et par pays.",
             ["Les sorties exposent les quantiles q10, q50 et q90 ; q50 est la prévision centrale.", "Les exports nuclear_kalman sont distincts des variantes historiques.", "Le mode incrémental réutilise les jours déjà validés ; il ne simplifie pas les fenêtres scientifiques."],
             ["q10 · q50 · q90", "BE · DE · FR · NL", "Exports figés"],
             [evidence("chronos2_hourly/nuclear_forecast.py", "QUANTILES =", "Quantiles produits"), evidence("run_nuclear_kalman.py", "--report-variants", "Variante nucléaire Kalman"), evidence("config/nuclear_forecast.yaml", "computation_mode:", "Calcul incrémental")]),
        node("reference", "Storm & observations", "RÉFÉRENCES INDÉPENDANTES", "Storm et les prix observés servent à évaluer la prévision lorsqu'ils sont disponibles.",
             ["Storm ne nourrit ni Chronos-2, ni le correcteur, ni le filtre Kalman dans ce parcours.", "Les observations de la livraison servent au reporting après leur disponibilité.", "Une source absente ou incomplète reste explicitement indisponible."],
             ["Comparaison", "Hors entrées du modèle"],
             [evidence("chronos2_hourly/nuclear_forecast.py", '"storm_used_as_input": False', "Storm exclu des entrées"), evidence("chronos2_hourly/model_storm_data.py", "def load_model_storm_payload", "Chargement des comparateurs"), evidence("chronos2_hourly/model_storm_data.py", '"Storm unavailable"', "Couverture des sources")]),
        node("comparison", "Les résultats en perspective", "RAPPORTS & ÉVALUATION", "Le rapport CWE aligne modèle, Storm et observations sur la même livraison.",
             ["Le lanceur regroupe les résultats BE, DE, FR et NL après les calculs par pays.", "Les rapports nucléaires conservent 365 jours d'évaluation ; le rapport CWE regroupe la livraison choisie.", "Les comparaisons doivent tenir compte de la période, de la cible, de l'horizon et de la couverture."],
             ["Rapport CWE", "Périmètres documentés", "Aucun classement automatique"],
             [evidence("run_nuclear_kalman.py", 'steps.append(RunStep("CWE_Model_Storm"', "Rapport final du batch"), evidence("chronos2_hourly/model_storm_data.py", "aligned = {key:", "Alignement par heure"), evidence("chronos2_hourly/nuclear_forecast.py", "EVALUATION_DAYS =", "Historique évalué")]),
    ]
    return {
        "title": "Architecture de NYX",
        "model_name": "NYX",
        "subtitle": "NYX, le modèle composite du parcours Nuclear Kalman configuré dans votre projet.",
        "description": "NYX est le nom du modèle composite qui associe Chronos-2, un correcteur résiduel CatBoost et un filtre Kalman gouverné. Ce nom de présentation conserve les identifiants techniques du parcours Nuclear Kalman et des résultats historiques.",
        "pipeline": "nuclear_kalman",
        "nodes": nodes,
        "edges": [
            {"source": "sources", "target": "pit", "label": "Sources disponibles", "kind": "main"},
            {"source": "pit", "target": "transformer", "label": "Contexte + futur connu", "kind": "main"},
            {"source": "transformer", "target": "residual", "label": "Prévision Chronos", "kind": "main"},
            {"source": "residual", "target": "kalman", "label": "residual_corrected", "kind": "main"},
            {"source": "kalman", "target": "forecast", "label": "residual_kalman", "kind": "main"},
            {"source": "forecast", "target": "comparison", "label": "Quantiles publiés", "kind": "main"},
            {"source": "reference", "target": "comparison", "label": "Références disponibles", "kind": "reference"},
            {"source": "pit", "target": "residual", "label": "Calendrier + covariables", "kind": "context"},
            {"source": "pit", "target": "kalman", "label": "Marché + nucléaire", "kind": "context"},
        ],
        "inner_transformer": [
            {"id": "patches", "title": "Segments temporels", "description": "Les valeurs, leurs masques et leur position temporelle sont assemblés en segments puis projetés dans une représentation interne."},
            {"id": "time_attention", "title": "Attention temporelle", "description": "L'attention relie les segments d'une série pour exploiter son contexte temporel."},
            {"id": "group_attention", "title": "Attention de groupe", "description": "Une attention supplémentaire opère sur les séries regroupées et les covariables, selon le masque fourni au modèle."},
            {"id": "feed_forward", "title": "Transformation interne", "description": "Un réseau feed-forward transforme les représentations après les deux opérations d'attention."},
            {"id": "quantiles", "title": "Projection des quantiles", "description": "La projection de sortie reconstruit les segments de prévision par quantile."},
        ] if transformer_verified else [],
        "inner_transformer_source": {"label": "Implémentation du paquet Chronos installé : chronos/chronos2/model.py", "verified": transformer_verified},
        "facts": {
            "model_id": model_id, "zones": list(ZONES), "context_hours": context,
            "context_hours_by_zone": contexts, "residual_backend": residual_backend,
            "residual_trees": trees, "residual_depth": depth, "residual_enabled_by_zone": residual_enabled,
            "lora_enabled_by_zone": lora_by_zone, "lora_active": lora_active,
            "lora_nuclear_compatible": False, "computation_mode": mode,
            "kalman_governance_days": numeric(nested(kalman, "filter_parameters", "governance_lookback_days")),
            "residual_refit_days": 365, "kalman_refit_days": 365, "evaluation_days": 365,
            "nuclear_signal_country": "FR", "storm_used_as_input": False,
            "scope": "Parcours nucléaire dédié ; configurations historiques distinctes préservées.",
        },
        "badges": [
            {"label": "Moteur Transformer", "value": "Chronos-2" if model_id else "À vérifier"},
            {"label": "Contexte", "value": common_context},
            {"label": "LoRA", "value": "Inactif" if lora_active is False else "Incompatible" if lora_active else "Inconnu"},
            {"label": "Calcul", "value": "Incrémental" if mode == "incremental" else "Complet" if mode == "full" else "Inconnu"},
        ],
        "warnings": list(dict.fromkeys(warnings)),
    }

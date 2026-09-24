"""One-off source migration for the requested results-first console UI."""
from pathlib import Path
p = Path(__file__).resolve().parents[1] / 'experiment_console/static/app.js'
s = p.read_text(encoding='utf-8')
s = s.replace("tab:'logs'", "tab:'results'").replace("state.tab='logs'", "state.tab='results'")
s = s.replace("[['logs','Journal en direct'],['results','Résultats']", "[['results','Résultats']")
start = s.index("if(state.tab==='logs'){content.innerHTML=")
end = s.index("else if(state.tab==='config')", start)
s = s[:start] + "if(state.tab==='config')" + s[end + len("else if(state.tab==='config')"):]
start = s.index('async function loadLogs()')
end = s.index('async function loadChart(', start)
s = s[:start] + s[end:]
s = s.replace("if(state.tab==='logs')await loadLogs();\n        else if(state.tab==='results'&&resultsChanged)", "if(state.tab==='results'&&resultsChanged)")
s = s.replace("if(e.target.id==='log-search')filterLogs();", '')
s = s.replace("if(t.id==='log-level'||t.id==='log-auto')filterLogs();", '')
s = s.replace("const artifacts=r.artifacts||[];", "const artifacts=(r.artifacts||[]).filter(a=>a.kind!=='log');")
s = s.replace("'Les métriques apparaîtront lorsque le pipeline les aura enregistrées.'", "'Consultez les rapports disponibles ci-dessous. Ce run ne publie pas de métriques structurées.'")
s = s.replace("Configuration copiée, commande, version Git, logs et résultats réunis pour chaque expérience.", "Configuration copiée, version Git et résultats réunis pour chaque expérience.")
s = s.replace("state.route=location.hash.slice(1)||'dashboard'", "state.route=location.hash.slice(1)||'architecture'")
s = s.replace("{dashboard:'Vue d’ensemble',history:", "{architecture:'Architecture du modèle',dashboard:'Résultats',history:")
s = s.replace("if(section==='new')renderNew();", "if(section==='architecture')await renderArchitecture();else if(section==='new')renderNew();")
s = s.replace("pageHead('PILOTAGE DES EXPÉRIENCES','Vue d’ensemble','Vos exécutions, du lancement aux résultats.'", "pageHead('EXPLORATION DES EXPÉRIENCES','Vos résultats','Prévisions, rapports et évaluations réunis au même endroit.'")
# Raw stdout belongs to backend diagnosis, not the results-oriented surface.
s = s.replace("${esc(r.activity||'Aucune étape publiée')}", "${esc(active(r)?'Le calcul est supervisé en arrière-plan.':r.status==='failed'?'Le calcul a échoué. Consultez les artefacts disponibles et sa configuration.':r.status==='succeeded'?'Les résultats sont disponibles ci-dessous.':'Le statut provient des métadonnées disponibles.')}")
p.write_text(s, encoding='utf-8')

# Handmatige controle van de antwoordevaluatie (vragenset v2)

Gecontroleerd tegen de geciteerde bronfragmenten in
[answer_eval_dense.json](answer_eval_dense.json) en [answer_eval_rerank.json](answer_eval_rerank.json).
De steekproef bevat alle negatieve juryoordelen (5), zes willekeurige positieve oordelen
(dense, `random.seed(42)`), alle beantwoorde onbeantwoordbare vragen (8) en de enige
foute citatie (1).

## Negatieve juryoordelen

| Vraag | Config | Jury | Controle |
|---|---|---|---|
| q009 · zorgverzekering opzeggen | dense | niet onderbouwd | **Jury te streng.** Alle beweringen staan in bron 1 of 2, ook "tot 1 februari". De jury eist een voorwaarde die het antwoord niet hoeft te herhalen. Hetzelfde patroon zat in v1 (q033). |
| q009 · zorgverzekering opzeggen | rerank | niet onderbouwd | Jury terecht. Het antwoord noemt "vóór 1 januari" strenger dan "uiterlijk 31 december"; dat is dezelfde deadline. |
| ga027 · woningruil huurprijscheck | dense | niet onderbouwd | Jury terecht. Bron 2 zegt dat sociale huur, kamers en middenhuur de check ook later mogen doen; het antwoord maakt van 6 maanden een algemene regel. |
| q037 · huurverhoging vrije sector | rerank | niet onderbouwd | Jury terecht. "Het maximum van 4,4% geldt tot 1 mei 2029" klopt niet: tot 2029 geldt de regeling, het percentage alleen voor 2026. |
| ga017 · zorgverzekering kind | rerank | niet onderbouwd | Jury grotendeels terecht. De slotzin zegt dat de bron geen toestemmingsregel noemt, terwijl de bron juist toestemmingsregels geeft. Randgeval. |

## Positieve juryoordelen (steekproef van 6)

q018, q004, q039, q035, q032 en q021: in alle zes de gevallen staan de kernbeweringen in de
geciteerde bron en bevat een geciteerd fragment het gelabelde citaat. Geen fouten gevonden.

## Wat de metrieken missen

- **Zachte weigeringen tellen als fout.** Alle 8 beantwoorde onbeantwoordbare vragen (dense:
  gu001, gu027; rerank: u008, gu001, gu004, gu010, gu027, gu028) zeggen in de eerste zin dat de
  bronnen het gevraagde niet noemen, en geven daarna verwante informatie. De jury vindt ze
  terecht onderbouwd, maar de status is `answered`, dus de abstentiescore telt ze als gemist.
  De score "terecht geweigerd" is daardoor een ondergrens. Ik heb de metriek niet achteraf
  aangepast; dat zou de meting bijsturen op de uitkomst.
- **Onderbouwd is niet hetzelfde als een antwoord.** q042 (dense) is volgens de jury onderbouwd,
  maar gaat over de rechter en een cv-ketel ouder dan 10 jaar, niet over wie normaal een nieuwe
  ketel betaalt. Citatie-correctheid vangt dit wel: geen geciteerd fragment bevat het label.
- **Ongeldige uitvoer.** 3 (dense) en 4 (rerank) antwoorden hadden een ongeldig formaat en
  kregen het vaste terugvalantwoord. Die tellen als gemiste antwoorden, niet als weigering.

Conclusie: van de 11 gecontroleerde juryoordelen kloppen er 10. De jury is soms te streng,
nooit te mild in deze steekproef. De steekproef is klein; dat blijft een grens van deze meting.

# Handmatige controle van de antwoordevaluatie

De controle gebruikt de opgeslagen tekstfragmenten en het antwoordrapport op commit
`28d48ce`. De voorbeelden omvatten alle negatieve juryoordelen, de gevallen waarin
de antwoordstatus afweek van het label, en antwoorden over verschillende onderwerpen.
Het [ruwe rapport](answer_eval.json) bewaart de oorspronkelijke juryoordelen.

| Vraag | Bevinding |
|---|---|
| q003 · huisarts | De aangehaalde pagina’s ondersteunen het antwoord op de vraag. De jury leest een opsomming als een indeling van verschillende zorgsoorten onder huisartsenzorg. In het antwoord staat ‘net als’, wat afzonderlijke voorbeelden aanduidt. De zin over aanvullend onderzoek en geneesmiddelen is onhandig geformuleerd. |
| q011 · huurverhoging | De tabel bevat de inkomensgrens voor het eenpersoonshuishouden en het percentage bij deze huur. De berekening met de opgegeven huur klopt. De bron vermeldt ook het plafond volgens het puntenaantal. |
| q020 · studentenreisproduct | Het verwachte fragment stond op rang 2. De generator leverde ongeldige uitvoer en de API gaf het vaste terugvalantwoord. |
| q022 · studieschuld | De bron noemt het inkomen van twee jaar geleden als uitgangspunt en het verzoek om het voorgaande of huidige jaar te gebruiken bij een plotselinge inkomensdaling. Het antwoord behoudt dat onderscheid. |
| q028 · kinderopvangtabel | Het geciteerde fragment bevat zowel de kop voor het eerste kind als de juiste inkomensschijf. Het antwoord herhaalt de uitkomst. |
| q033 · zorgverzekering overstappen | ‘Uiterlijk 31 december’ en ‘vóór 1 januari’ geven dezelfde opzegdeadline. Beide bronnen ondersteunen het antwoord. Het negatieve oordeel lijkt gebaseerd op een eis van letterlijke overeenkomst. |
| q041 · huur opzeggen | De bron bevestigt opzegging per e-mail bij ontvangstbevestiging en noemt de gebruikelijke opzegtermijn. De extra bewering over geldigheid bij overschrijding van die termijn heeft geen grond in het geciteerde fragment. |
| q042 · cv-ketel | Dense haalde het gelabelde fragment niet op. BM25 vond het op rang 4. Het fragment noemt expliciet wie groot onderhoud en vervanging van de cv-ketel betaalt. |
| u006 · bedrag kinderopvangtoeslag | Het antwoord zegt dat het exacte bedrag onbekend is en noemt algemene regels. De status is `answered`, waardoor de evaluatie dit telt als een gemiste onthouding. |

De jury scoort brononderbouwing. Een onderbouwd antwoord kan overbodige details bevatten
of maar een deel van de vraag beantwoorden. De handmatige controle maakt die verschillen
zichtbaar voor de getoonde gevallen.

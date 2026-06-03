# Montage- & omgevingschecklist — OAK-D Lite kabel-inspectie

> Doel: een betrouwbaar, afstand-onafhankelijk meetpunt voor de PoC
> (`cable_inspection_poc.py`) in een natte, maritieme kade-omgeving.
>
> **Belangrijk:** dit is een PoC / monitoringslaag. `trigger_emergency_stop()`
> is GEEN gecertificeerde veiligheidsfunctie (geen SIL/PL). De echte noodstop
> hoort in een safety-rated besturing (zie certificering-advies).

---

## 0. Veiligheid vooraf (vóór je iets monteert)
- [ ] Werkvergunning / LMRA voor werken nabij bewegende lier/gantry en water.
- [ ] Afstemming met de partij die de machine bedient — nooit alleen werken
      bij een lopende 7 km-kabel.
- [ ] Val- en verdrinkingsbeveiliging bij montage boven/naast water.
- [ ] Spanningsloos / vergrendeld waar elektrisch werk nodig is (LOTO).

## 1. Meetpunt kiezen (locatie nog niet bepaald)
Kies een punt dat zoveel mogelijk hieraan voldoet:
- [ ] **Kabel loopt er voorspelbaar en ~recht** langs (bij/na een geleiderrol),
      niet midden in een vrije overhang waar hij wild slingert.
- [ ] **Loodrechte zichtlijn** op de kabel-as mogelijk op **0,5–0,8 m**
      (sweet spot ~0,6–0,7 m). Niet dichterbij dan ~0,35 m (stereo + focus).
- [ ] **Stabiele, trillingsarme montagebasis** in de buurt (vast frame,
      niet een meeschommelend deel van de gantry).
- [ ] **Bereikbaar voor onderhoud** (lens schoonmaken!) zonder hoogwerker
      elke keer.
- [ ] **Beschermd tegen directe regen/spray en tegenlicht** of te beschermen
      met een kap.
- [ ] USB-C-kabelroute naar de host (MacBook/industrie-PC) **< ~1,5 m**
      passief, of actieve repeater/hub bij langere afstand.

> Het instappunt op de kade is vaak stabieler en beter benaderbaar dan een
> punt hoog op de transportbrug — weeg onderhoud mee, niet alleen zicht.

## 2. Camera-positie & uitlijning
- [ ] Camera **loodrecht** op de kabel-as (zowel horizontaal als verticaal),
      kabel ~horizontaal door beeld (matcht de aanname in de diktemeting).
- [ ] Kabel in het **midden** van beeld → binnen de `y_center`-band
      (config: 0,25–0,75 van de hoogte).
- [ ] Afstand binnen het Z-venster (config: `z_min=0,40 m`, `z_max=1,20 m`).
- [ ] **Scherpte gecontroleerd in het live-venster** (FF-lens: scherp vanaf
      ~30 cm; te dichtbij = wazig = slechte randdetectie).
- [ ] Stereo-baseline (de twee mono-camera's) **vrij zicht** op de kabel —
      niets dat één oog blokkeert, anders gaten in de depth.

## 3. Behuizing & maritieme bescherming
- [ ] **IP65+ behuizing** (OAK-D Lite heeft géén IP-rating; nat + zout).
- [ ] **Anti-condens / verwarmd kijkvenster** of droogmiddel tegen aanslag.
- [ ] Corrosiebestendige beugel/bevestiging (RVS A4 bij zoutwater).
- [ ] **Trillingsdemping** in de beugel (gantry/lier trilt → ruis op flow
      én randen).
- [ ] Kabeldoorvoer met **wartel + kabelontlasting** (USB-C niet op trek).
- [ ] Zonnescherm/kap tegen direct daglicht en regenspray op het venster.

## 4. Verlichting (kritiek voor de diktemeting)
De diktemeting hangt op Canny-randdetectie — licht maakt of breekt het.
- [ ] **Eigen, diffuse verlichting** bij het meetpunt i.p.v. afhankelijk van
      wisselend daglicht/bewolking.
- [ ] **Geen tegenlicht** (zon/water-reflecties) richting de lens.
- [ ] Reflecties op natte/glimmende kabel beperken (gepolariseerd licht of
      hoek vermijden waaronder het water spiegelt).
- [ ] Lichtniveau zo constant mogelijk dag/nacht.

## 5. Host & software-bring-up
- [ ] `pip install depthai opencv-python numpy` op de host.
- [ ] Camera detecteert: `python cable_inspection_poc.py` start zonder errors.
- [ ] **Intrinsics-log gecontroleerd** (regel `[CALIB] ... fx= fy=`) — dit
      bevestigt dat de fabriekskalibratie uit de chip is gelezen.
- [ ] USB-bus stabiel (USB3 indien mogelijk; geen brown-outs op een hub).

## 6. Kalibratie & tuning op locatie
- [ ] **Baseline-diameter**: laat de auto-kalibratie (eerste ~60 frames) op
      een SCHOON, representatief kabelstuk lopen, of zet
      `nominal_diameter_mm` vast op de bekende diameter.
- [ ] **Diktemeting cross-checken** met een handmeting (schuifmaat/meetlint)
      bij een paar bekende Z-afstanden → klopt de mm-uitkomst?
- [ ] **Z-venster** (`z_min`/`z_max`) afstemmen op de werkelijke doorhang.
- [ ] **`y_center`-band** afstemmen op waar de kabel echt in beeld zit.
- [ ] **Canny-drempels** (`canny_low/high`) bijstellen tot de randen stabiel
      zijn bij jouw verlichting (test bij nat én droog).
- [ ] **Snelheid valideren**: gemeten cm/s vs. bekende lijnsnelheid
      (~16,6 cm/s) — schaal/uitlijning controleren.
- [ ] Alarm-test: simuleer dikke tape/birdcage en te slap/strak → klopt
      ORANJE/ROOD?

## 7. Operationele afronding
- [ ] Lens-reinigingsschema (zout/spray → vervuiling = meetdrift).
- [ ] Logging/telemetrie (`send_mqtt_update`) naar het dashboard getest.
- [ ] Duidelijk gedocumenteerd dat de noodstop-haak een PLACEHOLDER is en
      de echte stop via de gecertificeerde besturing loopt.
- [ ] Onderhouds-/herkalibratie-interval afgesproken.

---

## Niet door een ongetrainde operator (herhaling van het advies)
- Fysiek ophangen + scherpte checken: kan een handige technicus.
- Uitlijning, tuning, en zeker de **veiligheidsintegratie + certificering**
  (Machinerichtlijn/CE, ISO 13849 / IEC 62061, evt. ATEX, scheepsklasse):
  **gekwalificeerde partijen**.

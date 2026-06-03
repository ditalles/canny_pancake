# Positie-eisen voor het meetpunt (samen op te stellen)

De ideale cameraplek staat nog niet vast. Dit document legt de eisen vast
waaraan een goede plek voldoet, plus wat ik van jou nodig heb om de positie
definitief te kiezen. Monitoring-only: we kiezen puur op meetkwaliteit en
onderhoudbaarheid, niet op machinebesturing.

## Harde eisen (meetkwaliteit)
1. **Loodrechte zichtlijn** op de kabel-as, afstand **0,5–0,8 m** (sweet spot
   ~0,6–0,7 m). Niet < ~0,35 m (stereo-min + fixed-focus scherpte).
2. **Kabel ~horizontaal** door beeld en in het midden (binnen de y-band 0,25–0,75).
   De diktemeting gaat uit van een horizontale kabel.
3. **Beide stereo-ogen vrij zicht** op de kabel (anders gaten in de Z → geen
   catenary-correctie).
4. **Afstand binnen het Z-venster** (0,40–1,20 m) bij normale doorhang;
   anders past het venster niet bij de werkelijke catenary.

## Belangrijke eisen (betrouwbaarheid / robuustheid)
5. **Rustige, contrasterende achtergrond** achter de kabel (lucht, een vlakke
   plaat, of het water in plaats van een drukke constructie). Dit maakt de
   randdetectie veel stabieler. Eventueel een eenvoudige achtergrondplaat
   plaatsen.
6. **Trillingsarme, vaste montagebasis** (geen meeschommelend deel van de gantry).
7. **Beschermd tegen direct tegenlicht / waterreflecties** in de lens-as
   (zon-arc en spiegeling van het water vermijden).
8. **Bereikbaar voor onderhoud** (lens schoonmaken zonder hoogwerker).
9. **Voeding aanwezig** en USB-route < ~1,5 m passief (anders actieve repeater).

## Wat ik van jou nodig heb om de positie te kiezen
- [ ] 2–3 **foto's vanaf kandidaat-plekken** richting de kabel (ongeveer op de
      beoogde camerahoogte/afstand), liefst bij verschillend weer/licht.
- [ ] De **echte kabeldiameter** (mm) en globale **doorhang-variatie** (hoeveel
      cm beweegt de kabel dichterbij/verder tijdens normaal bedrijf?).
- [ ] Wat er als **montagebasis** beschikbaar is (gantry-poot, pijler, mast) en
      of er **voeding** dichtbij zit.
- [ ] Of de kabel op een vast punt **kortstondig stilstaat of altijd beweegt**
      (beïnvloedt de odometer-validatie).

Met die input stel ik de config-waarden concreet voor (`z_min/z_max`,
`y_center`-band, Canny-drempels, baseline) en bevestigen we samen de plek.

## Eerst testen zonder camera
Je kunt het hele dashboard nu al bekijken met de simulator, zodat we de
visualisatie en alarmlogica kunnen afstemmen vóór montage:

```
python run_monitor.py --simulate     # open http://localhost:5006
```
